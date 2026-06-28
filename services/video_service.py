from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
import threading
import time

import moviepy
import numpy as np
import psutil
import requests
from moviepy import (
    AudioClip,
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)
from PIL import Image

from config import Settings
from core.models import Script, Segment
from core.retry import retry_call
from services.caption_service import CaptionService
from storage import RunPaths


def _get_container_memory() -> tuple[int, int | None, float]:
    # 1. Try Cgroup v2
    try:
        curr_path = Path("/sys/fs/cgroup/memory.current")
        max_path = Path("/sys/fs/cgroup/memory.max")
        if curr_path.exists():
            used = int(curr_path.read_text().strip())
            max_val = max_path.read_text().strip()
            limit = int(max_val) if max_val.isdigit() else None
            percent = (used / limit) * 100.0 if limit else 0.0
            return used, limit, percent
    except Exception:
        pass

    # 2. Try Cgroup v1
    try:
        usage_path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        limit_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if usage_path.exists():
            used = int(usage_path.read_text().strip())
            limit = int(limit_path.read_text().strip())
            if limit and limit < 9223372036854771712:
                percent = (used / limit) * 100.0
                return used, limit, percent
            else:
                return used, None, 0.0
    except Exception:
        pass

    # 3. Fallback to host/process memory (local development)
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used, mem.total, mem.percent
    except Exception:
        pass

    return 0, None, 0.0


class MemoryMonitor(threading.Thread):
    def __init__(self, logger: logging.Logger, interval: int = 15):
        super().__init__()
        self.logger = logger
        self.interval = interval
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            try:
                used, limit, percent = _get_container_memory()
                used_mb = used / (1024 * 1024)
                limit_str = f"{limit / (1024 * 1024):.2f} MB" if limit else "Unlimited"
                percent_str = f"{percent:.1f}%" if limit else "N/A"
                
                self.logger.info(
                    "[Memory Monitor] Container Memory Used: %.2f MB | Limit: %s | Percent: %s",
                    used_mb, limit_str, percent_str
                )
            except Exception as e:
                self.logger.warning("[Memory Monitor] Failed to log memory: %s", e)
            
            for _ in range(self.interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1.0)

    def stop(self):
        self.stop_event.set()


class VideoService:
    """Renders a vertical Shorts video from generated images, audio, and captions."""

    def __init__(self, settings: Settings, logger: logging.Logger, caption_service: CaptionService) -> None:
        self.settings = settings
        self.logger = logger
        self.caption_service = caption_service

    def _check_memory_and_optimize(self, step_label: str) -> bool:
        try:
            used, limit, percent = _get_container_memory()
            used_mb = used / (1024 * 1024)
            limit_str = f"{limit / (1024 * 1024):.2f} MB" if limit else "Unlimited"
            percent_str = f"{percent:.1f}%" if limit else "N/A"
            
            self.logger.info(
                "[%s] Container Memory Check: Used=%.2f MB, Limit=%s, Percent=%s",
                step_label, used_mb, limit_str, percent_str
            )
            
            if limit and percent > 80.0:
                self.logger.warning(
                    "[%s] Memory usage exceeded 80%% threshold (Percent=%s)! Triggering gc.collect()",
                    step_label, percent_str
                )
                gc.collect()
                return True
        except Exception as exc:
            self.logger.warning("Memory check failed: %s", exc)
        return False

    def _deep_close(self, clip) -> None:
        if not clip:
            return
        try:
            if hasattr(clip, "clips") and clip.clips:
                for subclip in list(clip.clips):
                    self._deep_close(subclip)
            if hasattr(clip, "audio") and clip.audio:
                self._deep_close(clip.audio)
                clip.audio = None
            if hasattr(clip, "close"):
                clip.close()
        except Exception as exc:
            self.logger.debug("Failed to close clip: %s", exc)

    def render(
        self,
        script: Script,
        image_paths: list[Path],
        audio_paths: list[Path],
        paths: RunPaths,
    ) -> Path:
        self._create_thumbnail(image_paths, paths.thumbnail_path)
        
        # Start background memory monitoring thread (15 seconds interval)
        monitor = MemoryMonitor(self.logger, interval=15)
        monitor.start()
        
        temp_dir = self.settings.storage_dir / "temp_scenes"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        scene_paths = []
        elapsed = 0.0
        max_seconds = self.settings.shorts_max_seconds
        ass_lines = []

        try:
            # Initial memory check & startup quality adjustment
            if self._check_memory_and_optimize("Startup"):
                orig_res = self.settings.video_resolution
                new_res = (int(orig_res[0] * 0.7), int(orig_res[1] * 0.7))
                orig_fps = self.settings.render_fps
                new_fps = 15 if orig_fps > 15 else 12
                self.logger.warning(
                    "Startup memory usage >80%%. Adjusting render profile: "
                    "Resolution: %s -> %s | FPS: %s -> %s",
                    orig_res, new_res, orig_fps, new_fps
                )
                object.__setattr__(self.settings, "video_resolution", new_res)
                object.__setattr__(self.settings, "render_fps", new_fps)
                gc.collect()

            for index, segment in enumerate(script.segments, start=1):
                if elapsed >= max_seconds:
                    self.logger.info("Reached %ss Shorts limit; skipping remaining scenes", max_seconds)
                    break
                if index > len(audio_paths) or index > len(image_paths):
                    self.logger.warning("Missing assets for scene %s; skipping", index)
                    continue

                audio_clip = AudioFileClip(str(audio_paths[index - 1]))
                audio_clip = self._with_volume(audio_clip, self.settings.voice_volume)
                remaining = max_seconds - elapsed
                duration = min(float(audio_clip.duration), remaining)
                if duration < 0.45:
                    audio_clip.close()
                    break
                if duration < audio_clip.duration:
                    audio_clip = self._subclip(audio_clip, 0, duration)

                self.logger.info("Scene render started: %s/%s (%.2fs)", index, len(script.segments), duration)
                visual_clip = self._image_clip(image_paths[index - 1], duration, index)
                overlays = self._caption_clips(segment, duration, elapsed, max_seconds)
                if overlays:
                    composed = CompositeVideoClip([visual_clip, *overlays], size=self.settings.video_resolution, use_bgclip=True)
                else:
                    composed = visual_clip
                composed = self._with_duration(composed, duration)
                composed = self._with_audio(composed, audio_clip)
                composed = self._apply_video_transitions(composed)
                
                scene_path = temp_dir / f"scene_{index}.mp4"
                self.logger.info("Writing individual scene %s to %s", index, scene_path)
                composed.write_videofile(
                    str(scene_path),
                    fps=self.settings.render_fps,
                    codec="libx264",
                    audio_codec="aac",
                    preset=self.settings.ffmpeg_preset,
                    logger=None,
                )
                
                # Release memory immediately
                self._deep_close(composed)
                gc.collect()
                
                self.logger.info("Scene render completed: %s/%s", index, len(script.segments))
                scene_paths.append(scene_path)

                # Gather ASS captions for this segment
                chunks = self.caption_service.chunks(segment.subtitle or segment.text)
                if chunks:
                    chunk_duration = duration / len(chunks)
                    for chunk_idx, chunk in enumerate(chunks):
                        start_time = chunk_idx * chunk_duration
                        end_time = duration if chunk_idx == len(chunks) - 1 else (chunk_idx + 1) * chunk_duration
                        
                        start_global = elapsed + start_time
                        end_global = elapsed + end_time
                        
                        dialogue_lines = self.caption_service.format_ass_dialogue(chunk, start_global, end_global)
                        ass_lines.extend(dialogue_lines)

                elapsed += duration
                
                # Log memory after each scene
                self._check_memory_and_optimize(f"Post-Scene {index}")

            if not scene_paths:
                raise RuntimeError("No valid scenes were generated; cannot render video")

            # Write ASS subtitle file
            ass_path = temp_dir / "subtitles.ass"
            ass_content = self.caption_service.generate_ass_header() + "\n".join(ass_lines)
            ass_path.write_text(ass_content, encoding="utf-8")

            # Final concatenation using FFmpeg concat demuxer
            self.logger.info("Final merge started")
            list_path = temp_dir / "concat_list.txt"
            concat_content = "\n".join([f"file '{str(p.resolve()).replace('\\', '/')}'" for p in scene_paths])
            list_path.write_text(concat_content, encoding="utf-8")
            
            merged_raw_path = temp_dir / "merged_raw.mp4"
            import subprocess
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                str(merged_raw_path)
            ]
            self.logger.info("Running FFmpeg concat: %s", " ".join(concat_cmd))
            subprocess.run(concat_cmd, check=True, capture_output=True)
            
            # Mix background music
            import random
            audio_dir = self.settings.audio_dir
            music_candidates = []
            if audio_dir.exists():
                for p in audio_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in (".mp3", ".wav") and not p.name.lower().startswith("scene_"):
                        music_candidates.append(p)
            
            # Don't Repeat Music: Load/Save history from storage/last_music.txt
            last_music_path = self.settings.storage_dir / "last_music.txt"
            last_music_name = ""
            if last_music_path.exists():
                try:
                    last_music_name = last_music_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            
            filtered_candidates = [m for m in music_candidates if m.name != last_music_name]
            if not filtered_candidates and music_candidates:
                filtered_candidates = music_candidates
            
            # Select background music track
            selected_music = None
            if filtered_candidates:
                selected_music = random.choice(filtered_candidates)
            else:
                default_music_path = audio_dir / "background_music.mp3"
                if not default_music_path.exists() or default_music_path.stat().st_size == 0:
                    self._download_background_music(default_music_path)
                if default_music_path.exists() and default_music_path.stat().st_size > 0:
                    selected_music = default_music_path
            
            if selected_music:
                try:
                    last_music_path.parent.mkdir(parents=True, exist_ok=True)
                    last_music_path.write_text(selected_music.name, encoding="utf-8")
                except Exception:
                    pass
                self.logger.info("Selected background music track: %s", selected_music.name)
            else:
                self.logger.warning("No background music track is available; proceeding with narration only")
                
            final_video_path = paths.video_path
            
            # Formatting paths for FFmpeg subtitles filter on Windows
            fonts_dir_str = str(Path(self.settings.base_dir) / "assets" / "fonts").replace("\\", "/")
            ass_path_str = str(ass_path).replace("\\", "/")
            escaped_ass_path = ass_path_str.replace(":", "\\:")
            escaped_fonts_dir = fonts_dir_str.replace(":", "\\:")
            
            # Adaptive Fade In and Fade Out
            total_duration = elapsed
            fade_duration = min(2.0, total_duration * 0.05)
            fade_out_start = total_duration - fade_duration
            
            if selected_music and selected_music.exists() and selected_music.stat().st_size > 0:
                bgm_base_volume = max(0.35, self.settings.background_music_volume * 5.0)
                filter_complex_str = (
                    f"[0:a]volume={self.settings.voice_volume},asplit=2[voice1][voice2];"
                    f"[1:a]volume={bgm_base_volume:.3f},"
                    f"afade=t=in:ss=0:d={fade_duration:.3f},"
                    f"afade=t=out:ss={fade_out_start:.3f}:d={fade_duration:.3f}[bg_music];"
                    f"[bg_music][voice1]sidechaincompress=threshold=0.15:ratio=4:attack=20:release=150[ducked_bg];"
                    f"[voice2][ducked_bg]amix=inputs=2:duration=first:normalize=0[mixed];"
                    f"[mixed]loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
                )
                mix_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(merged_raw_path),
                    "-stream_loop", "-1",
                    "-i", str(selected_music),
                    "-filter_complex", filter_complex_str,
                    "-vf", f"subtitles='{escaped_ass_path}':fontsdir='{escaped_fonts_dir}'",
                    "-map", "0:v",
                    "-map", "[aout]",
                    "-c:v", "libx264",
                    "-preset", self.settings.ffmpeg_preset,
                    "-c:a", "aac",
                    str(final_video_path)
                ]
                self.logger.info("Running FFmpeg audio mix and burn subtitles: %s", " ".join(mix_cmd))
                subprocess.run(mix_cmd, check=True, capture_output=True)
            else:
                filter_complex_str = f"[0:a]volume={self.settings.voice_volume},loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
                mix_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(merged_raw_path),
                    "-filter_complex", filter_complex_str,
                    "-vf", f"subtitles='{escaped_ass_path}':fontsdir='{escaped_fonts_dir}'",
                    "-map", "0:v",
                    "-map", "[aout]",
                    "-c:v", "libx264",
                    "-preset", self.settings.ffmpeg_preset,
                    "-c:a", "aac",
                    str(final_video_path)
                ]
                self.logger.info("Running FFmpeg burn subtitles and normalize narration: %s", " ".join(mix_cmd))
                subprocess.run(mix_cmd, check=True, capture_output=True)
                
            self.logger.info("Final merge completed: %s", final_video_path)

            # Cleanup temporary scene files
            try:
                import shutil
                shutil.rmtree(str(temp_dir), ignore_errors=True)
                self.logger.info("Temporary scene files deleted successfully")
            except Exception as exc:
                self.logger.warning("Failed to clean up temporary scene files: %s", exc)

        finally:
            monitor.stop()
            monitor.join(timeout=2.0)
            
        return paths.video_path

    def _caption_clips(self, segment: Segment, duration: float, scene_start: float, total_seconds: float) -> list[ImageClip]:
        return []

    def _image_clip(self, image_path: Path, duration: float, index: int):
        target_size = self.settings.video_resolution
        try:
            clip = self._with_duration(ImageClip(str(image_path)), duration)
            clip = self._resize_cover(clip, target_size)
            return self._ken_burns(clip, duration, index, image_path)
        except Exception as exc:
            self.logger.warning("Could not load image %s: %s", image_path, exc)
            return self._with_duration(ColorClip(size=target_size, color=(12, 16, 22)), duration)

    def _ken_burns(self, clip, duration: float, index: int, image_path: Path):
        try:
            target_w, target_h = self.settings.video_resolution
            pan_left_to_right = index % 2 == 0

            # Pre-scale the original image *once* to exactly 1.1x of target resolution using BILINEAR
            # Keep as PIL Image to minimize raw NumPy memory footprint
            with Image.open(image_path) as img:
                source_w, source_h = img.size
                scale = max((target_w * 1.1) / source_w, (target_h * 1.1) / source_h)
                pre_w = math.ceil(source_w * scale)
                pre_h = math.ceil(source_h * scale)
                pre_scaled = img.resize((pre_w, pre_h), Image.Resampling.BILINEAR)
                
                # Crop to center cover at 1.1x target size
                left = (pre_w - int(target_w * 1.1)) // 2
                top = (pre_h - int(target_h * 1.1)) // 2
                pre_scaled_cover = pre_scaled.crop((left, top, left + int(target_w * 1.1), top + int(target_h * 1.1)))

            pre_w, pre_h = pre_scaled_cover.size # Exactly 1.1 * target_w and 1.1 * target_h

            def transform(get_frame, t: float):
                progress = min(1.0, max(0.0, float(t) / max(duration, 0.01)))
                
                # Zoom factor on the pre-scaled image
                # At progress=0, crop a slightly larger box (zoom out)
                # At progress=1, crop a slightly smaller box (zoom in)
                zoom = 1.08 - (0.07 * progress) if index % 2 == 0 else 1.01 + (0.07 * progress)
                
                crop_w = int(target_w * zoom)
                crop_h = int(target_h * zoom)
                
                max_x = pre_w - crop_w
                max_y = pre_h - crop_h
                
                pan_progress = 0.2 + (0.6 * progress)
                if not pan_left_to_right:
                    pan_progress = 1.0 - pan_progress
                    
                x1 = min(max_x, max(0, round(max_x * pan_progress)))
                y1 = min(max_y, max(0, round(max_y * 0.5)))
                
                # Crop and resize in PIL before converting to np.array (reduces memory usage)
                crop = pre_scaled_cover.crop((x1, y1, x1 + crop_w, y1 + crop_h))
                resized = crop.resize((target_w, target_h), Image.Resampling.BILINEAR)
                return np.array(resized)

            return clip.transform(transform, keep_duration=True)
        except Exception as exc:
            self.logger.debug("Ken Burns effect skipped: %s", exc)
            return clip

    def _resize_cover(self, clip, target_size: tuple[int, int]):
        target_w, target_h = target_size
        source_w, source_h = clip.size
        scale = max(target_w / source_w, target_h / source_h)
        new_size = (math.ceil(source_w * scale), math.ceil(source_h * scale))
        clip = clip.resized(new_size=new_size)
        return clip.cropped(x_center=new_size[0] // 2, y_center=new_size[1] // 2, width=target_w, height=target_h)

    def _background_music(self, duration: float):
        music_path = self.settings.audio_dir / "background_music.mp3"
        if not music_path.exists() or music_path.stat().st_size == 0:
            self._download_background_music(music_path)

        if music_path.exists() and music_path.stat().st_size > 0:
            try:
                music = AudioFileClip(str(music_path))
                if music.duration < duration:
                    from moviepy import concatenate_audioclips

                    loops = int(duration / music.duration) + 1
                    music = concatenate_audioclips([music] * loops)
                music = self._subclip(music, 0, duration)
                music = self._with_volume(music, self.settings.background_music_volume)
                return self._apply_audio_fades(music)
            except Exception as exc:
                self.logger.warning("Background music failed; using synthetic bed: %s", exc)
        return self._synthetic_music(duration)

    def _download_background_music(self, music_path: Path) -> None:
        if not self.settings.background_music_url:
            return
        self.logger.info("Downloading cached background music")

        def download() -> None:
            response = requests.get(
                self.settings.background_music_url,
                headers={"User-Agent": "HindiShortsAutomation/1.0"},
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
            music_path.write_bytes(response.content)

        retry_call(
            download,
            attempts=self.settings.retry_attempts,
            backoff_seconds=self.settings.retry_backoff_seconds,
            logger=self.logger,
            label="background music download",
        )

    def _synthetic_music(self, duration: float):
        fps = 44100
        volume = self.settings.background_music_volume

        def make_frame(t):
            t = np.asarray(t)
            fade_in = np.minimum(1.0, t / 2.0)
            fade_out = np.minimum(1.0, np.maximum(0.0, (duration - t) / 2.0))
            envelope = np.minimum(fade_in, fade_out)
            pad = 0.50 * np.sin(2 * np.pi * 110 * t) + 0.28 * np.sin(2 * np.pi * 165 * t)
            shimmer = np.sin(2 * np.pi * 440 * t) * 0.04
            mono = (pad + shimmer) * envelope * volume
            if mono.ndim == 0:
                return np.array([mono, mono])
            return np.column_stack([mono, mono])

        return AudioClip(make_frame, duration=duration, fps=fps)

    def _apply_video_transitions(self, clip):
        try:
            return clip.with_effects([moviepy.vfx.FadeIn(0.12), moviepy.vfx.FadeOut(0.12)])
        except Exception as exc:
            self.logger.debug("Video fade skipped: %s", exc)
            return clip

    def _apply_audio_fades(self, clip):
        try:
            return clip.with_effects([moviepy.afx.AudioFadeIn(1.5), moviepy.afx.AudioFadeOut(1.5)])
        except Exception as exc:
            self.logger.debug("Audio fade skipped: %s", exc)
            return clip

    def _with_volume(self, clip, volume: float):
        if hasattr(clip, "with_volume_scaled"):
            return clip.with_volume_scaled(volume)
        if hasattr(clip, "volumex"):
            return clip.volumex(volume)
        return clip

    def _subclip(self, clip, start: float, end: float):
        if hasattr(clip, "subclipped"):
            return clip.subclipped(start, end)
        return clip.subclip(start, end)

    def _with_duration(self, clip, duration: float):
        if hasattr(clip, "with_duration"):
            return clip.with_duration(duration)
        return clip.set_duration(duration)

    def _with_audio(self, clip, audio_clip):
        if hasattr(clip, "with_audio"):
            return clip.with_audio(audio_clip)
        return clip.set_audio(audio_clip)

    def _create_thumbnail(self, image_paths: list[Path], thumbnail_path: Path) -> None:
        if not image_paths:
            return
        with Image.open(image_paths[0]) as image:
            image.convert("RGB").save(thumbnail_path, quality=92)
