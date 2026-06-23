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


class MemoryMonitor(threading.Thread):
    def __init__(self, logger: logging.Logger, interval: int = 30):
        super().__init__()
        self.logger = logger
        self.interval = interval
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            try:
                mem = psutil.virtual_memory()
                self.logger.info(
                    "[Memory Monitor] System Memory: %.1f%% | Available: %.2f MB | Used: %.2f MB",
                    mem.percent, mem.available / (1024*1024), mem.used / (1024*1024)
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
            mem = psutil.virtual_memory()
            self.logger.info(
                "[%s] Memory usage check: %.1f%% | Used: %.2f MB | Available: %.2f MB",
                step_label, mem.percent, mem.used / (1024*1024), mem.available / (1024*1024)
            )
            if mem.percent > 80.0:
                self.logger.warning(
                    "[%s] Memory usage exceeded 80%% threshold (%.1f%%)! Triggering gc.collect()",
                    step_label, mem.percent
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
        
        # Start background memory monitoring thread
        monitor = MemoryMonitor(self.logger, interval=30)
        monitor.start()
        
        scene_clips = []
        elapsed = 0.0
        max_seconds = self.settings.shorts_max_seconds
        quality_reduced = False

        try:
            # Initial memory check
            self._check_memory_and_optimize("Startup")

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

                self.logger.info("Composing scene %s/%s (%.2fs)", index, len(script.segments), duration)
                visual_clip = self._image_clip(image_paths[index - 1], duration, index)
                overlays = self._caption_clips(segment, duration, elapsed, max_seconds)
                composed = CompositeVideoClip([visual_clip, *overlays], size=self.settings.video_resolution, use_bgclip=True)
                composed = self._with_duration(composed, duration)
                composed = self._with_audio(composed, audio_clip)
                composed = self._apply_video_transitions(composed)
                scene_clips.append(composed)
                elapsed += duration
                
                # Check memory after scene creation and collect garbage
                if self._check_memory_and_optimize(f"Scene {index}"):
                    quality_reduced = True

            if not scene_clips:
                raise RuntimeError("No valid scenes were generated; cannot render video")

            # Check memory before video concatenation and write
            if self._check_memory_and_optimize("Pre-Concatenation") or quality_reduced:
                # Downscale resolution dynamically if memory is above threshold to prevent write_videofile OOM
                orig_res = self.settings.video_resolution
                new_res = (int(orig_res[0] * 0.7), int(orig_res[1] * 0.7))
                orig_fps = self.settings.render_fps
                new_fps = 15 if orig_fps > 15 else 12
                self.logger.warning(
                    "OOM Protection Triggered. Automatically reducing render profile: "
                    "Resolution: %s -> %s | FPS: %s -> %s",
                    orig_res, new_res, orig_fps, new_fps
                )
                # Apply quality reductions to settings so caption generation and composition scale down
                object.__setattr__(self.settings, "video_resolution", new_res)
                object.__setattr__(self.settings, "render_fps", new_fps)
                
                # Resize all scene clips to new resolution
                scene_clips = [clip.resized(new_size=new_res) for clip in scene_clips]
                gc.collect()

            self.logger.info("Rendering %s scenes into %s", len(scene_clips), paths.video_path)
            final_video = concatenate_videoclips(scene_clips, method="compose")
            if final_video.audio:
                music_clip = self._background_music(final_video.duration)
                music_composite = CompositeAudioClip([final_video.audio, music_clip])
                final_video = self._with_audio(final_video, music_composite)

            # Final memory check before write
            self._check_memory_and_optimize("Pre-Write")

            final_video.write_videofile(
                str(paths.video_path),
                fps=self.settings.render_fps,
                codec="libx264",
                audio_codec="aac",
                preset=self.settings.ffmpeg_preset,
                temp_audiofile=str(self.settings.storage_dir / "temp-audio.m4a"),
                remove_temp=True,
            )
            
            # Post-write cleanup
            self.logger.info("Video render finished. Releasing all video clips and subclips.")
            self._deep_close(final_video)
            for clip in scene_clips:
                self._deep_close(clip)
            scene_clips.clear()
            gc.collect()
            
        finally:
            monitor.stop()
            monitor.join(timeout=2.0)
            
        return paths.video_path

    def _caption_clips(self, segment: Segment, duration: float, scene_start: float, total_seconds: float) -> list[ImageClip]:
        chunks = self.caption_service.chunks(segment.subtitle or segment.text)
        if not chunks:
            return []
        chunk_duration = duration / len(chunks)
        overlays: list[ImageClip] = []
        for index, caption in enumerate(chunks):
            start = index * chunk_duration
            end = duration if index == len(chunks) - 1 else (index + 1) * chunk_duration
            overlay = self.caption_service.make_overlay(caption, scene_start + start, total_seconds)
            clip = self._with_duration(ImageClip(overlay), max(0.1, end - start)).with_start(start)
            overlays.append(clip)
        return overlays

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
