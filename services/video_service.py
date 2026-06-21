from __future__ import annotations

import logging
import math
from pathlib import Path

import moviepy
import numpy as np
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


class VideoService:
    """Renders a vertical Shorts video from generated images, audio, and captions."""

    def __init__(self, settings: Settings, logger: logging.Logger, caption_service: CaptionService) -> None:
        self.settings = settings
        self.logger = logger
        self.caption_service = caption_service

    def render(
        self,
        script: Script,
        image_paths: list[Path],
        audio_paths: list[Path],
        paths: RunPaths,
    ) -> Path:
        self._create_thumbnail(image_paths, paths.thumbnail_path)
        scene_clips = []
        elapsed = 0.0
        max_seconds = self.settings.shorts_max_seconds

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

        if not scene_clips:
            raise RuntimeError("No valid scenes were generated; cannot render video")

        self.logger.info("Rendering %s scenes into %s", len(scene_clips), paths.video_path)
        final_video = concatenate_videoclips(scene_clips, method="compose")
        if final_video.audio:
            music_clip = self._background_music(final_video.duration)
            final_video = self._with_audio(final_video, CompositeAudioClip([final_video.audio, music_clip]))

        final_video.write_videofile(
            str(paths.video_path),
            fps=self.settings.render_fps,
            codec="libx264",
            audio_codec="aac",
            preset=self.settings.ffmpeg_preset,
            temp_audiofile=str(self.settings.storage_dir / "temp-audio.m4a"),
            remove_temp=True,
            threads=4,
        )
        final_video.close()
        for clip in scene_clips:
            clip.close()
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
            return self._ken_burns(clip, duration, index)
        except Exception as exc:
            self.logger.warning("Could not load image %s: %s", image_path, exc)
            return self._with_duration(ColorClip(size=target_size, color=(12, 16, 22)), duration)

    def _ken_burns(self, clip, duration: float, index: int):
        try:
            target_w, target_h = self.settings.video_resolution
            pan_left_to_right = index % 2 == 0

            def transform(get_frame, t: float):
                progress = min(1.0, max(0.0, float(t) / max(duration, 0.01)))
                scale = 1.03 + (0.055 * progress)
                frame = get_frame(t).astype("uint8")
                new_w = max(target_w, math.ceil(frame.shape[1] * scale))
                new_h = max(target_h, math.ceil(frame.shape[0] * scale))
                resized = np.array(
                    Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.LANCZOS)
                )

                max_x = max(0, new_w - target_w)
                max_y = max(0, new_h - target_h)
                pan_progress = 0.2 + (0.6 * progress)
                if not pan_left_to_right:
                    pan_progress = 1.0 - pan_progress
                x1 = min(max_x, max(0, round(max_x * pan_progress)))
                y1 = min(max_y, max(0, round(max_y * 0.5)))
                return resized[y1:y1 + target_h, x1:x1 + target_w]

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
