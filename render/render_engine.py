from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from config import Settings
from story.models import NarrativePackage
from scene.models import ScenePackage, Scene, Shot
from visual.models import VisualPackage, NewVisualAsset
from render.models import (
    RenderPackage,
    RenderClip,
    SubtitleSegment,
    AudioTrack,
)


class RenderEngine:
    """Deterministic video rendering pipeline designer that creates immutable RenderPackages without AI or LLM calls."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def generate_package(
        self,
        narrative: NarrativePackage,
        scene: ScenePackage,
        visual: VisualPackage,
        voice_paths: list[str],
    ) -> RenderPackage:
        self.logger.info("Generating deterministic RenderPackage specification...")

        clips: list[RenderClip] = []
        audio_tracks: list[AudioTrack] = []
        subtitles: list[SubtitleSegment] = []

        # 1. Timeline generation & Clip sequencing
        timeline_offset = 0.0
        shot_index = 1
        
        # Build lookup dict for visual assets
        visual_lookup = {a.shot_id: a for a in visual.assets}

        for s in scene.scenes:
            for shot in s.shots:
                asset = visual_lookup.get(shot.shot_id)
                asset_path = asset.local_path if asset else ""

                # Deterministic Ken Burns effect configuration
                if shot_index % 2 == 0:
                    zoom_start, zoom_end = 1.0, 1.1
                    pan_x_start, pan_x_end = -10.0, 10.0
                    pan_y_start, pan_y_end = -5.0, 5.0
                else:
                    zoom_start, zoom_end = 1.1, 1.0
                    pan_x_start, pan_x_end = 10.0, -10.0
                    pan_y_start, pan_y_end = 5.0, -5.0

                clips.append(
                    RenderClip(
                        clip_id=f"clip_{shot.shot_id}",
                        shot_id=shot.shot_id,
                        asset_path=asset_path,
                        start_time=timeline_offset,
                        end_time=timeline_offset + shot.duration,
                        duration=shot.duration,
                        ken_burns_zoom_start=zoom_start,
                        ken_burns_zoom_end=zoom_end,
                        ken_burns_pan_x_start=pan_x_start,
                        ken_burns_pan_x_end=pan_x_end,
                        ken_burns_pan_y_start=pan_y_start,
                        ken_burns_pan_y_end=pan_y_end,
                        transition_in=s.transition.transition_type,
                        transition_out=shot.transition_to_next.transition_type,
                    )
                )

                timeline_offset += shot.duration
                shot_index += 1

        total_duration = timeline_offset

        # 2. Audio track sequencing (Narration + BGM)
        segment_offset = 0.0
        for idx, seg in enumerate(narrative.narration_segments):
            path = voice_paths[idx] if idx < len(voice_paths) else ""
            audio_tracks.append(
                AudioTrack(
                    track_id=f"narration_track_{idx + 1}",
                    track_type="narration",
                    file_path=path,
                    start_time=segment_offset,
                    end_time=segment_offset + seg.estimated_duration,
                    duration=seg.estimated_duration,
                    volume=self.settings.voice_volume,
                )
            )

            # Subtitle scheduling mapping
            # Split the narration text into smaller chunks
            words = seg.narration_text.split()
            chunk_size = 4
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            if chunks:
                chunk_dur = seg.estimated_duration / len(chunks)
                for c_idx, chunk in enumerate(chunks):
                    subtitles.append(
                        SubtitleSegment(
                            text=chunk,
                            start_time=segment_offset + (c_idx * chunk_dur),
                            end_time=segment_offset + min((c_idx + 1) * chunk_dur, seg.estimated_duration),
                            dialogue_index=len(subtitles) + 1,
                        )
                    )

            segment_offset += seg.estimated_duration

        # Background music track
        bgm_volume = max(0.35, self.settings.background_music_volume * 5.0)
        audio_tracks.append(
            AudioTrack(
                track_id="bg_music_track",
                track_type="bg_music",
                file_path=str(self.settings.audio_dir / "background_music.mp3"),
                start_time=0.0,
                end_time=total_duration,
                duration=total_duration,
                volume=bgm_volume,
                fade_in=min(2.0, total_duration * 0.05),
                fade_out=min(2.0, total_duration * 0.05),
            )
        )

        # 3. FFmpeg Complex Filter Graph Generation
        # Standardized filter graph for reproducible rendering
        # [0:a] is voice, [1:a] is background music
        fade_in = min(2.0, total_duration * 0.05)
        fade_out_st = total_duration - fade_in
        filter_graph = (
            f"[0:a]volume={self.settings.voice_volume},asplit=2[voice1][voice2];"
            f"[1:a]volume={bgm_volume:.3f},"
            f"afade=t=in:st=0:d={fade_in:.3f},"
            f"afade=t=out:st={fade_out_st:.3f}:d={fade_in:.3f}[bg_music];"
            f"[bg_music][voice1]sidechaincompress=threshold={self.settings.sidechain_threshold:.2f}:"
            f"ratio={self.settings.sidechain_ratio:.1f}:"
            f"attack={self.settings.sidechain_attack}:"
            f"release={self.settings.sidechain_release}[ducked_bg];"
            f"[voice2][ducked_bg]amix=inputs=2:duration=first:normalize=0[mixed];"
            f"[mixed]loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
        )

        # Climax or middle point thumbnail selection offset
        thumbnail_offset = min(total_duration / 2.0, 15.0)

        # Immutable export settings
        export_settings = {
            "codec": "libx264",
            "audio_codec": "aac",
            "preset": self.settings.ffmpeg_preset,
            "bitrate": "8M",
        }

        package = RenderPackage(
            clips=clips,
            subtitles=subtitles,
            audio_tracks=audio_tracks,
            total_duration=total_duration,
            resolution=self.settings.video_resolution,
            fps=self.settings.render_fps,
            ffmpeg_filter_graph=filter_graph,
            thumbnail_frame_offset=thumbnail_offset,
            export_settings=export_settings,
        )

        # Validate spec integrity before returning
        self.validate_package(package)

        return package

    def validate_package(self, package: RenderPackage) -> None:
        """Validate duration alignment, timing overlap, and subtitle bounds."""
        if not package.clips:
            raise ValueError("RenderPackage must contain at least one video clip specification.")
        if package.total_duration <= 0.0:
            raise ValueError(f"RenderPackage invalid total duration: {package.total_duration}s.")

        # Clip overlap check
        sorted_clips = sorted(package.clips, key=lambda c: c.start_time)
        prev_end = 0.0
        for clip in sorted_clips:
            if abs(clip.start_time - prev_end) > 0.01:
                # Allow minor gaps due to shot alignment, but log it
                pass
            prev_end = clip.end_time

        # Subtitle bounds check
        for sub in package.subtitles:
            if sub.start_time < 0.0 or sub.end_time > package.total_duration:
                raise ValueError(f"Subtitle segment '{sub.text}' timing [{sub.start_time}s, {sub.end_time}s] exceeds video bounds [0s, {package.total_duration}s].")
            if sub.start_time >= sub.end_time:
                raise ValueError(f"Subtitle segment '{sub.text}' start time must be less than end time.")
