from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class RenderClip:
    """A deterministic video clip mapping to a shot on the rendering timeline."""
    clip_id: str
    shot_id: str
    asset_path: str
    start_time: float
    end_time: float
    duration: float
    ken_burns_zoom_start: float
    ken_burns_zoom_end: float
    ken_burns_pan_x_start: float
    ken_burns_pan_x_end: float
    ken_burns_pan_y_start: float
    ken_burns_pan_y_end: float
    transition_in: str
    transition_out: str


@dataclass(slots=True, frozen=True)
class SubtitleSegment:
    """A deterministic subtitle interval mapped to a timeline offset."""
    text: str
    start_time: float
    end_time: float
    dialogue_index: int


@dataclass(slots=True, frozen=True)
class AudioTrack:
    """A deterministic audio channel description."""
    track_id: str
    track_type: str  # narration, bg_music, sound_effect
    file_path: str
    start_time: float
    end_time: float
    duration: float
    volume: float
    fade_in: float = 0.0
    fade_out: float = 0.0


@dataclass(slots=True, frozen=True)
class RenderPackage:
    """Deterministic, immutable specification of the compiled video timeline."""
    clips: list[RenderClip]
    subtitles: list[SubtitleSegment]
    audio_tracks: list[AudioTrack]
    total_duration: float
    resolution: tuple[int, int]
    fps: int
    ffmpeg_filter_graph: str
    thumbnail_frame_offset: float
    export_settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================
# Legacy Adapters
# ==========================================

class RenderPackageAdapter:
    """Exposes a legacy-compatible interface for downstream components expecting Script metadata."""

    def __init__(self, package: RenderPackage) -> None:
        self._package = package

    @property
    def total_duration(self) -> float:
        return self._package.total_duration

    @property
    def resolution(self) -> tuple[int, int]:
        return self._package.resolution

    @property
    def fps(self) -> int:
        return self._package.fps
