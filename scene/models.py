from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SceneShot:
    """Legacy SceneShot retained for backward compatibility."""
    scene_index: int
    shot_index: int
    duration_seconds: float
    purpose: str
    visual_description: str
    camera_angle: str
    camera_motion: str
    lens_type: str
    lighting: str
    environment: str
    time_of_day: str
    color_palette: str
    emotion: str
    transition_in: str
    transition_out: str
    caption_style: str
    search_query: str
    ai_image_prompt: str
    stock_video_query: str
    sound_effects: str
    background_music_mood: str
    priority: str
    stock_search_query: str = ""
    cache_key: str = ""


@dataclass(slots=True)
class ScenePlanManifest:
    """Legacy ScenePlanManifest retained for backward compatibility."""
    overall_style: str
    scene_count: int
    estimated_runtime: float
    scenes: list[SceneShot]

    def to_dict(self) -> dict:
        return asdict(self)


# ==========================================
# New Structured Scene Planner Models
# ==========================================

@dataclass(slots=True, frozen=True)
class CameraInstruction:
    motion_type: str  # zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down, tilt, static
    speed: str  # slow, medium, fast


@dataclass(slots=True, frozen=True)
class TransitionInstruction:
    transition_type: str  # fade, crossfade, dissolve, zoom, wipe, slide, none


@dataclass(slots=True, frozen=True)
class OverlayInstruction:
    overlay_type: str  # text, subtitle, diagram, map, label, none
    text: str | None = None
    position: str = "center"  # top, center, bottom
    style: str = "default"
    animation: str = "none"
    duration: float = 0.0


@dataclass(slots=True, frozen=True)
class Shot:
    """Granular camera shot within a scene."""
    shot_id: str
    visual_goal: str
    camera_motion: CameraInstruction
    duration: float
    transition_to_next: TransitionInstruction
    visual_reference: str | None = None  # Reference to entities or opportunities
    visual_source_strategy: str = "stock_only"  # stock_only, ai_preferred, ai_required, archival, map, diagram, hybrid
    shot_type: str | None = None  # establishing, close_up, medium, aerial, macro, diagram, map, archive, reconstruction
    aspect_ratio_hint: str = "9:16"
    safe_crop_region: dict[str, float] | None = None
    focus_subject: str | None = None


@dataclass(slots=True, frozen=True)
class Scene:
    """Cinematic scene grouping one or more shots mapping to a narration segment."""
    scene_id: str
    narration_segment_id: int
    target_start: float
    target_end: float
    visual_type: str  # stock_video, stock_image, ai_image, diagram, map
    visual_priority: str  # critical, high, medium, low
    transition: TransitionInstruction
    overlay: OverlayInstruction
    shots: list[Shot]
    continuity_group: str = "default"


@dataclass(slots=True, frozen=True)
class ScenePackage:
    """Immutable cinematic scene plan output."""
    scenes: list[Scene]
    estimated_total_duration: float
    pacing_score: float
    visual_variety_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================
# Scene / Shot Adapters
# ==========================================

class SceneShotAdapter:
    """Adapts a new nested Shot (within parent Scene context) to the legacy flat SceneShot interface."""

    def __init__(self, shot: Shot, scene: Scene, scene_index: int, shot_index: int) -> None:
        self._shot = shot
        self._scene = scene
        self._scene_index = scene_index
        self._shot_index = shot_index

    @property
    def scene_index(self) -> int:
        return self._scene_index

    @property
    def shot_index(self) -> int:
        return self._shot_index

    @property
    def duration_seconds(self) -> float:
        return self._shot.duration

    @property
    def purpose(self) -> str:
        return self._scene.overlay.overlay_type

    @property
    def visual_description(self) -> str:
        return self._shot.visual_goal

    @property
    def camera_angle(self) -> str:
        st = self._shot.shot_type
        if st in {"establishing", "aerial"}:
            return "wide-angle"
        elif st in {"close_up", "macro"}:
            return "close-up"
        return "eye-level"

    @property
    def camera_motion(self) -> str:
        return f"{self._shot.camera_motion.motion_type} ({self._shot.camera_motion.speed})"

    @property
    def lens_type(self) -> str:
        return "medium"

    @property
    def lighting(self) -> str:
        return "cinematic"

    @property
    def environment(self) -> str:
        return "abstract"

    @property
    def time_of_day(self) -> str:
        return "not-applicable"

    @property
    def color_palette(self) -> str:
        return "rich"

    @property
    def emotion(self) -> str:
        return "curiosity"

    @property
    def transition_in(self) -> str:
        return self._scene.transition.transition_type

    @property
    def transition_out(self) -> str:
        return self._shot.transition_to_next.transition_type

    @property
    def caption_style(self) -> str:
        return "standard"

    @property
    def search_query(self) -> str:
        return self._shot.visual_reference or self._shot.visual_goal

    @property
    def ai_image_prompt(self) -> str:
        prompt = f"A cinematic vertical 9:16 shot of {self._shot.visual_goal}. {self._scene.continuity_group} style, photorealistic, highly detailed."
        if self._shot.focus_subject:
            prompt = f"{prompt} Emphasizing {self._shot.focus_subject}."
        return prompt

    @property
    def stock_video_query(self) -> str:
        return self.search_query

    @property
    def sound_effects(self) -> str:
        return "none"

    @property
    def background_music_mood(self) -> str:
        return "mysterious"

    @property
    def priority(self) -> str:
        return self._scene.visual_priority.upper()

    @property
    def stock_search_query(self) -> str:
        return self.search_query

    @property
    def cache_key(self) -> str:
        return self.search_query


class ScenePackageAdapter:
    """Adapts a nested ScenePackage to the legacy ScenePlanManifest interface by flattening nested shots."""

    def __init__(self, package: ScenePackage) -> None:
        self._package = package

    @property
    def overall_style(self) -> str:
        return "cinematic"

    @property
    def scene_count(self) -> int:
        return len(self._package.scenes)

    @property
    def estimated_runtime(self) -> float:
        return self._package.estimated_total_duration

    @property
    def scenes(self) -> list[SceneShotAdapter]:
        flattened = []
        for s_idx, scene in enumerate(self._package.scenes, start=1):
            for sh_idx, shot in enumerate(scene.shots, start=1):
                flattened.append(SceneShotAdapter(shot, scene, s_idx, sh_idx))
        return flattened

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_style": self.overall_style,
            "scene_count": self.scene_count,
            "estimated_runtime": self.estimated_runtime,
            "scenes": [
                {
                    "scene_index": s.scene_index,
                    "shot_index": s.shot_index,
                    "duration_seconds": s.duration_seconds,
                    "purpose": s.purpose,
                    "visual_description": s.visual_description,
                    "camera_angle": s.camera_angle,
                    "camera_motion": s.camera_motion,
                    "lens_type": s.lens_type,
                    "lighting": s.lighting,
                    "environment": s.environment,
                    "time_of_day": s.time_of_day,
                    "color_palette": s.color_palette,
                    "emotion": s.emotion,
                    "transition_in": s.transition_in,
                    "transition_out": s.transition_out,
                    "caption_style": s.caption_style,
                    "search_query": s.search_query,
                    "ai_image_prompt": s.ai_image_prompt,
                    "stock_video_query": s.stock_video_query,
                    "sound_effects": s.sound_effects,
                    "background_music_mood": s.background_music_mood,
                    "priority": s.priority,
                    "stock_search_query": s.stock_search_query,
                    "cache_key": s.cache_key,
                }
                for s in self.scenes
            ],
        }
