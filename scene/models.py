from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SceneShot:
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
    overall_style: str
    scene_count: int
    estimated_runtime: float
    scenes: list[SceneShot]

    def to_dict(self) -> dict:
        return asdict(self)
