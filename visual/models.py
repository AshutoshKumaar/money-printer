from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class VisualAsset:
    scene_index: int
    provider: str
    asset_type: str  # image / video
    prompt: str
    file_path: str
    quality_score: float
    confidence: float
    cache_hit: bool
    generation_time: float
    cache_key: str


@dataclass(slots=True)
class VisualAssetManifest:
    assets: list[VisualAsset]

    def to_dict(self) -> dict:
        return asdict(self)
