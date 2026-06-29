from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class VisualAsset:
    """Legacy VisualAsset retained for backward compatibility."""
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
    """Legacy VisualAssetManifest retained for backward compatibility."""
    assets: list[VisualAsset]

    def to_dict(self) -> dict:
        return asdict(self)


# ==========================================
# New Structured Visual Decision Models
# ==========================================

@dataclass(slots=True, frozen=True)
class AssetMetadata:
    perceptual_hash: str | None = None
    dominant_colors: list[str] = field(default_factory=list)
    detected_subjects: list[str] = field(default_factory=list)
    creation_timestamp: float = 0.0
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class NewVisualAsset:
    """Immutable asset record resolved by the Visual Engine V2."""
    shot_id: str
    asset_id: str
    source: str  # cache, existing, stock, archive, diagram, ai, fallback
    confidence: float
    cache_hit: bool
    continuity_group: str
    quality_score: float
    resolution: str
    orientation: str
    local_path: str
    metadata: AssetMetadata


@dataclass(slots=True, frozen=True)
class AssetDecision:
    """Telemetry and logic records of the visual source decision for a shot."""
    shot_id: str
    evaluated_sources: list[str]
    selected_source: str
    reasoning: str
    decision_time_ms: float = 0.0
    fallback_triggered: bool = False
    retry_count: int = 0


@dataclass(slots=True, frozen=True)
class VisualPackage:
    """Immutable package of resolved assets and decisions."""
    assets: list[NewVisualAsset]
    decisions: list[AssetDecision]
    total_duration: float
    ai_generation_count: int
    cache_efficiency: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================
# Visual Adapters
# ==========================================

class VisualAssetAdapter:
    """Adapts NewVisualAsset to the legacy VisualAsset interface."""

    def __init__(self, asset: NewVisualAsset, scene_index: int) -> None:
        self._asset = asset
        self._scene_index = scene_index

    @property
    def scene_index(self) -> int:
        return self._scene_index

    @property
    def provider(self) -> str:
        s = self._asset.source
        if s == "ai":
            return "aiimage"
        elif s == "stock":
            return "pexels"
        return s

    @property
    def asset_type(self) -> str:
        return "image"

    @property
    def prompt(self) -> str:
        return self._asset.asset_id

    @property
    def file_path(self) -> str:
        return self._asset.local_path

    @property
    def quality_score(self) -> float:
        return self._asset.quality_score

    @property
    def confidence(self) -> float:
        return self._asset.confidence

    @property
    def cache_hit(self) -> bool:
        return self._asset.cache_hit

    @property
    def generation_time(self) -> float:
        return 0.0

    @property
    def cache_key(self) -> str:
        return self._asset.local_path


class VisualPackageAdapter:
    """Adapts VisualPackage to the legacy VisualAssetManifest interface."""

    def __init__(self, package: VisualPackage) -> None:
        self._package = package

    @property
    def assets(self) -> list[VisualAssetAdapter]:
        adapted = []
        for asset in self._package.assets:
            import re
            match = re.search(r'shot_(\d+)_', asset.shot_id)
            scene_idx = int(match.group(1)) if match else 1
            adapted.append(VisualAssetAdapter(asset, scene_idx))
        return adapted

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": [
                {
                    "scene_index": a.scene_index,
                    "provider": a.provider,
                    "asset_type": a.asset_type,
                    "prompt": a.prompt,
                    "file_path": a.file_path,
                    "quality_score": a.quality_score,
                    "confidence": a.confidence,
                    "cache_hit": a.cache_hit,
                    "generation_time": a.generation_time,
                    "cache_key": a.cache_key,
                }
                for a in self.assets
            ]
        }
