from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class LearningState:
    """Deterministic, immutable representation of the learned weights and parameters."""
    category_weights: dict[str, float]
    topic_weights: dict[str, float]
    pacing_weights: dict[str, float]
    visual_weights: dict[str, float]
    upload_schedule_weights: dict[str, float]
    confidence_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
