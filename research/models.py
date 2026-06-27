from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ResearchContext:
    topic: str
    summary: str
    facts: list[str]
    statistics: list[str]
    timeline: list[str]
    locations: list[str]
    people: list[str]
    scientific_explanations: list[str]
    myths: list[str]
    controversies: list[str]
    sources: list[str]
    interesting_hooks: list[str]
    warnings: list[str]
    confidence_score: float

    def to_dict(self) -> dict:
        return asdict(self)
