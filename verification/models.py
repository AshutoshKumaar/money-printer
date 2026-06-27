from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class FactVerificationRecord:
    original_fact: str
    status: str  # verified, partially_verified, unverified, contradictory
    corrected_version: str
    confidence: float
    explanation: str
    importance_score: float
    category: str


@dataclass(slots=True)
class VerificationReport:
    verified_facts: list[FactVerificationRecord]
    rejected_facts: list[FactVerificationRecord]
    corrected_facts: list[FactVerificationRecord]
    warnings: list[str]
    confidence_score: float
    research_quality_score: float

    def to_dict(self) -> dict:
        return asdict(self)
