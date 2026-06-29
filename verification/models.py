from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from research.models import ResearchPackage


@dataclass(slots=True)
class FactVerificationRecord:
    """Legacy FactVerificationRecord retained for backward compatibility."""
    original_fact: str
    status: str  # verified, partially_verified, unverified, contradictory
    corrected_version: str
    confidence: float
    explanation: str
    importance_score: float
    category: str


@dataclass(slots=True)
class VerificationReport:
    """Legacy VerificationReport retained for backward compatibility."""
    verified_facts: list[FactVerificationRecord]
    rejected_facts: list[FactVerificationRecord]
    corrected_facts: list[FactVerificationRecord]
    warnings: list[str]
    confidence_score: float
    research_quality_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================
# New Structured Verification Models
# ==========================================

@dataclass(slots=True, frozen=True)
class VerifiedFactRecord:
    """Immutable record of fact verification containing granular results."""
    fact: str
    status: str  # verified, partially_verified, disputed, insufficient_evidence, unverified
    original_fact: str
    suggested_clarification: str | None
    confidence_score: float
    reasoning: str
    evidence_level: str
    source_type: str
    contradictions_detected: list[str] = field(default_factory=list)
    is_duplicate: bool = False
    importance_score: float = 0.5
    category: str = "general"


@dataclass(slots=True, frozen=True)
class VerifiedResearchPackage:
    """Immutable package representing the fully verified research context."""
    research_package: ResearchPackage
    verified_facts: list[VerifiedFactRecord]
    verification_warnings: list[str] = field(default_factory=list)
    overall_confidence_score: float = 0.0
    verification_quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================
# Verification Adapter
# ==========================================

class VerificationAdapter:
    """Adapts VerifiedResearchPackage to the legacy VerificationReport interface."""

    def __init__(self, package: VerifiedResearchPackage) -> None:
        self._package = package

    @property
    def verified_facts(self) -> list[FactVerificationRecord]:
        # Filter out duplicates and return only verified/partially_verified facts for storytelling
        records = []
        for r in self._package.verified_facts:
            if r.is_duplicate:
                continue
            if r.status in {"verified", "partially_verified"}:
                records.append(self._adapt_record(r))
        return records

    @property
    def rejected_facts(self) -> list[FactVerificationRecord]:
        # Return facts with negative outcomes
        records = []
        for r in self._package.verified_facts:
            if r.status in {"unverified", "disputed", "insufficient_evidence"}:
                records.append(self._adapt_record(r))
        return records

    @property
    def corrected_facts(self) -> list[FactVerificationRecord]:
        # Return partially_verified (corrected) facts
        records = []
        for r in self._package.verified_facts:
            if r.is_duplicate:
                continue
            if r.status == "partially_verified":
                records.append(self._adapt_record(r))
        return records

    @property
    def warnings(self) -> list[str]:
        return self._package.verification_warnings

    @property
    def confidence_score(self) -> float:
        return self._package.overall_confidence_score

    @property
    def research_quality_score(self) -> float:
        return self._package.verification_quality_score

    def _adapt_record(self, r: VerifiedFactRecord) -> FactVerificationRecord:
        # Fallback corrected version to original fact if no suggested clarification
        corrected = r.suggested_clarification or r.fact
        
        # Map statuses
        legacy_status = r.status
        if legacy_status == "disputed":
            legacy_status = "contradictory"
        elif legacy_status == "insufficient_evidence":
            legacy_status = "unverified"

        return FactVerificationRecord(
            original_fact=r.original_fact,
            status=legacy_status,
            corrected_version=corrected,
            confidence=r.confidence_score,
            explanation=r.reasoning,
            importance_score=r.importance_score,
            category=r.category,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_facts": [asdict(f) for f in self.verified_facts],
            "rejected_facts": [asdict(f) for f in self.rejected_facts],
            "corrected_facts": [asdict(f) for f in self.corrected_facts],
            "warnings": self.warnings,
            "confidence_score": self.confidence_score,
            "research_quality_score": self.research_quality_score,
        }
