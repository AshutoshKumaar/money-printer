from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ResearchContext:
    """Legacy ResearchContext model retained for backward compatibility."""
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================
# New Structured Research Package Models
# ==========================================

@dataclass(slots=True, frozen=True)
class HistoricalContext:
    timeline: list[str]
    overview: str = ""


@dataclass(slots=True, frozen=True)
class ScientificContext:
    explanation: str
    concepts: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ImportantEntities:
    people: list[str]
    places: list[str]
    organizations: list[str]
    technologies: list[str]
    events: list[str]
    objects: list[str]


@dataclass(slots=True, frozen=True)
class VerifiedFact:
    fact: str
    confidence_score: float
    verification_status: str
    reasoning: str
    evidence_level: str
    source_type: str


@dataclass(slots=True, frozen=True)
class CommonMisconception:
    myth: str
    verified_fact: str


@dataclass(slots=True, frozen=True)
class UnansweredQuestion:
    question: str
    uncertainty_type: str  # scientific, historical, general


@dataclass(slots=True, frozen=True)
class VisualOpportunity:
    opportunity_type: str  # photograph, map, diagram, historical artwork, reconstruction, animation
    description: str


@dataclass(slots=True, frozen=True)
class StoryOpportunities:
    strongest_hook: str
    biggest_surprise: str
    emotional_moments: list[str]
    best_climax: str
    strongest_ending: str


@dataclass(slots=True, frozen=True)
class SEOResearch:
    primary_keywords: list[str]
    secondary_keywords: list[str]
    related_concepts: list[str]


@dataclass(slots=True, frozen=True)
class ResearchConfidence:
    overall_score: float
    potential_weak_areas: list[str]
    recommended_verification_priority: str


@dataclass(slots=True, frozen=True)
class ResearchPackage:
    """Immutable structured knowledge output representing the single source of truth."""
    topic: str
    topic_summary: str
    historical_context: HistoricalContext
    scientific_context: ScientificContext
    important_entities: ImportantEntities
    verified_facts: list[VerifiedFact]
    common_misconceptions: list[CommonMisconception]
    unanswered_questions: list[UnansweredQuestion]
    visual_opportunities: list[VisualOpportunity]
    story_opportunities: StoryOpportunities
    seo_research: SEOResearch
    research_confidence: ResearchConfidence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================
# Legacy Adapter
# ==========================================

class ResearchPackageAdapter:
    """Exposes the legacy ResearchContext interface to downstream modules."""

    def __init__(self, package: ResearchPackage) -> None:
        self._package = package

    @property
    def topic(self) -> str:
        return self._package.topic

    @property
    def summary(self) -> str:
        return self._package.topic_summary

    @property
    def facts(self) -> list[str]:
        return [f.fact for f in self._package.verified_facts]

    @property
    def statistics(self) -> list[str]:
        # Legacy placeholder
        return []

    @property
    def timeline(self) -> list[str]:
        return self._package.historical_context.timeline

    @property
    def locations(self) -> list[str]:
        return self._package.important_entities.places

    @property
    def people(self) -> list[str]:
        return self._package.important_entities.people

    @property
    def scientific_explanations(self) -> list[str]:
        return [self._package.scientific_context.explanation] if self._package.scientific_context.explanation else []

    @property
    def myths(self) -> list[str]:
        return [f"Myth: {m.myth} | Truth: {m.verified_fact}" for m in self._package.common_misconceptions]

    @property
    def controversies(self) -> list[str]:
        return [q.question for q in self._package.unanswered_questions]

    @property
    def sources(self) -> list[str]:
        return []

    @property
    def interesting_hooks(self) -> list[str]:
        return [self._package.story_opportunities.strongest_hook] if self._package.story_opportunities.strongest_hook else []

    @property
    def warnings(self) -> list[str]:
        return self._package.research_confidence.potential_weak_areas

    @property
    def confidence_score(self) -> float:
        return self._package.research_confidence.overall_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "facts": self.facts,
            "statistics": self.statistics,
            "timeline": self.timeline,
            "locations": self.locations,
            "people": self.people,
            "scientific_explanations": self.scientific_explanations,
            "myths": self.myths,
            "controversies": self.controversies,
            "sources": self.sources,
            "interesting_hooks": self.interesting_hooks,
            "warnings": self.warnings,
            "confidence_score": self.confidence_score,
        }
