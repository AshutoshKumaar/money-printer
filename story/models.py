from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EmotionCurve:
    """Legacy EmotionCurve retained for backward compatibility."""
    curiosity: float
    fear: float
    surprise: float
    wonder: float
    urgency: float


@dataclass(slots=True)
class StorySegment:
    """Legacy StorySegment retained for backward compatibility."""
    index: int
    spoken_hindi: str
    caption_keywords: str
    search_query: str
    visual_concept: str
    emotion_curve: EmotionCurve
    subtitle_text: str = ""


@dataclass(slots=True)
class NarrativeScript:
    """Legacy NarrativeScript retained for backward compatibility."""
    hook: str
    context: str
    segments: list[StorySegment]
    ending: str
    cta: str
    estimated_duration: float
    estimated_words: int
    emotion_curve: EmotionCurve
    retention_score: float
    seo: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ==========================================
# New Structured Story Models
# ==========================================

@dataclass(slots=True, frozen=True)
class NarrationSegment:
    """Granular narrative segment focusing solely on spoken script content."""
    index: int
    narration_text: str
    estimated_duration: float
    target_start: float
    target_end: float
    emotion: str  # curiosity, suspense, wonder, surprise, fear, urgency, neutral
    purpose: str
    verified_fact_ids: list[str]  # References verified facts from VerifiedResearchPackage
    beat_type: str  # hook, question, setup, evidence, twist, reveal, reflection, cta


@dataclass(slots=True, frozen=True)
class NarrativeQuality:
    """Metrics tracking quality and pacing of the narrative structure."""
    retention_score: float
    pacing_score: float
    curiosity_score: float
    clarity_score: float
    emotional_score: float
    estimated_retention_curve: list[float]  # Predicted retention curve at regular intervals


@dataclass(slots=True, frozen=True)
class NarrativePackage:
    """Immutable language-independent narrative script design."""
    language: str
    hook: str
    context: str
    escalation: str
    climax: str
    ending: str
    narration_segments: list[NarrationSegment]
    quality: NarrativeQuality

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================
# Story / Narrative Adapters
# ==========================================

class StorySegmentAdapter:
    """Adapts NarrationSegment to legacy StorySegment interface."""

    def __init__(self, segment: NarrationSegment) -> None:
        self._segment = segment

    @property
    def index(self) -> int:
        return self._segment.index

    @property
    def spoken_hindi(self) -> str:
        return self._segment.narration_text

    @property
    def caption_keywords(self) -> str:
        # Downstream fallback
        return ""

    @property
    def search_query(self) -> str:
        # Downstream fallback
        return ""

    @property
    def visual_concept(self) -> str:
        # Downstream fallback
        return "Visual representation of narration."

    @property
    def emotion_curve(self) -> EmotionCurve:
        # Map controlled emotion enum to legacy EmotionCurve values
        curiosity = 0.5
        fear = 0.1
        surprise = 0.1
        wonder = 0.1
        urgency = 0.1
        
        e = self._segment.emotion
        if e == "curiosity":
            curiosity = 0.9
        elif e == "suspense":
            curiosity = 0.8
            urgency = 0.6
        elif e == "wonder":
            wonder = 0.9
        elif e == "surprise":
            surprise = 0.9
        elif e == "fear":
            fear = 0.9
        elif e == "urgency":
            urgency = 0.9

        return EmotionCurve(
            curiosity=curiosity,
            fear=fear,
            surprise=surprise,
            wonder=wonder,
            urgency=urgency,
        )

    @property
    def subtitle_text(self) -> str:
        return ""


class NarrativeAdapter:
    """Adapts NarrativePackage to legacy NarrativeScript interface."""

    def __init__(self, package: NarrativePackage) -> None:
        self._package = package

    @property
    def hook(self) -> str:
        return self._package.hook

    @property
    def context(self) -> str:
        return self._package.context

    @property
    def segments(self) -> list[StorySegmentAdapter]:
        return [StorySegmentAdapter(s) for s in self._package.narration_segments]

    @property
    def ending(self) -> str:
        return self._package.ending

    @property
    def cta(self) -> str:
        # Standard default CTA
        return "Subscribe for more mysterious facts."

    @property
    def estimated_duration(self) -> float:
        # Compute total duration from segments
        return sum(s.estimated_duration for s in self._package.narration_segments)

    @property
    def estimated_words(self) -> int:
        return sum(len(s.narration_text.split()) for s in self._package.narration_segments)

    @property
    def emotion_curve(self) -> EmotionCurve:
        return EmotionCurve(curiosity=0.8, fear=0.2, surprise=0.5, wonder=0.5, urgency=0.3)

    @property
    def retention_score(self) -> float:
        return self._package.quality.retention_score

    @property
    def seo(self) -> dict:
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook": self.hook,
            "context": self.context,
            "segments": [
                {
                    "index": s.index,
                    "spoken_hindi": s.spoken_hindi,
                    "caption_keywords": s.caption_keywords,
                    "search_query": s.search_query,
                    "visual_concept": s.visual_concept,
                    "emotion_curve": {
                        "curiosity": s.emotion_curve.curiosity,
                        "fear": s.emotion_curve.fear,
                        "surprise": s.emotion_curve.surprise,
                        "wonder": s.emotion_curve.wonder,
                        "urgency": s.emotion_curve.urgency,
                    },
                    "subtitle_text": s.subtitle_text,
                }
                for s in self.segments
            ],
            "ending": self.ending,
            "cta": self.cta,
            "estimated_duration": self.estimated_duration,
            "estimated_words": self.estimated_words,
            "emotion_curve": {
                "curiosity": self.emotion_curve.curiosity,
                "fear": self.emotion_curve.fear,
                "surprise": self.emotion_curve.surprise,
                "wonder": self.emotion_curve.wonder,
                "urgency": self.emotion_curve.urgency,
            },
            "retention_score": self.retention_score,
            "seo": self.seo,
        }
