from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class EmotionCurve:
    curiosity: float
    fear: float
    surprise: float
    wonder: float
    urgency: float


@dataclass(slots=True)
class StorySegment:
    index: int
    spoken_hindi: str
    caption_keywords: str
    search_query: str
    visual_concept: str
    emotion_curve: EmotionCurve
    subtitle_text: str = ""


@dataclass(slots=True)
class NarrativeScript:
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

