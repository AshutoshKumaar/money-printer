from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Segment:
    """One timed spoken scene in a Shorts video."""

    text: str
    subtitle: str
    image_prompt: str
    search_query: str = ""
    visual_type: str = "ai_image"
    visual_category: str = ""
    visual_concept: str = ""
    visual_provider: str = ""
    visual_confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], topic: str, index: int) -> "Segment":
        default_prompt = (
            f"Vertical 9:16 cinematic Hindi YouTube Shorts scene about {topic}, "
            f"scene {index}, dramatic realistic lighting, high contrast"
        )
        text = str(data.get("text", "")).strip()
        subtitle = str(data.get("subtitle", "")).strip()
        return cls(
            text=text or f"यह {topic} के बारे में एक महत्वपूर्ण बात है।",
            subtitle=subtitle or f"Important point about {topic}",
            image_prompt=str(data.get("image_prompt", default_prompt)).strip() or default_prompt,
            search_query=str(data.get("search_query", f"{topic} cinematic vertical")).strip(),
            visual_type="ai_image",
            visual_category=str(data.get("visual_category", "")).strip(),
            visual_concept=str(data.get("visual_concept", "")).strip(),
            visual_provider=str(data.get("visual_provider", "")).strip(),
            visual_confidence=float(data.get("visual_confidence", 0.0) or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Script:
    """Structured script and publishing metadata."""

    title: str
    description: str
    tags: list[str]
    hashtags: list[str]
    segments: list[Segment] = field(default_factory=list)
    topic: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], topic: str) -> "Script":
        tags = [str(tag).strip() for tag in data.get("tags", []) if str(tag).strip()]
        hashtags = [str(tag).strip().lstrip("#") for tag in data.get("hashtags", []) if str(tag).strip()]
        if not hashtags:
            hashtags = [tag for tag in tags[:5] if tag]
        raw_segments = data.get("segments", [])
        segments = [
            Segment.from_dict(segment, topic, index)
            for index, segment in enumerate(raw_segments, start=1)
            if isinstance(segment, dict)
        ]
        return cls(
            title=str(data.get("title") or f"{topic} in 60 Seconds").strip(),
            description=str(data.get("description") or f"Fast Hindi Shorts video about {topic}.").strip(),
            tags=tags or ["shorts", "hindi", "facts", topic],
            hashtags=hashtags or ["Shorts", "Hindi", "Facts"],
            segments=segments,
            topic=topic,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "hashtags": self.hashtags,
            "topic": self.topic,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(slots=True)
class GeneratedVideo:
    """Result of a generation/upload run."""

    topic: str
    script: Script
    video_path: Path | None
    metadata_path: Path
    thumbnail_path: Path | None = None
    youtube_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "video_path": str(self.video_path) if self.video_path else None,
            "metadata_path": str(self.metadata_path),
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "youtube_url": self.youtube_url,
            "script": self.script.to_dict(),
        }
