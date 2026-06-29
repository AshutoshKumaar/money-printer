from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def clean_and_normalize(text: str) -> list[str]:
    """Clean and tokenize text, removing common stop words in English and Hindi."""
    text = text.lower()
    # Keep alphanumeric characters and Devanagari Unicode block
    text = re.sub(r"[^a-z0-9\s\u0900-\u097f]+", " ", text)
    words = text.split()
    
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "about", "of", "by",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "this", "that", "these", "those",
        "ka", "ki", "ke", "ko", "me", "par", "se", "ek", "aur", "hota", "hoti", "hote", "hai", "hain", "tha", "thi",
        "के", "में", "की", "का", "है", "हैं", "पर", "को", "से", "एक", "और", "होता", "होती", "होते", "था", "थी",
        "facts", "mysteries", "secrets", "tathya", "rahasya", "तथ्य", "रहस्य", "shorts", "hindi", "youtube", "viral",
        "refined", "strange", "amazing", "unsolved", "mind", "blowing", "dangerous", "mysterious", "deadliest",
        "science", "mystery", "secret", "facts", "legend", "creepy", "scary", "unbelievable", "shocking", "shocked",
        "could", "destroy", "our", "galaxy", "why", "scientists", "place", "on", "earth", "most", "discoveries",
        "behind", "feels", "broken", "superpowers", "avoid", "should", "you", "lost", "crimes", "haunted", "places",
        "three", "two", "five", "ten", "one", "four", "ancient", "inventions", "ahead", "time", "fall", "deep",
        "creature", "historical", "found", "lesser", "known", "natural", "phenomenons", "plants", "world", "underwater",
        "future", "technologies", "may", "change", "your", "life", "signals"
    }
    
    # Filter out stop words and pure numbers
    filtered = []
    for w in words:
        if w in stop_words:
            continue
        if re.match(r"^\d+$", w):
            continue
        filtered.append(w)
        
    return sorted(list(set(filtered)))


def generate_fingerprint(topic: str) -> str:
    """Generate a sorted, clean fingerprint representation of a topic string."""
    cleaned = clean_and_normalize(topic)
    return "|".join(cleaned)


def compute_similarity(f1: str, f2: str) -> float:
    """Compute the overlap similarity coefficient between two fingerprints."""
    w1 = set(f1.split("|")) if f1 else set()
    w2 = set(f2.split("|")) if f2 else set()
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))


@dataclass
class TopicDecision:
    topic: str
    category: str
    is_evergreen: bool = True
    is_trending: bool = False
    rationale: str = ""
    keywords: list[str] = field(default_factory=list)
    fingerprint: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = generate_fingerprint(self.topic)
        if not self.keywords:
            self.keywords = clean_and_normalize(self.topic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "category": self.category,
            "is_evergreen": self.is_evergreen,
            "is_trending": self.is_trending,
            "rationale": self.rationale,
            "keywords": self.keywords,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicDecision:
        return cls(
            topic=data.get("topic", ""),
            category=data.get("category", "unknown"),
            is_evergreen=data.get("is_evergreen", True),
            is_trending=data.get("is_trending", False),
            rationale=data.get("rationale", ""),
            keywords=data.get("keywords", []),
            fingerprint=data.get("fingerprint", ""),
            timestamp=data.get("timestamp", ""),
            run_id=data.get("run_id", ""),
            metadata=data.get("metadata", {}),
        )
