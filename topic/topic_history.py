from __future__ import annotations

import logging
from typing import Any

from config import Settings
from topic.models import generate_fingerprint, compute_similarity, TopicDecision
from analytics.analytics_engine import AnalyticsEngine


class TopicHistory:
    """Manages history search, duplicate detection, and recent category/topic lookups using analytics history."""

    def __init__(self, settings: Settings, logger: logging.Logger, analytics_engine: AnalyticsEngine) -> None:
        self.settings = settings
        self.logger = logger
        self.analytics_engine = analytics_engine

    def is_duplicate(self, candidate_topic: str, threshold: float = 0.5) -> tuple[bool, float, str | None]:
        """
        Check if a candidate topic is a duplicate or close variation of any previously used topic.
        Returns (is_duplicate, max_similarity, closest_topic_name).
        """
        records = self.analytics_engine.load_records()
        if not records:
            return False, 0.0, None

        candidate_fp = generate_fingerprint(candidate_topic)
        max_sim = 0.0
        closest_topic = None

        for r in records:
            past_topic = r.get("topic", "")
            past_fp = r.get("fingerprint", "") or generate_fingerprint(past_topic)
            similarity = compute_similarity(candidate_fp, past_fp)
            if similarity > max_sim:
                max_sim = similarity
                closest_topic = past_topic

        is_dup = max_sim >= threshold
        return is_dup, max_sim, closest_topic

    def get_recent_categories(self, limit: int = 5) -> list[str]:
        """Get the categories used in the most recent 'limit' runs, sorted newest to oldest."""
        records = self.analytics_engine.load_records()
        if not records:
            return []

        # Sort by upload_date descending (newest first)
        sorted_records = sorted(
            records,
            key=lambda x: x.get("upload_date", ""),
            reverse=True
        )
        
        recent_cats: list[str] = []
        for r in sorted_records:
            cat = r.get("category")
            if cat and cat not in recent_cats:
                recent_cats.append(cat)
                if len(recent_cats) >= limit:
                    break
        return recent_cats

    def get_recent_topics(self, limit: int = 50) -> list[str]:
        """Get the topic strings of the most recent 'limit' runs, sorted newest to oldest."""
        records = self.analytics_engine.load_records()
        if not records:
            return []

        sorted_records = sorted(
            records,
            key=lambda x: x.get("upload_date", ""),
            reverse=True
        )
        return [r.get("topic", "") for r in sorted_records[:limit] if r.get("topic")]
