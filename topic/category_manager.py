from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from config import Settings
from topic.topic_history import TopicHistory
from analytics.analytics_engine import AnalyticsEngine


class CategoryManager:
    """Manages allowed topic categories, loading from JSON files, and selecting the best category using rotation rules."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.topics_dir = self.settings.storage_dir / "topics"
        self.categories: dict[str, dict[str, Any]] = {}
        self._load_categories()

    def _load_categories(self) -> None:
        """Dynamically load category JSON files from storage/topics/."""
        if not self.topics_dir.exists():
            self.logger.warning("Topics directory %s does not exist.", self.topics_dir)
            return

        for p in self.topics_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                cat = data.get("category")
                if cat:
                    self.categories[cat] = data
            except Exception as e:
                self.logger.error("Failed to load category file %s: %s", p.name, e)

        if not self.categories:
            self.logger.warning("No categories loaded from JSON. Please ensure storage/topics/ has category JSON files.")

    def select_category(self, topic_history: TopicHistory, analytics: AnalyticsEngine) -> str:
        """
        Select a category based on rotation rules and historical performance.
        Avoids recently used categories and prioritizes historically successful ones.
        """
        allowed_cats = list(self.categories.keys())
        if not allowed_cats:
            raise ValueError("No topic categories available in CategoryManager")

        # Avoid categories used in the last len(allowed_cats)//2 runs to ensure rotation
        rotation_limit = max(1, len(allowed_cats) // 2)
        recent_cats = topic_history.get_recent_categories(limit=rotation_limit)
        available_cats = [c for c in allowed_cats if c not in recent_cats]

        if not available_cats:
            self.logger.info("All categories recently used. Resetting rotation rules.")
            available_cats = allowed_cats

        # Get performance scores from AnalyticsEngine
        metrics = analytics.get_category_metrics()
        
        # Check if learned weights exist
        learning_state_path = self.settings.storage_dir / "learning_state.json"
        learned_weights = {}
        if learning_state_path.exists():
            try:
                state_data = json.loads(learning_state_path.read_text(encoding="utf-8"))
                learned_weights = state_data.get("category_weights", {})
            except Exception:
                pass

        # Build selection weights
        choices = []
        weights = []
        for cat in available_cats:
            if cat in learned_weights:
                weight = max(0.01, float(learned_weights[cat]))
            else:
                cat_score = metrics.get(cat, {}).get("category_score", 1.0)
                # Ensure weight is strictly positive
                weight = max(0.1, float(cat_score))
            choices.append(cat)
            weights.append(weight)

        self.logger.info("Category selection pool: %s with weights: %s", choices, [round(w, 2) for w in weights])
        
        # Perform weighted selection
        selected = random.choices(choices, weights=weights, k=1)[0]
        self.logger.info("Selected Category: '%s'", selected)
        return selected

    def generate_candidates(self, category: str) -> list[str]:
        """
        Generate candidate topics for a given category.
        Combines direct evergreen topics and templated subjects.
        """
        cat_data = self.categories.get(category)
        if not cat_data:
            return []

        candidates = list(cat_data.get("evergreen_topics", []))
        templates = cat_data.get("templates", [])
        subjects = cat_data.get("subjects", [])

        # Generate templated topics
        for temp in templates:
            for sub in subjects:
                try:
                    candidates.append(temp.format(subject=sub))
                except Exception:
                    pass

        # De-duplicate pool and preserve order
        seen = set()
        unique_candidates = []
        for c in candidates:
            c_clean = c.strip()
            if c_clean and c_clean not in seen:
                seen.add(c_clean)
                unique_candidates.append(c_clean)

        return unique_candidates
