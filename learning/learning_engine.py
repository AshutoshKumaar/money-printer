from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Settings
from learning.models import LearningState


class LearningEngine:
    """Deterministic, self-healing Learning Engine that optimizes topic and planning weights based on historical performance."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.history_path = self.settings.storage_dir / "analytics_history.json"
        self.state_path = self.settings.storage_dir / "learning_state.json"

    def get_default_state(self) -> LearningState:
        """Return the default learning state with equalized weights and zero confidence."""
        # Detect categories dynamically from storage/topics/
        categories = ["history", "horror", "nature", "ocean", "science", "space"]
        topics_dir = self.settings.storage_dir / "topics"
        if topics_dir.exists():
            found_cats = [p.stem for p in topics_dir.glob("*.json")]
            if found_cats:
                categories = found_cats

        num_cats = len(categories)
        cat_weights = {cat: 1.0 / num_cats for cat in categories}
        
        # Default weights for visual strategies
        visual_weights = {
            "stock_only": 0.5,
            "ai_preferred": 0.5,
            "ai_required": 0.5,
            "archival": 0.5,
            "hybrid": 0.5,
        }

        # Default pacing weights
        pacing_weights = {
            "fast": 0.5,
            "balanced": 0.5,
            "slow": 0.5,
        }

        # Upload schedule hourly weights (0-23)
        schedule_weights = {str(hour): 1.0 for hour in range(24)}

        # Default topic weights
        topic_weights = {}

        # Default confidence scores
        confidence_scores = {cat: 0.0 for cat in categories}
        confidence_scores["overall"] = 0.0

        return LearningState(
            category_weights=cat_weights,
            topic_weights=topic_weights,
            pacing_weights=pacing_weights,
            visual_weights=visual_weights,
            upload_schedule_weights=schedule_weights,
            confidence_scores=confidence_scores,
        )

    def load_previous_state(self) -> LearningState:
        """Load the existing learning state, returning default if missing or invalid."""
        if not self.state_path.exists():
            return self.get_default_state()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return LearningState(
                category_weights=data.get("category_weights", {}),
                topic_weights=data.get("topic_weights", {}),
                pacing_weights=data.get("pacing_weights", {}),
                visual_weights=data.get("visual_weights", {}),
                upload_schedule_weights=data.get("upload_schedule_weights", {}),
                confidence_scores=data.get("confidence_scores", {}),
            )
        except Exception as e:
            self.logger.warning("Failed to load previous learning state: %s. Reverting to default.", e)
            return self.get_default_state()

    def train(self) -> LearningState:
        """Analyze histories, calculate rolling metrics, update weights, and save learning_state.json."""
        self.logger.info("Running deterministic Learning Engine train pass...")

        prev_state = self.load_previous_state()
        
        # 1. Load and parse analytics history (self-healing for corruption)
        records: list[dict[str, Any]] = []
        if self.history_path.exists():
            try:
                content = self.history_path.read_text(encoding="utf-8")
                records = json.loads(content)
            except Exception as e:
                self.logger.warning("analytics_history.json is corrupted: %s. Recovery triggered.", e)
                # Attempt to recover partial JSON lines if possible
                try:
                    recovered = []
                    for line in content.split("\n"):
                        clean_line = line.strip().rstrip(",").lstrip(",")
                        if clean_line.startswith("{") and clean_line.endswith("}"):
                            recovered.append(json.loads(clean_line))
                    records = recovered
                    self.logger.info("Successfully recovered %d records from corrupted history", len(records))
                except Exception:
                    self.logger.error("Could not recover corrupted history file. Reverting to empty.")
                    records = []

        # 2. Filter valid performance observations (require views to be synced)
        valid_records = [r for r in records if r.get("views") is not None]
        
        # Insufficient data check (requires at least 3 valid records)
        if len(valid_records) < 3:
            self.logger.warning("Insufficient data (%d valid runs, need 3). Preserving existing weights.", len(valid_records))
            # Save the current state to ensure learning_state.json exists
            self.save_state(prev_state)
            return prev_state

        # 3. Category Weight Learning
        # Calculate average performance score per category
        category_scores: dict[str, list[float]] = {}
        for r in valid_records:
            cat = r.get("category", "unknown")
            # Calculate performance score if not stored
            perf = r.get("performance_score")
            if perf is None:
                # Fallback calculation
                views = float(r.get("views", 0))
                ret = float(r.get("retention_rate") or r.get("retention_score") or 0.6)
                eng = float(r.get("engagement_rate") or 0.05)
                perf = (ret * 100.0 * 0.4) + (eng * 100.0 * 4.0) + (min(views, 10000) / 10000.0 * 40.0)
            category_scores.setdefault(cat, []).append(perf)

        # Update category weights based on average performance
        default_state = self.get_default_state()
        new_cat_weights = dict(prev_state.category_weights)
        new_confidence = dict(prev_state.confidence_scores)
        
        total_score_sum = 0.0
        calculated_categories = {}
        
        for cat, scores in category_scores.items():
            avg_score = sum(scores) / len(scores)
            calculated_categories[cat] = max(0.1, avg_score)
            total_score_sum += calculated_categories[cat]
            # Confidence grows with sample size
            new_confidence[cat] = min(len(scores) / 10.0, 1.0)

        # Normalize weights
        if total_score_sum > 0.0:
            for cat in new_cat_weights:
                if cat in calculated_categories:
                    new_cat_weights[cat] = calculated_categories[cat] / total_score_sum
                else:
                    # Baseline weight for unvisited categories is always read from the constant default state
                    new_cat_weights[cat] = default_state.category_weights.get(cat, 0.1)
            
            # Normalize entire dict to sum to 1.0
            sum_w = sum(new_cat_weights.values())
            for cat in new_cat_weights:
                new_cat_weights[cat] /= sum_w

        # 4. Visual Strategy Preference Learning
        visual_scores: dict[str, list[float]] = {}
        for r in valid_records:
            run_id = r.get("run_id")
            perf = r.get("performance_score", 50.0)
            # Try to load visual.json from debug folder to extract source strategy
            visual_path = self.settings.storage_dir / "debug" / run_id / "visual.json"
            if visual_path.exists():
                try:
                    v_data = json.loads(visual_path.read_text(encoding="utf-8"))
                    # Analyze resolved assets strategies
                    for asset in v_data.get("assets", []):
                        strategy = asset.get("source", "stock")
                        visual_scores.setdefault(strategy, []).append(perf)
                except Exception:
                    pass

        new_visual_weights = dict(prev_state.visual_weights)
        for strategy, scores in visual_scores.items():
            if scores:
                new_visual_weights[strategy] = sum(scores) / len(scores)

        # 5. Pacing Weights Learning
        pacing_scores: dict[str, list[float]] = {}
        for r in valid_records:
            run_id = r.get("run_id")
            perf = r.get("performance_score", 50.0)
            story_path = self.settings.storage_dir / "debug" / run_id / "story.json"
            if story_path.exists():
                try:
                    s_data = json.loads(story_path.read_text(encoding="utf-8"))
                    pacing = s_data.get("quality", {}).get("pacing_score", 0.5)
                    key = "balanced"
                    if pacing > 0.7:
                        key = "fast"
                    elif pacing < 0.4:
                        key = "slow"
                    pacing_scores.setdefault(key, []).append(perf)
                except Exception:
                    pass

        new_pacing_weights = dict(prev_state.pacing_weights)
        for key, scores in pacing_scores.items():
            if scores:
                new_pacing_weights[key] = sum(scores) / len(scores)

        # 6. Upload Schedule Weight Learning
        hourly_scores: dict[str, list[float]] = {}
        for r in valid_records:
            pub_str = r.get("publish_time") or r.get("upload_date")
            perf = r.get("performance_score", 50.0)
            if pub_str:
                try:
                    dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    hour_key = str(dt.hour)
                    hourly_scores.setdefault(hour_key, []).append(perf)
                except Exception:
                    pass

        new_schedule_weights = dict(prev_state.upload_schedule_weights)
        for hour_key, scores in hourly_scores.items():
            if scores:
                new_schedule_weights[hour_key] = sum(scores) / len(scores)

        # 7. Topic/Template weights learning
        new_topic_weights = dict(prev_state.topic_weights)
        topic_scores: dict[str, list[float]] = {}
        for r in valid_records:
            topic = r.get("topic", "")
            perf = r.get("performance_score", 50.0)
            if topic:
                topic_scores.setdefault(topic, []).append(perf)
        for t, scores in topic_scores.items():
            new_topic_weights[t] = sum(scores) / len(scores)

        # Update overall confidence
        new_confidence["overall"] = min(len(valid_records) / 10.0, 1.0)

        learned_state = LearningState(
            category_weights=new_cat_weights,
            topic_weights=new_topic_weights,
            pacing_weights=new_pacing_weights,
            visual_weights=new_visual_weights,
            upload_schedule_weights=new_schedule_weights,
            confidence_scores=new_confidence,
        )

        self.save_state(learned_state)
        return learned_state

    def save_state(self, state: LearningState) -> None:
        """Save LearningState to learning_state.json."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            self.logger.info("Successfully saved deterministic learning_state.json specification.")
        except Exception as e:
            self.logger.error("Failed to write learning_state.json: %s", e)
