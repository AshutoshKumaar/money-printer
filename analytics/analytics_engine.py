from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings
from topic.models import generate_fingerprint, compute_similarity


class AnalyticsEngine:
    """Calculates factual topic/category metrics based on local upload and performance history."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.history_path = self.settings.storage_dir / "analytics_history.json"
        self._ensure_and_seed_history()

    def _ensure_and_seed_history(self) -> None:
        """Ensure analytics_history.json exists, seeding it from topic_history.json if missing."""
        if self.history_path.exists():
            return

        self.logger.info("analytics_history.json not found. Seeding from topic_history.json...")
        topic_history_path = self.settings.storage_dir / "topic_history.json"
        records: list[dict[str, Any]] = []

        if topic_history_path.exists():
            try:
                topic_history = json.loads(topic_history_path.read_text(encoding="utf-8"))
                if isinstance(topic_history, list):
                    for entry in topic_history:
                        if not isinstance(entry, dict):
                            continue
                        topic = entry.get("topic", "")
                        run_id = entry.get("run_id", "")
                        
                        # Guess category based on topic/title keywords
                        category = self.guess_category(topic, entry.get("title", ""))
                        
                        # Parse upload date from run_id (format: YYYYMMDD-HHMMSS)
                        upload_date = self._parse_run_id_date(run_id)
                        
                        records.append({
                            "run_id": run_id,
                            "topic": topic,
                            "category": category,
                            "upload_date": upload_date,
                            "status": "success" if entry.get("youtube_url") else "generated",
                            "views": None,
                            "retention_rate": None,
                            "engagement_rate": None,
                            "youtube_url": entry.get("youtube_url"),
                            "fingerprint": generate_fingerprint(topic),
                        })
            except Exception as e:
                self.logger.warning("Failed to parse topic_history.json during seeding: %s", e)

        # Write seeded records
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
            self.logger.info("Successfully seeded %d analytics history records.", len(records))
        except Exception as e:
            self.logger.error("Failed to write analytics_history.json: %s", e)

    def _parse_run_id_date(self, run_id: str) -> str:
        """Parse run_id into ISO 8601 string or fallback to current time."""
        try:
            # format: 20260623-170458
            match = re.match(r"(\d{8})-(\d{6})", run_id)
            if match:
                dt = datetime.strptime(run_id, "%Y%m%d-%H%M%S")
                return dt.replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            pass
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def guess_category(topic: str, title: str) -> str:
        """Guess topic category based on topic and title keywords."""
        combined = f"{topic} {title}".lower()
        if any(w in combined for w in ["space", "black hole", "galaxy", "universe", "planet", "stars", "astronaut", "cosmos", "isro", "rocketry"]):
            return "space"
        if any(w in combined for w in ["ocean", "sea", "marine", "underwater", "deep water", "fish", "shark", "abyss"]):
            return "ocean"
        if any(w in combined for w in ["history", "ancient", "war", "pyramid", "disaster", "past", "century", "emperor", "india in", "partition", "nuclear", "taj mahal", "dyatlov"]):
            return "history"
        if any(w in combined for w in ["brain", "human body", "psychology", "body", "science", "physics", "gravity", "technology", "future"]):
            return "science"
        if any(w in combined for w in ["animal", "creature", "nature", "forest", "jungle", "wildlife"]):
            return "nature"
        if any(w in combined for w in ["mystery", "mysteries", "unsolved", "bermuda", "creepy", "scary", "ghost", "horror"]):
            return "horror"
        return "science"

    def load_records(self) -> list[dict[str, Any]]:
        """Load performance records from analytics_history.json."""
        if not self.history_path.exists():
            return []
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.error("Failed to read analytics_history.json: %s", e)
            return []

    def get_category_metrics(self) -> dict[str, dict[str, Any]]:
        """
        Calculate metrics (average views, retention, engagement, and category score) for all categories.
        If a metric is null/None for all items in a category, the averages remain None.
        """
        records = self.load_records()
        by_category: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            cat = r.get("category", "unknown")
            by_category.setdefault(cat, []).append(r)

        metrics: dict[str, dict[str, Any]] = {}
        
        # We need max views to normalize views score if views exist
        all_views = [r["views"] for r in records if r.get("views") is not None]
        max_views = max(all_views) if all_views else 0

        for cat, cat_records in by_category.items():
            views_list = [r["views"] for r in cat_records if r.get("views") is not None]
            ret_list = [r["retention_rate"] for r in cat_records if r.get("retention_rate") is not None]
            eng_list = [r["engagement_rate"] for r in cat_records if r.get("engagement_rate") is not None]

            avg_views = sum(views_list) / len(views_list) if views_list else None
            avg_ret = sum(ret_list) / len(ret_list) if ret_list else None
            avg_eng = sum(eng_list) / len(eng_list) if eng_list else None

            # Category score formula:
            # If no data exists, we assign a baseline score of 1.0 (untouched/neutral).
            # Otherwise, combine normalized views, retention, and engagement.
            if avg_views is None and avg_ret is None and avg_eng is None:
                score = 1.0
            else:
                score_views = (avg_views / max_views) if (avg_views is not None and max_views > 0) else 0.0
                score_ret = avg_ret if avg_ret is not None else 0.0
                score_eng = avg_eng if avg_eng is not None else 0.0
                
                # Formula: 40% views, 40% retention, 20% engagement
                score = (score_views * 40.0) + (score_ret * 40.0) + (score_eng * 20.0)

            metrics[cat] = {
                "avg_views": avg_views,
                "avg_retention": avg_ret,
                "avg_engagement": avg_eng,
                "category_score": score,
                "count": len(cat_records),
            }

        return metrics

    def get_topic_score(self, topic: str) -> float:
        """
        Calculate topic score based on past similar topics.
        Compares fingerprints and scales by performance if similar topics exist.
        Returns a baseline score of 1.0 if no similar topic is found.
        """
        records = self.load_records()
        target_fp = generate_fingerprint(topic)
        
        similar_records = []
        for r in records:
            past_topic = r.get("topic", "")
            past_fp = r.get("fingerprint", "") or generate_fingerprint(past_topic)
            similarity = compute_similarity(target_fp, past_fp)
            if similarity > 0.5:  # threshold for similarity
                similar_records.append((r, similarity))

        if not similar_records:
            return 1.0

        total_weight = 0.0
        weighted_score = 0.0

        all_views = [r["views"] for r in records if r.get("views") is not None]
        max_views = max(all_views) if all_views else 0

        for r, similarity in similar_records:
            views = r.get("views")
            ret = r.get("retention_rate")
            eng = r.get("engagement_rate")

            if views is None and ret is None and eng is None:
                item_score = 1.0
            else:
                v_score = (views / max_views) if (views is not None and max_views > 0) else 0.0
                r_score = ret if ret is not None else 0.0
                e_score = eng if eng is not None else 0.0
                item_score = (v_score * 40.0) + (r_score * 40.0) + (e_score * 20.0)

            weighted_score += item_score * similarity
            total_weight += similarity

        return weighted_score / total_weight if total_weight > 0 else 1.0

    def get_overall_metrics(self) -> dict[str, float | None]:
        """Compute overall average views, retention, and engagement across history."""
        records = self.load_records()
        views_list = [r["views"] for r in records if r.get("views") is not None]
        ret_list = [r["retention_rate"] for r in records if r.get("retention_rate") is not None]
        eng_list = [r["engagement_rate"] for r in records if r.get("engagement_rate") is not None]

        return {
            "avg_views": sum(views_list) / len(views_list) if views_list else None,
            "avg_retention": sum(ret_list) / len(ret_list) if ret_list else None,
            "avg_engagement": sum(eng_list) / len(eng_list) if eng_list else None,
            "total_count": len(records),
        }
