from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from config import Settings
from core.models import GeneratedVideo
from topic.models import TopicDecision, generate_fingerprint


class FeedbackEngine:
    """Saves factual run metadata to analytics history, paving the way for future performance imports."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.history_path = self.settings.storage_dir / "analytics_history.json"

    def save_run_performance(self, result: GeneratedVideo, topic_decision: TopicDecision) -> None:
        """Save a completed run's metadata into analytics_history.json, keeping metrics null."""
        records: list[dict[str, Any]] = []
        if self.history_path.exists():
            try:
                records = json.loads(self.history_path.read_text(encoding="utf-8"))
            except Exception as e:
                self.logger.error("Failed to read analytics_history.json during save: %s", e)

        run_id = topic_decision.run_id or (result.metadata_path.stem.split("-")[0] if result.metadata_path else "")
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Prepare new performance record
        new_record = {
            "run_id": run_id,
            "topic": topic_decision.topic,
            "category": topic_decision.category,
            "upload_date": topic_decision.timestamp or datetime.now(timezone.utc).isoformat(),
            "status": "success" if result.youtube_url else "generated",
            "views": None,
            "retention_rate": None,
            "engagement_rate": None,
            "youtube_url": result.youtube_url,
            "fingerprint": topic_decision.fingerprint or generate_fingerprint(topic_decision.topic),
        }

        # Check for existing run_id to update, otherwise append
        updated = False
        for idx, r in enumerate(records):
            if r.get("run_id") == run_id:
                # Update existing record (preserving existing views/retention if already imported)
                records[idx].update({
                    "topic": new_record["topic"],
                    "category": new_record["category"],
                    "status": new_record["status"],
                    "youtube_url": new_record["youtube_url"],
                    "fingerprint": new_record["fingerprint"],
                })
                updated = True
                break

        if not updated:
            records.append(new_record)

        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
            self.logger.info("Factual run performance metrics saved for run_id %s.", run_id)
        except Exception as e:
            self.logger.error("Failed to write to analytics_history.json: %s", e)
