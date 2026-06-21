from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Settings


def slugify(value: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_length] or "hindi-shorts-video"


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    slug: str
    image_dir: Path
    audio_dir: Path
    metadata_path: Path
    video_path: Path
    thumbnail_path: Path


class StorageManager:
    """Owns output paths and safe cleanup rules."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_run(self, topic: str) -> RunPaths:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = slugify(topic)
        image_dir = self.settings.image_dir / run_id
        audio_dir = self.settings.audio_dir / run_id
        image_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        self.settings.final_dir.mkdir(parents=True, exist_ok=True)
        self.settings.metadata_dir.mkdir(parents=True, exist_ok=True)
        return RunPaths(
            run_id=run_id,
            slug=slug,
            image_dir=image_dir,
            audio_dir=audio_dir,
            metadata_path=self.settings.metadata_dir / f"{run_id}-{slug}.metadata.json",
            video_path=self.settings.final_dir / f"{run_id}-{slug}.mp4",
            thumbnail_path=self.settings.final_dir / f"{run_id}-{slug}.jpg",
        )

    def save_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_topic_history(self, limit: int = 50) -> list[dict[str, Any]]:
        path = self.settings.storage_dir / "topic_history.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)][-limit:]

    def append_topic_history(self, entry: dict[str, Any], limit: int = 200) -> None:
        path = self.settings.storage_dir / "topic_history.json"
        history = self.load_topic_history(limit=limit)
        history.append(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history[-limit:], indent=2, ensure_ascii=False), encoding="utf-8")
