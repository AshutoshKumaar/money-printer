from __future__ import annotations

import json
import logging
from pathlib import Path

from config import Settings
from core.models import GeneratedVideo, Script
from services.caption_service import CaptionService
from services.gemini_service import GeminiService
from services.image_service import ImageService
from services.notification_service import NotificationService
from services.video_service import VideoService
from services.voice_service import VoiceService
from services.youtube_service import YouTubeService
from storage import StorageManager


class ShortsPipeline:
    """Coordinates script, assets, render, metadata, and YouTube upload."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.storage = StorageManager(settings)
        self.gemini = GeminiService(settings, logger)
        self.images = ImageService(settings, logger)
        self.voice = VoiceService(settings, logger)
        self.captions = CaptionService(settings)
        self.video = VideoService(settings, logger, self.captions)
        self.youtube = YouTubeService(settings, logger, self.gemini)
        self.notifications = NotificationService(settings, logger)

    def validate_youtube_authentication(self) -> None:
        self.youtube.validate_authentication()

    def run(
        self,
        topic: str | None,
        *,
        dry_run: bool = False,
        generate_only: bool = False,
        use_existing_assets: bool = False,
    ) -> GeneratedVideo:
        recent_history = self.storage.load_topic_history(limit=50)
        recent_topics = [str(item.get("topic", "")).strip() for item in recent_history if item.get("topic")]
        topic = topic or self.gemini.generate_topic(recent_topics)
        self.logger.info("Starting Hindi Shorts automation for topic: %s", topic)
        paths = self.storage.create_run(topic)

        script = self.gemini.generate_script(topic)
        script = self.gemini.improve_metadata(script)
        self._save_metadata(paths.metadata_path, script, video_path=None, thumbnail_path=None, youtube_url=None)

        if dry_run:
            self.logger.info("Dry run complete. Metadata saved at %s", paths.metadata_path)
            return GeneratedVideo(topic, script, None, paths.metadata_path)

        image_paths = self.images.generate_images(script, paths, use_existing=use_existing_assets)
        audio_paths = self.voice.generate_voiceovers(script, paths, use_existing=use_existing_assets)
        video_path = self.video.render(script, image_paths, audio_paths, paths)

        youtube_url = None
        if not generate_only:
            youtube_url = self.youtube.upload(video_path, script, paths.thumbnail_path)

        self._save_metadata(
            paths.metadata_path,
            script,
            video_path=video_path,
            thumbnail_path=paths.thumbnail_path,
            youtube_url=youtube_url,
        )
        result = GeneratedVideo(topic, script, video_path, paths.metadata_path, paths.thumbnail_path, youtube_url)
        self.storage.append_topic_history(
            {
                "run_id": paths.run_id,
                "topic": topic,
                "title": script.title,
                "video_path": str(video_path),
                "metadata_path": str(paths.metadata_path),
                "youtube_url": youtube_url,
            }
        )
        self.notifications.send(
            "Hindi Shorts automation succeeded",
            f"Video: {video_path}" + (f"\nYouTube: {youtube_url}" if youtube_url else ""),
            success=True,
        )
        return result

    def upload_existing(self, video_path: Path, metadata_path: Path, thumbnail_path: Path | None = None) -> str:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        script_payload = payload.get("script", payload)
        script = Script.from_dict(script_payload, script_payload.get("topic", video_path.stem))
        thumbnail = thumbnail_path if thumbnail_path and thumbnail_path.exists() else video_path.with_suffix(".jpg")
        youtube_url = self.youtube.upload(video_path, script, thumbnail if thumbnail.exists() else None)
        payload["youtube_url"] = youtube_url
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.notifications.send("Hindi Shorts upload succeeded", youtube_url, success=True)
        return youtube_url

    def scheduled_job(self) -> None:
        try:
            self.settings.validate(require_youtube=True)
            self.run(topic=None, dry_run=False, generate_only=False, use_existing_assets=False)
        except Exception as exc:
            self.logger.exception("Scheduled run failed")
            self.notifications.send("Hindi Shorts automation failed", str(exc), success=False)

    def _save_metadata(
        self,
        metadata_path: Path,
        script: Script,
        *,
        video_path: Path | None,
        thumbnail_path: Path | None,
        youtube_url: str | None,
    ) -> None:
        payload = {
            "topic": script.topic,
            "title": script.title,
            "description": script.description,
            "tags": script.tags,
            "hashtags": script.hashtags,
            "video_path": str(video_path) if video_path else None,
            "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
            "youtube_url": youtube_url,
            "script": script.to_dict(),
        }
        self.storage.save_json(metadata_path, payload)
