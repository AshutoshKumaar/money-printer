from __future__ import annotations

import json
import logging
from pathlib import Path

from config import Settings
from core.models import GeneratedVideo, Script
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from story.models import NarrativePackage
    from scene.models import ScenePackage
    from visual.models import VisualPackage
from services.caption_service import CaptionService
from services.gemini_service import GeminiService
from services.image_service import ImageService
from services.notification_service import NotificationService
from services.video_service import VideoService
from services.voice_service import VoiceService
from services.youtube_service import YouTubeService
from storage import StorageManager

class ModularGeneratedVideo(GeneratedVideo):
    __slots__ = ("timings",)

    def __init__(
        self,
        topic: str,
        script: Script,
        video_path: Path | None,
        metadata_path: Path,
        thumbnail_path: Path | None = None,
        youtube_url: str | None = None,
        timings: dict | None = None,
    ) -> None:
        self.topic = topic
        self.script = script
        self.video_path = video_path
        self.metadata_path = metadata_path
        self.thumbnail_path = thumbnail_path
        self.youtube_url = youtube_url
        self.timings = timings or {}


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

    def validate_gemini_connectivity(self) -> None:
        self.logger.info("Validating Gemini API key connectivity at startup...")
        from core.retry import retry_call
        try:
            retry_call(
                lambda: self.gemini.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Startup connectivity check.",
                ),
                attempts=self.settings.retry_attempts,
                backoff_seconds=self.settings.retry_backoff_seconds,
                logger=self.logger,
                label="Gemini startup connectivity check",
            )
            self.logger.info("Gemini API connectivity check succeeded")
        except Exception as exc:
            self.logger.error("Gemini API connectivity check failed: %s", exc)
            raise ValueError(f"Startup validation failed: Gemini API key is invalid or unreachable. Error: {exc}") from exc

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

        image_paths = self.images.generate_images(script, paths, use_existing=use_existing_assets, gemini_service=self.gemini)
        audio_paths = self.voice.generate_voiceovers(script, paths, use_existing=use_existing_assets)
        video_path = self.video.render(script, image_paths, audio_paths, paths)
        
        # Clean up temporary scene images and fallback files after successful render (Requirement 2)
        self.images.delete_fallback_cache(paths)

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

    def run_modular(
        self,
        script: Script,
        image_paths: list[Path],
        paths: RunPaths,
        *,
        dry_run: bool = False,
        generate_only: bool = False,
        use_existing_assets: bool = False,
        narrative_package: NarrativePackage | None = None,
        scene_package: ScenePackage | None = None,
        visual_package: VisualPackage | None = None,
    ) -> ModularGeneratedVideo:
        self.logger.info("Running modular compilation pipeline...")
        import time

        self._save_metadata(paths.metadata_path, script, video_path=None, thumbnail_path=None, youtube_url=None)

        if dry_run:
            self.logger.info("Dry run complete. Metadata saved at %s", paths.metadata_path)
            return ModularGeneratedVideo(
                topic=script.topic,
                script=script,
                video_path=None,
                metadata_path=paths.metadata_path,
                thumbnail_path=None,
                youtube_url=None,
                timings={
                    "voice_time": 0.0,
                    "render_time": 0.0,
                    "upload_time": 0.0,
                }
            )

        # 1. Voice generation
        t0 = time.time()
        audio_paths = self.voice.generate_voiceovers(script, paths, use_existing=use_existing_assets)
        voice_time = round(time.time() - t0, 2)
        self.logger.info("Voice generation completed in %.2fs", voice_time)

        # 2. Render final video
        t0 = time.time()
        render_package = None
        if narrative_package and scene_package and visual_package:
            from render.render_engine import RenderEngine
            engine = RenderEngine(self.settings, self.logger)
            render_package = engine.generate_package(
                narrative_package,
                scene_package,
                visual_package,
                [str(p) for p in audio_paths],
            )
            # Save RenderPackage to render.json for debug trace
            self.logger.info("Saving deterministic RenderPackage specification to debug folder...")
            debug_dir = self.settings.storage_dir / "debug" / paths.run_id
            debug_dir.mkdir(parents=True, exist_ok=True)
            try:
                import json
                (debug_dir / "render.json").write_text(json.dumps(render_package.to_dict(), indent=2), encoding="utf-8")
            except Exception as exc:
                self.logger.warning("Failed to save render.json: %s", exc)

        video_path = self.video.render(script, image_paths, audio_paths, paths, render_package=render_package)
        render_time = round(time.time() - t0, 2)
        self.logger.info("Render completed in %.2fs", render_time)

        # Clean up temporary scene images
        self.images.delete_fallback_cache(paths)

        # 3. Upload to YouTube
        t0 = time.time()
        youtube_url = None
        if not generate_only:
            youtube_url = self.youtube.upload(video_path, script, paths.thumbnail_path)
        upload_time = round(time.time() - t0, 2)
        if not generate_only:
            self.logger.info("Upload completed in %.2fs", upload_time)

        self._save_metadata(
            paths.metadata_path,
            script,
            video_path=video_path,
            thumbnail_path=paths.thumbnail_path,
            youtube_url=youtube_url,
        )

        result = ModularGeneratedVideo(
            topic=script.topic,
            script=script,
            video_path=video_path,
            metadata_path=paths.metadata_path,
            thumbnail_path=paths.thumbnail_path,
            youtube_url=youtube_url,
            timings={
                "voice_time": voice_time,
                "render_time": render_time,
                "upload_time": upload_time,
            }
        )

        self.storage.append_topic_history(
            {
                "run_id": paths.run_id,
                "topic": script.topic,
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
