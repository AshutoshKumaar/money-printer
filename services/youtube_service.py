from __future__ import annotations

import logging
from pathlib import Path

from config import Settings
from core.exceptions import UploadError
from core.models import Script
from services.gemini_service import GeminiService
from services.youtube_auth import YouTubeAuth


class YouTubeService:
    """Uploads rendered Shorts with YouTube Data API v3 OAuth 2.0."""

    def __init__(self, settings: Settings, logger: logging.Logger, gemini: GeminiService | None = None) -> None:
        self.settings = settings
        self.logger = logger
        self.auth = YouTubeAuth(settings, logger)
        self.gemini = gemini

    def validate_authentication(self) -> None:
        self.auth.validate_authentication()

    def upload(self, video_path: Path, script: Script, thumbnail_path: Path | None = None) -> str:
        if not video_path.exists():
            raise UploadError(f"Video file not found: {video_path}")
        if self.gemini:
            script = self.gemini.improve_metadata(script)
        youtube = self.auth.get_authenticated_service()
        body = {
            "snippet": {
                "title": script.title[:100],
                "description": self._description(script),
                "tags": script.tags[:30],
                "categoryId": "27",
            },
            "status": {"privacyStatus": self.settings.youtube_privacy_status},
        }

        self.logger.info("Uploading video to YouTube: %s", video_path)
        try:
            from googleapiclient.http import MediaFileUpload

            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=MediaFileUpload(str(video_path), chunksize=-1, resumable=True),
            )
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    self.logger.info("YouTube upload progress: %s%%", int(status.progress() * 100))
            video_id = response["id"]
            if thumbnail_path and thumbnail_path.exists():
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg"),
                ).execute()
            return f"https://www.youtube.com/watch?v={video_id}"
        except Exception as exc:
            raise UploadError(f"YouTube upload failed: {exc}") from exc

    def _description(self, script: Script) -> str:
        hashtags = " ".join(f"#{tag.lstrip('#')}" for tag in script.hashtags)
        description = script.description.strip()
        if hashtags and hashtags not in description:
            description = f"{description}\n\n{hashtags}"
        return description[:4900]
