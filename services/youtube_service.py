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

        import os
        simulate = os.getenv("SIMULATE_UPLOAD", "").strip().lower() in ("1", "true", "yes", "on")
        if simulate:
            self.logger.info("[SIMULATION] Simulating YouTube upload for: %s", video_path)
            import time
            time.sleep(0.5)
            return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        max_attempts = 4
        backoff_factor = 2.0
        
        for attempt in range(1, max_attempts + 1):
            try:
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

                self.logger.info("Uploading video to YouTube (Attempt %d/%d): %s", attempt, max_attempts, video_path)
                import time
                import json
                from core.telemetry import telemetry_tracker
                from googleapiclient.http import MediaFileUpload

                t_start = time.perf_counter()
                response = None
                
                request = youtube.videos().insert(
                    part="snippet,status",
                    body=body,
                    media_body=MediaFileUpload(str(video_path), chunksize=-1, resumable=True),
                )
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        self.logger.info("YouTube upload progress: %s%%", int(status.progress() * 100))
                
                latency = time.perf_counter() - t_start
                video_id = response["id"]
                
                try:
                    response_size = len(json.dumps(response).encode("utf-8")) if response else 0
                except Exception:
                    response_size = 0

                try:
                    telemetry_tracker.record(
                        stage="upload",
                        provider="YouTube",
                        model="youtube-v3-api",
                        endpoint="videos.insert",
                        attempt_number=attempt,
                        retry_count=attempt - 1,
                        status_code=200,
                        latency=latency,
                        response_size_bytes=response_size,
                        cache_hit=False
                    )
                except Exception:
                    pass

                if thumbnail_path and thumbnail_path.exists():
                    t_thumb_start = time.perf_counter()
                    try:
                        thumb_res = youtube.thumbnails().set(
                            videoId=video_id,
                            media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg"),
                        ).execute()
                        thumb_latency = time.perf_counter() - t_thumb_start
                        
                        try:
                            thumb_response_size = len(json.dumps(thumb_res).encode("utf-8")) if thumb_res else 0
                        except Exception:
                            thumb_response_size = 0
                            
                        try:
                            telemetry_tracker.record(
                                stage="upload",
                                provider="YouTube",
                                model="youtube-v3-api",
                                endpoint="thumbnails.set",
                                attempt_number=1,
                                retry_count=0,
                                status_code=200,
                                latency=thumb_latency,
                                response_size_bytes=thumb_response_size,
                                cache_hit=False
                            )
                        except Exception:
                            pass
                    except Exception as thumb_exc:
                        thumb_latency = time.perf_counter() - t_thumb_start
                        try:
                            status_code = 500
                            if hasattr(thumb_exc, "resp") and thumb_exc.resp:
                                status_code = getattr(thumb_exc.resp, "status", 500)
                            telemetry_tracker.record(
                                stage="upload",
                                provider="YouTube",
                                model="youtube-v3-api",
                                endpoint="thumbnails.set",
                                attempt_number=1,
                                retry_count=0,
                                status_code=status_code,
                                latency=thumb_latency,
                                response_size_bytes=0,
                                cache_hit=False
                            )
                        except Exception:
                            pass
                        raise thumb_exc

                return f"https://www.youtube.com/watch?v={video_id}"
            except Exception as exc:
                self.logger.error("YouTube upload attempt %d failed: %s", attempt, exc)
                if attempt == max_attempts:
                    raise UploadError(f"YouTube upload failed after {max_attempts} attempts: {exc}") from exc
                sleep_time = backoff_factor ** attempt
                self.logger.info("Waiting %.1fs before retrying upload...", sleep_time)
                import time
                time.sleep(sleep_time)

    def _description(self, script: Script) -> str:
        hashtags = " ".join(f"#{tag.lstrip('#')}" for tag in script.hashtags)
        description = script.description.strip()
        if hashtags and hashtags not in description:
            description = f"{description}\n\n{hashtags}"
        return description[:4900]
