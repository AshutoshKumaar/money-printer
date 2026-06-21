from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    base_dir: Path
    assets_dir: Path
    audio_dir: Path
    image_dir: Path
    video_dir: Path
    final_dir: Path
    metadata_dir: Path
    logs_dir: Path
    storage_dir: Path
    credentials_dir: Path
    gemini_api_key: str | None
    pexels_api_key: str | None
    youtube_client_secrets_file: Path | None
    youtube_client_secrets_json: str | None
    youtube_client_secrets_base64: str | None
    youtube_token_file: Path
    youtube_token_json: str | None
    youtube_token_base64: str | None
    youtube_privacy_status: str
    notification_webhook_url: str | None
    schedule_time: str
    timezone: str
    voice_name: str
    voice_rate: str
    auto_fit_voice_duration: bool
    video_resolution: tuple[int, int]
    shorts_target_seconds: int
    shorts_max_seconds: int
    min_segments: int
    max_segments: int
    narration_max_words: int
    segment_max_words: int
    caption_words_per_chunk: int
    render_fps: int
    ffmpeg_preset: str
    background_music_volume: float
    voice_volume: float
    request_timeout_seconds: int
    retry_attempts: int
    retry_backoff_seconds: float
    image_provider: str
    gemini_image_model: str
    hf_token: str | None
    hf_image_model: str
    hf_provider: str | None
    hf_image_width: int
    hf_image_height: int
    hf_max_images_per_video: int
    visual_min_confidence: float
    visual_cache_dir: Path
    pexels_max_results: int
    pollinations_base_url: str
    background_music_url: str | None

    def ensure_directories(self) -> None:
        for folder in (
            self.assets_dir,
            self.audio_dir,
            self.image_dir,
            self.video_dir,
            self.final_dir,
            self.metadata_dir,
            self.logs_dir,
            self.storage_dir,
            self.credentials_dir,
            self.visual_cache_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def validate(self, require_youtube: bool = False) -> None:
        """Fail fast for required credentials and invalid runtime settings."""
        missing: list[str] = []
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")

        if require_youtube:
            has_client_secret_env = bool(self.youtube_client_secrets_json or self.youtube_client_secrets_base64)
            if not self.youtube_client_secrets_file:
                if not has_client_secret_env:
                    missing.append("YOUTUBE_CLIENT_SECRETS_FILE")
            elif not self.youtube_client_secrets_file.exists() and not has_client_secret_env:
                missing.append(f"YOUTUBE_CLIENT_SECRETS_FILE not found: {self.youtube_client_secrets_file}")
            if not self.youtube_token_file.parent.exists():
                missing.append(f"YouTube token directory does not exist: {self.youtube_token_file.parent}")

        if self.min_segments < 8:
            missing.append("MIN_SEGMENTS must be at least 8")
        if self.max_segments < self.min_segments:
            missing.append("MAX_SEGMENTS must be greater than or equal to MIN_SEGMENTS")
        if self.shorts_target_seconds > self.shorts_max_seconds:
            missing.append("SHORTS_TARGET_SECONDS must be <= SHORTS_MAX_SECONDS")
        if self.narration_max_words < self.min_segments * 6:
            missing.append("NARRATION_MAX_WORDS is too small for the configured MIN_SEGMENTS")
        if self.segment_max_words < 6:
            missing.append("SEGMENT_MAX_WORDS must be at least 6")
        if not re.fullmatch(r"[+-]\d+%", self.voice_rate):
            missing.append("VOICE_RATE must look like +12% or -5%")
        if not re.fullmatch(r"\d{2}:\d{2}", self.schedule_time):
            missing.append("SCHEDULE_TIME must use 24-hour HH:MM format, for example 18:00")

        if missing:
            joined = "\n  - ".join(missing)
            raise ValueError(
                "Startup validation failed. Fix these settings in .env:\n"
                f"  - {joined}"
            )


def _path_env(name: str, base_dir: Path, default: str | None = None) -> Path | None:
    value = os.getenv(name, default)
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _is_cloud_runtime() -> bool:
    return any(
        os.getenv(marker)
        for marker in (
            "RAILWAY_ENVIRONMENT",
            "RAILWAY_PROJECT_ID",
            "RENDER",
            "RENDER_SERVICE_ID",
            "K_SERVICE",
            "DYNO",
        )
    )


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load .env and construct a validated settings object."""
    base_dir = Path(__file__).resolve().parent.parent
    dotenv_path = Path(env_file) if env_file else base_dir / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)

    final_dir = _path_env("FINAL_SHORTS_DIR", base_dir, "final_shorts")
    metadata_dir = _path_env("METADATA_DIR", base_dir, "storage/metadata")
    storage_dir = _path_env("STORAGE_DIR", base_dir, "storage")
    credentials_dir = _path_env("CREDENTIALS_DIR", base_dir, "storage/credentials")
    client_secrets_file = _path_env("YOUTUBE_CLIENT_SECRETS_FILE", base_dir, "storage/credentials/client_secret.json")
    token_file = _path_env("YOUTUBE_TOKEN_FILE", base_dir, "storage/credentials/youtube_token.json")
    cloud_runtime = _is_cloud_runtime()
    default_width = 720 if cloud_runtime else 1080
    default_height = 1280 if cloud_runtime else 1920
    default_fps = 20 if cloud_runtime else 24
    default_preset = "ultrafast" if cloud_runtime else "veryfast"

    settings = Settings(
        base_dir=base_dir,
        assets_dir=_path_env("ASSETS_DIR", base_dir, "assets") or base_dir / "assets",
        audio_dir=_path_env("AUDIO_DIR", base_dir, "assets/audio") or base_dir / "assets/audio",
        image_dir=_path_env("IMAGE_DIR", base_dir, "assets/images") or base_dir / "assets/images",
        video_dir=_path_env("VIDEO_DIR", base_dir, "assets/video") or base_dir / "assets/video",
        final_dir=final_dir or base_dir / "final_shorts",
        metadata_dir=metadata_dir or base_dir / "storage/metadata",
        logs_dir=_path_env("LOGS_DIR", base_dir, "logs") or base_dir / "logs",
        storage_dir=storage_dir or base_dir / "storage",
        credentials_dir=credentials_dir or base_dir / "storage/credentials",
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        pexels_api_key=os.getenv("PEXELS_API_KEY"),
        youtube_client_secrets_file=client_secrets_file,
        youtube_client_secrets_json=os.getenv("YOUTUBE_CLIENT_SECRETS_JSON"),
        youtube_client_secrets_base64=os.getenv("YOUTUBE_CLIENT_SECRETS_BASE64"),
        youtube_token_file=token_file or base_dir / "storage/credentials/youtube_token.json",
        youtube_token_json=os.getenv("YOUTUBE_TOKEN_JSON"),
        youtube_token_base64=os.getenv("YOUTUBE_TOKEN_BASE64"),
        youtube_privacy_status=os.getenv("YOUTUBE_PRIVACY_STATUS", "private"),
        notification_webhook_url=os.getenv("NOTIFICATION_WEBHOOK_URL"),
        schedule_time=os.getenv("SCHEDULE_TIME", "18:00"),
        timezone=os.getenv("TIMEZONE", "Asia/Kolkata"),
        voice_name=os.getenv("VOICE_NAME", "hi-IN-MadhurNeural"),
        voice_rate=os.getenv("VOICE_RATE", "+12%"),
        auto_fit_voice_duration=_bool_env("AUTO_FIT_VOICE_DURATION", True),
        video_resolution=(
            _int_env("VIDEO_WIDTH", default_width),
            _int_env("VIDEO_HEIGHT", default_height),
        ),
        shorts_target_seconds=_int_env("SHORTS_TARGET_SECONDS", 58),
        shorts_max_seconds=_int_env("SHORTS_MAX_SECONDS", 60),
        min_segments=_int_env("MIN_SEGMENTS", 10),
        max_segments=_int_env("MAX_SEGMENTS", 12),
        narration_max_words=_int_env("NARRATION_MAX_WORDS", 132),
        segment_max_words=_int_env("SEGMENT_MAX_WORDS", 12),
        caption_words_per_chunk=_int_env("CAPTION_WORDS_PER_CHUNK", 3),
        render_fps=_int_env("RENDER_FPS", default_fps),
        ffmpeg_preset=os.getenv("FFMPEG_PRESET", default_preset),
        background_music_volume=_float_env("BACKGROUND_MUSIC_VOLUME", 0.065),
        voice_volume=_float_env("VOICE_VOLUME", 1.0),
        request_timeout_seconds=_int_env("REQUEST_TIMEOUT_SECONDS", 30),
        retry_attempts=_int_env("RETRY_ATTEMPTS", 3),
        retry_backoff_seconds=_float_env("RETRY_BACKOFF_SECONDS", 1.5),
        image_provider=os.getenv("IMAGE_PROVIDER", "gemini").strip().lower(),
        gemini_image_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"),
        hf_token=os.getenv("HF_TOKEN"),
        hf_image_model=os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell"),
        hf_provider=os.getenv("HF_PROVIDER", "fal-ai") or None,
        hf_image_width=_int_env("HF_IMAGE_WIDTH", 576),
        hf_image_height=_int_env("HF_IMAGE_HEIGHT", 1024),
        hf_max_images_per_video=_int_env("HF_MAX_IMAGES_PER_VIDEO", 3),
        visual_min_confidence=_float_env("VISUAL_MIN_CONFIDENCE", 0.65),
        visual_cache_dir=_path_env("VISUAL_CACHE_DIR", base_dir, "storage/visual_cache")
        or base_dir / "storage/visual_cache",
        pexels_max_results=_int_env("PEXELS_MAX_RESULTS", 8),
        pollinations_base_url=os.getenv("POLLINATIONS_BASE_URL", "https://image.pollinations.ai/prompt"),
        background_music_url=os.getenv(
            "BACKGROUND_MUSIC_URL",
            "https://www.chosic.com/wp-content/uploads/2022/10/Horror-Long-Version(chosic.com).mp3",
        ),
    )
    settings.ensure_directories()
    return settings
