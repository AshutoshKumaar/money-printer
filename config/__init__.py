from .settings import Settings, load_settings

_settings = load_settings()

BASE_DIR = str(_settings.base_dir)
ASSETS_DIR = str(_settings.assets_dir)
AUDIO_DIR = str(_settings.audio_dir)
VIDEO_DIR = str(_settings.video_dir)
IMAGE_DIR = str(_settings.image_dir)
OUTPUT_DIR = str(_settings.final_dir)
GEMINI_API_KEY = _settings.gemini_api_key
PEXELS_API_KEY = _settings.pexels_api_key
VOICE_NAME = _settings.voice_name
VIDEO_RESOLUTION = _settings.video_resolution
SHORTS_TARGET_SECONDS = _settings.shorts_target_seconds
SHORTS_MAX_SECONDS = _settings.shorts_max_seconds
CAPTION_WORDS_PER_CHUNK = _settings.caption_words_per_chunk
RENDER_FPS = _settings.render_fps
FFMPEG_PRESET = _settings.ffmpeg_preset
BACKGROUND_MUSIC_VOLUME = _settings.background_music_volume
FALLBACK_AI_IMAGE_URL = f"{_settings.pollinations_base_url}/{{prompt}}?width=1080&height=1920&nologo=true"

__all__ = [
    "Settings",
    "load_settings",
    "BASE_DIR",
    "ASSETS_DIR",
    "AUDIO_DIR",
    "VIDEO_DIR",
    "IMAGE_DIR",
    "OUTPUT_DIR",
    "GEMINI_API_KEY",
    "PEXELS_API_KEY",
    "VOICE_NAME",
    "VIDEO_RESOLUTION",
    "SHORTS_TARGET_SECONDS",
    "SHORTS_MAX_SECONDS",
    "CAPTION_WORDS_PER_CHUNK",
    "RENDER_FPS",
    "FFMPEG_PRESET",
    "BACKGROUND_MUSIC_VOLUME",
    "FALLBACK_AI_IMAGE_URL",
]
