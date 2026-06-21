from __future__ import annotations

import urllib.parse
from pathlib import Path

import requests

from config import load_settings
from core.logging import configure_logging
from core.models import Segment
from services.image_service import ImageService


def generate_ai_image(prompt: str, filename: str) -> str | None:
    """Backward-compatible AI image helper using the production image service."""
    settings = load_settings()
    settings.validate(require_youtube=False)
    logger = configure_logging(settings.logs_dir)
    output_path = settings.image_dir / filename
    try:
        segment = Segment(text="", subtitle="", image_prompt=prompt)
        generated = ImageService(settings, logger)._generate_one(segment, output_path)
        ImageService(settings, logger)._ensure_vertical(generated)
        return str(generated)
    except Exception as exc:
        print(f"Error generating image: {exc}")
        return None


def download_pexels_video(query: str, filename: str) -> str | None:
    """Optional legacy stock video fallback; AI images are the primary workflow."""
    settings = load_settings()
    if not settings.pexels_api_key:
        print("PEXELS_API_KEY is not set; skipping Pexels video fallback.")
        return None
    output_path = settings.video_dir / filename
    url = f"https://api.pexels.com/videos/search?query={urllib.parse.quote(query)}&per_page=5&orientation=portrait"
    try:
        response = requests.get(
            url,
            headers={"Authorization": settings.pexels_api_key},
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        videos = response.json().get("videos", [])
        if not videos:
            return None
        video_files = videos[0].get("video_files", [])
        selected = next((item.get("link") for item in video_files if item.get("file_type") == "video/mp4"), None)
        if not selected:
            return None
        download = requests.get(selected, stream=True, timeout=settings.request_timeout_seconds)
        download.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in download.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        return str(output_path)
    except Exception as exc:
        print(f"Pexels video fallback failed: {exc}")
        return None


def download_pexels_image(query: str, filename: str) -> str | None:
    settings = load_settings()
    if not settings.pexels_api_key:
        return None
    output_path = settings.image_dir / filename
    url = f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}&per_page=5&orientation=portrait"
    try:
        response = requests.get(
            url,
            headers={"Authorization": settings.pexels_api_key},
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        photos = response.json().get("photos", [])
        if not photos:
            return None
        source = photos[0].get("src", {})
        link = source.get("large2x") or source.get("large") or source.get("original")
        if not link:
            return None
        image = requests.get(link, timeout=settings.request_timeout_seconds)
        image.raise_for_status()
        output_path.write_bytes(image.content)
        return str(output_path)
    except Exception as exc:
        print(f"Pexels image fallback failed: {exc}")
        return None


if __name__ == "__main__":
    print(generate_ai_image("Vertical 9:16 cinematic cat astronaut in deep space", "cat_space_test.jpg"))
