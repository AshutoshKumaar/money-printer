from __future__ import annotations

from pathlib import Path

from config import load_settings
from core.logging import configure_logging
from core.models import Script
from services.caption_service import CaptionService
from services.video_service import VideoService
from storage.paths import RunPaths, slugify


def compile_video(scenes_data: list[dict], output_filename: str = "output.mp4") -> str | None:
    """Backward-compatible wrapper around the production video service."""
    settings = load_settings()
    logger = configure_logging(settings.logs_dir)
    script = Script.from_dict(
        {
            "title": Path(output_filename).stem,
            "description": "Generated Hindi Shorts video.",
            "tags": ["shorts", "hindi"],
            "segments": scenes_data,
        },
        Path(output_filename).stem,
    )
    run_id = "legacy"
    slug = slugify(Path(output_filename).stem)
    paths = RunPaths(
        run_id=run_id,
        slug=slug,
        image_dir=settings.image_dir,
        audio_dir=settings.audio_dir,
        metadata_path=settings.metadata_dir / f"{run_id}-{slug}.metadata.json",
        video_path=settings.final_dir / Path(output_filename).name,
        thumbnail_path=settings.final_dir / f"{slug}.jpg",
    )
    image_paths = [settings.image_dir / f"scene_{index}.jpg" for index in range(1, len(script.segments) + 1)]
    audio_paths = [settings.audio_dir / f"scene_{index}.mp3" for index in range(1, len(script.segments) + 1)]
    try:
        return str(VideoService(settings, logger, CaptionService(settings)).render(script, image_paths, audio_paths, paths))
    except Exception as exc:
        print(f"Error compiling final video: {exc}")
        return None
