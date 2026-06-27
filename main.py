from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import load_settings
from core.logging import configure_logging
from director import Director
from scheduler.daily_scheduler import DailyScheduler


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production AI Hindi Shorts automation: script, images, voice, captions, render, and YouTube upload."
    )
    parser.add_argument("topic", nargs="*", help="Topic for the Shorts video. If omitted, Gemini generates one.")
    parser.add_argument("--dry-run", action="store_true", help="Generate script and metadata only; no assets, render, or upload.")
    parser.add_argument("--generate-only", action="store_true", help="Generate the video locally without YouTube upload.")
    parser.add_argument("--upload-only", action="store_true", help="Upload an existing video using saved metadata.")
    parser.add_argument("--auth-only", action="store_true", help="Authenticate YouTube and create/refresh youtube_token.json, then exit.")
    parser.add_argument("--use-existing-assets", action="store_true", help="Reuse run asset files when present.")
    parser.add_argument("--schedule", action="store_true", help="Run daily automation at SCHEDULE_TIME, default 18:00.")
    parser.add_argument("--video-path", type=Path, help="Video path for --upload-only.")
    parser.add_argument("--metadata-path", type=Path, help="Metadata JSON path for --upload-only.")
    parser.add_argument("--thumbnail-path", type=Path, help="Optional thumbnail path for --upload-only.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = load_settings()
    logger = configure_logging(settings.logs_dir, verbose=args.verbose)

    try:
        needs_youtube = args.auth_only or args.upload_only or args.schedule or (not args.dry_run and not args.generate_only)
        settings.validate(require_youtube=needs_youtube)
        director = Director(settings, logger)
        if needs_youtube:
            director.pipeline.validate_youtube_authentication()

        if args.auth_only:
            print(f"YouTube authentication ready. Token path: {settings.youtube_token_file}")
            return 0

        if args.schedule:
            scheduler = DailyScheduler(settings, logger)
            scheduler.run_forever(director.scheduled_job)
            return 0

        if args.upload_only:
            if not args.video_path or not args.metadata_path:
                raise ValueError("--upload-only requires --video-path and --metadata-path")
            url = director.execute_upload_only(args.video_path, args.metadata_path, args.thumbnail_path)
            print(f"Uploaded video URL: {url}")
            return 0

        topic = " ".join(args.topic).strip() or None
        if args.generate_only:
            result = director.execute_generate_only(
                topic,
                use_existing_assets=args.use_existing_assets,
            )
        else:
            result = director.execute_run(
                topic,
                dry_run=args.dry_run,
                generate_only=args.generate_only,
                use_existing_assets=args.use_existing_assets,
            )
        print(f"Metadata saved at: {result.metadata_path}")
        if result.video_path:
            print(f"Final video saved at: {result.video_path}")
        if result.youtube_url:
            print(f"Uploaded video URL: {result.youtube_url}")
        return 0
    except Exception as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
