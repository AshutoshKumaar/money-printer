from __future__ import annotations

import json
import sys

from config import load_settings
from core.logging import configure_logging
from services.gemini_service import GeminiService


def generate_script(topic: str) -> dict:
    """Backward-compatible wrapper around the production Gemini service."""
    settings = load_settings()
    settings.validate(require_youtube=False)
    logger = configure_logging(settings.logs_dir)
    return GeminiService(settings, logger).generate_script(topic).to_dict()


if __name__ == "__main__":
    test_topic = " ".join(sys.argv[1:]) or "Artificial Intelligence in daily life"
    print(json.dumps(generate_script(test_topic), indent=2, ensure_ascii=False))
