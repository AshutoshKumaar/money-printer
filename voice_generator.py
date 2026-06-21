from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts

from config import load_settings


async def _synthesize_text(text: str, output_file: Path, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_file))


def generate_voice(text: str, filename: str) -> str | None:
    """Backward-compatible voice generation helper."""
    settings = load_settings()
    output_path = settings.audio_dir / filename
    try:
        asyncio.run(_synthesize_text(text, output_path, settings.voice_name))
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)
        raise RuntimeError("Voice file was empty")
    except Exception as exc:
        print(f"Error generating voice: {exc}")
        return None


if __name__ == "__main__":
    path = generate_voice("नमस्ते दोस्तों, आज हम AI के बारे में बात करेंगे।", "test_voice.mp3")
    print(f"Generated test voice at: {path}")
