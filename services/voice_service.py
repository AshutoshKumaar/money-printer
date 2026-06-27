from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import edge_tts

from config import Settings
from core.models import Script
from core.retry import retry_call
from storage import RunPaths


class VoiceService:
    """Creates Hindi voiceover clips with Edge TTS."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def generate_voiceovers(self, script: Script, paths: RunPaths, use_existing: bool = False) -> list[Path]:
        audio_paths: list[Path] = []
        for index, segment in enumerate(script.segments, start=1):
            audio_path = paths.audio_dir / f"scene_{index:02d}.mp3"
            if not (segment.text or "").strip():
                self.logger.info("Empty segment text for scene %s; generating silent audio clip using FFmpeg", index)
                import subprocess
                command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=24000:cl=mono",
                    "-t",
                    "1.0",
                    "-acodec",
                    "libmp3lame",
                    str(audio_path),
                ]
                subprocess.run(command, check=True, capture_output=True)
                audio_paths.append(audio_path)
                from core.telemetry import telemetry_tracker
                telemetry_tracker.record(
                    stage="voice",
                    provider="silent_fallback",
                    model="silent",
                    endpoint="ffmpeg_silent",
                    attempt_number=1,
                    retry_count=0,
                    status_code=200,
                    latency=0.01,
                    response_size_bytes=audio_path.stat().st_size if audio_path.exists() else 0,
                    scene_index=index,
                )
                continue

            if use_existing:
                cached = self._cached_audio(index, audio_path)
                if cached:
                    from core.telemetry import telemetry_tracker
                    telemetry_tracker.record(
                        stage="voice",
                        provider="cache",
                        model="cache",
                        endpoint="cache",
                        cache_hit=True,
                        scene_index=index,
                    )
                    audio_paths.append(cached)
                    continue
            self.logger.info("Generating voiceover %s/%s", index, len(script.segments))
            
            import time
            from core.telemetry import telemetry_tracker
            attempt_tracker = {"count": 0}

            def run_voice_gen():
                attempt_tracker["count"] += 1
                t_start = time.time()
                try:
                    res = self._generate_one(segment.text, audio_path)
                    latency = time.time() - t_start
                    telemetry_tracker.record(
                        stage="voice",
                        provider="Microsoft",
                        model=self.settings.voice_name,
                        endpoint="EdgeTTS.synthesize",
                        attempt_number=attempt_tracker["count"],
                        retry_count=attempt_tracker["count"] - 1,
                        status_code=200,
                        latency=latency,
                        response_size_bytes=res.stat().st_size if res.exists() else 0,
                        scene_index=index,
                    )
                    return res
                except Exception as e:
                    latency = time.time() - t_start
                    telemetry_tracker.record(
                        stage="voice",
                        provider="Microsoft",
                        model=self.settings.voice_name,
                        endpoint="EdgeTTS.synthesize",
                        attempt_number=attempt_tracker["count"],
                        retry_count=attempt_tracker["count"] - 1,
                        status_code=500,
                        latency=latency,
                        scene_index=index,
                    )
                    raise e

            generated = retry_call(
                run_voice_gen,
                attempts=self.settings.retry_attempts,
                backoff_seconds=self.settings.retry_backoff_seconds,
                logger=self.logger,
                label=f"voice generation scene {index}",
            )
            audio_paths.append(generated)
        if self.settings.auto_fit_voice_duration:
            self._fit_total_duration(audio_paths)
        return audio_paths

    def _generate_one(self, text: str, output_path: Path) -> Path:
        async def synthesize() -> None:
            communicate = edge_tts.Communicate(
                text,
                self.settings.voice_name,
                rate=self.settings.voice_rate,
            )
            await communicate.save(str(output_path))

        asyncio.run(synthesize())
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Voice file was not created: {output_path}")
        return output_path

    def _fit_total_duration(self, audio_paths: list[Path]) -> None:
        durations = [self._duration(path) for path in audio_paths]
        total = sum(durations)
        target = float(self.settings.shorts_target_seconds)
        if total <= target or not audio_paths:
            self.logger.info("Voiceover duration %.2fs fits the %.2fs target", total, target)
            return

        speed = min(1.35, total / target)
        self.logger.warning(
            "Voiceover duration %.2fs exceeds target; applying %.3fx audio speed",
            total,
            speed,
        )
        for audio_path in audio_paths:
            temp_path = audio_path.with_name(f"{audio_path.stem}.fitted.mp3")
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-filter:a",
                f"atempo={speed:.5f}",
                "-vn",
                str(temp_path),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            temp_path.replace(audio_path)

        fitted_total = sum(self._duration(path) for path in audio_paths)
        self.logger.info("Voiceover duration after fitting: %.2fs", fitted_total)

    @staticmethod
    def _duration(audio_path: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())

    def _cached_audio(self, index: int, preferred_path: Path) -> Path | None:
        candidates = [
            preferred_path,
            self.settings.audio_dir / f"scene_{index:02d}.mp3",
            self.settings.audio_dir / f"scene_{index}.mp3",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        return None
