from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types

from config import Settings
from core.retry import retry_call
from verification.models import VerificationReport
from story.models import EmotionCurve, NarrativeScript, StorySegment
from story.prompts import STORY_PROMPT_TEMPLATE


class BaseStoryProvider:
    """Base interface for narrative story script providers to support future extensions."""

    def generate_story(self, report: VerificationReport) -> NarrativeScript:
        raise NotImplementedError("Providers must implement generate_story.")


class GeminiStoryProvider(BaseStoryProvider):
    """Creative script generation provider powered by Google Gemini API."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def generate_story(self, report: VerificationReport) -> NarrativeScript:
        facts_list = []
        for index, record in enumerate(report.verified_facts, start=1):
            claim = record.corrected_version or record.original_fact
            facts_list.append(f"{index}. [Category: {record.category}] {claim} (Importance: {record.importance_score})")

        facts_block = "\n".join(facts_list) if facts_list else "No verified facts found."
        prompt = STORY_PROMPT_TEMPLATE.format(verified_facts=facts_block)

        import time
        from core.telemetry import telemetry_tracker

        attempt_tracker = {"count": 0}

        def run_call():
            attempt_tracker["count"] += 1
            t_start = time.time()
            try:
                res = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                latency = time.time() - t_start
                input_tokens = 0
                output_tokens = 0
                if res.usage_metadata:
                    input_tokens = res.usage_metadata.prompt_token_count or 0
                    output_tokens = res.usage_metadata.candidates_token_count or 0

                telemetry_tracker.record(
                    stage="story",
                    provider="Google",
                    model="gemini-2.5-flash",
                    endpoint="models.generate_content",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attempt_number=attempt_tracker["count"],
                    retry_count=attempt_tracker["count"] - 1,
                    status_code=200,
                    latency=latency,
                    response_size_bytes=len(res.text or "") if res.text else 0,
                )
                return res
            except Exception as e:
                latency = time.time() - t_start
                telemetry_tracker.record(
                    stage="story",
                    provider="Google",
                    model="gemini-2.5-flash",
                    endpoint="models.generate_content",
                    attempt_number=attempt_tracker["count"],
                    retry_count=attempt_tracker["count"] - 1,
                    status_code=getattr(e, "status_code", 500) or 500,
                    latency=latency,
                    response_size_bytes=0,
                )
                raise e

        response = retry_call(
            run_call,
            attempts=self.settings.retry_attempts,
            backoff_seconds=self.settings.retry_backoff_seconds,
            logger=self.logger,
            label="Gemini script writing",
        )

        raw_text = response.text or ""
        cleaned = self._clean_json_text(raw_text)
        data = json.loads(cleaned)

        def make_emotion(curve_data: dict) -> EmotionCurve:
            return EmotionCurve(
                curiosity=float(curve_data.get("curiosity", 0.0) or 0.0),
                fear=float(curve_data.get("fear", 0.0) or 0.0),
                surprise=float(curve_data.get("surprise", 0.0) or 0.0),
                wonder=float(curve_data.get("wonder", 0.0) or 0.0),
                urgency=float(curve_data.get("urgency", 0.0) or 0.0),
            )

        def make_segment(item: dict) -> StorySegment:
            return StorySegment(
                index=int(item.get("index", 1)),
                spoken_hindi=str(item.get("spoken_hindi", "")).strip(),
                caption_keywords=str(item.get("caption_keywords", item.get("caption_hinglish", ""))).strip(),
                search_query=str(item.get("search_query", "")).strip(),
                visual_concept=str(item.get("visual_concept", "")).strip(),
                emotion_curve=make_emotion(item.get("emotion_curve", {})),
            )

        segments = [make_segment(item) for item in data.get("segments", []) if isinstance(item, dict)]

        cta_text = str(data.get("cta", "")).strip()
        if cta_text and segments:
            # Check if CTA is already present in the last segment's spoken_hindi
            if cta_text.lower() not in segments[-1].spoken_hindi.lower():
                # Append CTA as a new final segment
                cta_index = len(segments) + 1
                default_emotions = make_emotion({})
                cta_segment = StorySegment(
                    index=cta_index,
                    spoken_hindi=cta_text,
                    caption_keywords="लाइक सब्सक्राइब कमेंट",
                    search_query="social media call to action like subscribe button",
                    visual_concept="Clean call to action graphic",
                    emotion_curve=default_emotions,
                )
                segments.append(cta_segment)

        return NarrativeScript(
            hook=str(data.get("hook", "")).strip(),
            context=str(data.get("context", "")).strip(),
            segments=segments,
            ending=str(data.get("ending", "")).strip(),
            cta=cta_text,
            estimated_duration=float(data.get("estimated_duration", 0.0) or 0.0),
            estimated_words=int(data.get("estimated_words", 0) or 0),
            emotion_curve=make_emotion(data.get("emotion_curve", {})),
            retention_score=float(data.get("retention_score", 0.0) or 0.0),
        )

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text


class StoryEngine:
    """Orchestrates creative script generation from fact-checked reports."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        provider: BaseStoryProvider | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.provider = provider or GeminiStoryProvider(settings, logger)

    def write_story(self, report: VerificationReport) -> NarrativeScript:
        self.logger.info("Starting creative story script writing from verification report...")
        if not report.verified_facts:
            self.logger.warning("No verified facts in report; returning a default fallback script")
            return self._fallback_script()

        try:
            return self.provider.generate_story(report)
        except Exception as exc:
            self.logger.error("Story script writing failed: %s. Returning fallback script.", exc)
            return self._fallback_script()

    def _fallback_script(self) -> NarrativeScript:
        default_emotions = EmotionCurve(0.5, 0.1, 0.5, 0.5, 0.5)
        fallback_segment = StorySegment(
            index=1,
            spoken_hindi="अद्भुत तथ्यों के लिए वीडियो को अंत तक देखें।",
            caption_keywords="रहस्य अद्भुत तथ्य वीडियो",
            search_query="cinematic horizontal background",
            visual_concept="mysterious motion background",
            emotion_curve=default_emotions,
        )
        cta_segment = StorySegment(
            index=2,
            spoken_hindi="अपनी राय कमेंट करें और फॉलो करें!",
            caption_keywords="लाइक सब्सक्राइब कमेंट",
            search_query="social media call to action like subscribe button",
            visual_concept="Clean call to action graphic",
            emotion_curve=default_emotions,
        )
        return NarrativeScript(
            hook="रुकिए, क्या आप जानते हैं?",
            context="दुनिया में कई रहस्य छिपे हुए हैं।",
            segments=[fallback_segment, cta_segment],
            ending="आज की जानकारी कैसी लगी?",
            cta="अपनी राय कमेंट करें और फॉलो करें!",
            estimated_duration=12.0,
            estimated_words=20,
            emotion_curve=default_emotions,
            retention_score=0.5,
        )
