from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

from config import Settings
from verification.models import VerifiedResearchPackage
from story.models import NarrativePackage, NarrationSegment, NarrativeQuality
from story.prompts import STORY_PROMPT_TEMPLATE


class BaseStoryProvider:
    """Base interface for narrative story script providers to support future extensions."""

    def generate_story(self, package: VerifiedResearchPackage, language: str) -> NarrativePackage:
        raise NotImplementedError("Providers must implement generate_story.")


class GeminiStoryProvider(BaseStoryProvider):
    """Creative script generation provider powered by Google Gemini API, with JSON validation and retry loop."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def generate_story(self, package: VerifiedResearchPackage, language: str) -> NarrativePackage:
        # Format verified facts for prompt
        facts_list = []
        for idx, f in enumerate(package.verified_facts, start=1):
            claim = f.suggested_clarification or f.fact
            facts_list.append(f"Fact ID: fact_{idx} | claim: {claim} | status: {f.status} | evidence: {f.evidence_level}")

        facts_block = "\n".join(facts_list) if facts_list else "No verified facts found."
        
        prompt = STORY_PROMPT_TEMPLATE.format(
            verified_facts=facts_block,
            language=language,
        )

        from core.telemetry import telemetry_tracker

        attempts = self.settings.retry_attempts
        current_prompt = prompt
        last_exception = None

        for attempt in range(1, attempts + 1):
            t_start = time.time()
            try:
                self.logger.info("Calling Gemini story writing attempt %d/%d...", attempt, attempts)
                res = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=current_prompt,
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
                    attempt_number=attempt,
                    retry_count=attempt - 1,
                    status_code=200,
                    latency=latency,
                    response_size_bytes=len(res.text or "") if res.text else 0,
                )

                raw_text = res.text or ""
                cleaned = self._clean_json_text(raw_text)
                data = json.loads(cleaned)

                # Validate JSON schema and value constraints
                self.validate_schema(data, package)

                # Parse and return NarrativePackage
                if attempt > 1:
                    telemetry_tracker.retries["Gemini"]["recovered"] = True
                return self.parse_narrative_package(data)

            except Exception as exc:
                latency = time.time() - t_start
                telemetry_tracker.record(
                    stage="story",
                    provider="Google",
                    model="gemini-2.5-flash",
                    endpoint="models.generate_content",
                    attempt_number=attempt,
                    retry_count=attempt - 1,
                    status_code=getattr(exc, "status_code", 500) or 500,
                    latency=latency,
                    response_size_bytes=0,
                )
                last_exception = exc
                self.logger.warning("Story writing attempt %d failed: %s", attempt, exc)
                telemetry_tracker.record_retry("Gemini", str(exc), recovered=False, fallback=False)
                
                if attempt < attempts:
                    current_prompt = (
                        f"{prompt}\n\n"
                        f"WARNING: Your previous response failed parsing or validation with error: {exc}\n"
                        f"Please fix this error and output ONLY a valid JSON object matching the requested story schema."
                    )
                    time.sleep(self.settings.retry_backoff_seconds * attempt)

        raise ValueError(f"Story script generation failed validation: {last_exception}") from last_exception

    def validate_schema(self, data: dict[str, Any], package: VerifiedResearchPackage) -> None:
        """Validate structure and constraint values of the narrative script payload."""
        required_keys = {"language", "hook", "context", "escalation", "climax", "ending", "narration_segments", "quality"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"Missing required story schema sections: {missing}")

        segments = data.get("narration_segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("narration_segments must be a non-empty list of narration records")

        valid_emotions = {"curiosity", "suspense", "wonder", "surprise", "fear", "urgency", "neutral"}
        valid_beats = {"hook", "question", "setup", "evidence", "twist", "reveal", "reflection", "cta"}

        total_duration = 0.0
        for idx, s in enumerate(segments):
            if not isinstance(s, dict):
                raise ValueError(f"Narration segment at index {idx} must be a JSON object")
            required_seg_keys = {
                "index",
                "narration_text",
                "estimated_duration",
                "target_start",
                "target_end",
                "emotion",
                "purpose",
                "verified_fact_ids",
                "beat_type",
            }
            missing_seg = required_seg_keys - s.keys()
            if missing_seg:
                raise ValueError(f"Narration segment at index {idx} missing keys: {missing_seg}")

            total_duration += float(s.get("estimated_duration", 0.0) or 0.0)

            # Emotion enum check
            emotion = s.get("emotion")
            if emotion not in valid_emotions:
                raise ValueError(f"Segment index {idx} has invalid emotion: '{emotion}'")

            # Beat type enum check
            beat = s.get("beat_type")
            if beat not in valid_beats:
                raise ValueError(f"Segment index {idx} has invalid beat_type: '{beat}'")

            # Traceability check: fact IDs must exist in research package
            fact_ids = s.get("verified_fact_ids")
            if not isinstance(fact_ids, list):
                raise ValueError(f"Segment index {idx} verified_fact_ids must be a list")
            
            allowed_fact_ids = {f"fact_{i}" for i in range(1, len(package.verified_facts) + 1)}
            for fid in fact_ids:
                if fid not in allowed_fact_ids:
                    raise ValueError(f"Segment index {idx} references invalid fact ID: '{fid}'. Allowed: {allowed_fact_ids}")

        # Target range check: 35 to 55 seconds
        if not (35.0 <= total_duration <= 55.0):
            raise ValueError(f"Total estimated script duration ({total_duration:.1f}s) is outside target range of 35-55s")

        # Quality dict check
        quality = data.get("quality")
        if not isinstance(quality, dict):
            raise ValueError("quality section must be a JSON object")
        required_quality_keys = {"retention_score", "pacing_score", "curiosity_score", "clarity_score", "emotional_score", "estimated_retention_curve"}
        missing_quality = required_quality_keys - quality.keys()
        if missing_quality:
            raise ValueError(f"quality section missing keys: {missing_quality}")

    def parse_narrative_package(self, data: dict[str, Any]) -> NarrativePackage:
        """Parse structured dict into NarrativePackage dataclass."""
        segments = []
        for s in data.get("narration_segments", []):
            segments.append(
                NarrationSegment(
                    index=int(s.get("index", 1)),
                    narration_text=str(s.get("narration_text", "")).strip(),
                    estimated_duration=float(s.get("estimated_duration", 0.0) or 0.0),
                    target_start=float(s.get("target_start", 0.0) or 0.0),
                    target_end=float(s.get("target_end", 0.0) or 0.0),
                    emotion=str(s.get("emotion", "neutral")),
                    purpose=str(s.get("purpose", "")).strip(),
                    verified_fact_ids=[str(fid).strip() for fid in s.get("verified_fact_ids", []) if str(fid).strip()],
                    beat_type=str(s.get("beat_type", "evidence")),
                )
            )

        q = data.get("quality", {})
        quality = NarrativeQuality(
            retention_score=float(q.get("retention_score", 0.0) or 0.0),
            pacing_score=float(q.get("pacing_score", 0.0) or 0.0),
            curiosity_score=float(q.get("curiosity_score", 0.0) or 0.0),
            clarity_score=float(q.get("clarity_score", 0.0) or 0.0),
            emotional_score=float(q.get("emotional_score", 0.0) or 0.0),
            estimated_retention_curve=[float(c) for c in q.get("estimated_retention_curve", [])],
        )

        return NarrativePackage(
            language=str(data.get("language", "hi")).strip(),
            hook=str(data.get("hook", "")).strip(),
            context=str(data.get("context", "")).strip(),
            escalation=str(data.get("escalation", "")).strip(),
            climax=str(data.get("climax", "")).strip(),
            ending=str(data.get("ending", "")).strip(),
            narration_segments=segments,
            quality=quality,
        )

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text


class StoryEngine:
    """Orchestrates structured narrative scripting from verified research facts."""

    def __init__(self, settings: Settings, logger: logging.Logger, provider: BaseStoryProvider | None = None) -> None:
        self.settings = settings
        self.logger = logger
        self.provider = provider or GeminiStoryProvider(settings, logger)

    def write_story(self, report: Any) -> NarrativePackage:
        self.logger.info("Starting creative story script writing (V2 Engine)...")

        # Unwrap VerifiedResearchPackage
        if hasattr(report, "_package"):
            package = report._package
        else:
            package = self._dummy_package_from_report(report)

        language = "Devanagari Hindi"
        try:
            return self.provider.generate_story(package, language)
        except Exception as exc:
            self.logger.error("Story script generation V2 failed completely: %s. Constructing fallback package.", exc)
            
            # Simple fallback narration segment from verified facts
            fallback_segments = []
            current_time = 0.0
            for idx, fact_rec in enumerate(package.verified_facts, start=1):
                dur = 5.0
                fallback_segments.append(
                    NarrationSegment(
                        index=idx,
                        narration_text=f"क्या आप जानते हैं? {fact_rec.fact}",
                        estimated_duration=dur,
                        target_start=current_time,
                        target_end=current_time + dur,
                        emotion="curiosity",
                        purpose="Fallback fact representation",
                        verified_fact_ids=[f"fact_{idx}"],
                        beat_type="evidence",
                    )
                )
                current_time += dur
            
            # Pad to 35s if too short
            if current_time < 35.0:
                pad_dur = 35.0 - current_time
                fallback_segments.append(
                    NarrationSegment(
                        index=len(fallback_segments) + 1,
                        narration_text="इस बारे में अपने विचार कमेंट्स में बताएं और सब्सक्राइब करें।",
                        estimated_duration=pad_dur,
                        target_start=current_time,
                        target_end=35.0,
                        emotion="neutral",
                        purpose="Pad script duration to 35s",
                        verified_fact_ids=[],
                        beat_type="cta",
                    )
                )

            return NarrativePackage(
                language=language,
                hook="रहस्यमयी कहानी की शुरुआत।",
                context="विषय का परिचय।",
                escalation="रहस्य गहराता जा रहा है।",
                climax="सच्चाई का खुलासा।",
                ending="कमेंट करें और सब्सक्राइब करें।",
                narration_segments=fallback_segments,
                quality=NarrativeQuality(
                    retention_score=0.5,
                    pacing_score=0.5,
                    curiosity_score=0.5,
                    clarity_score=0.5,
                    emotional_score=0.5,
                    estimated_retention_curve=[1.0, 0.5],
                ),
            )

    def _dummy_package_from_report(self, report: Any) -> VerifiedResearchPackage:
        """Construct a compatibility VerifiedResearchPackage from legacy VerificationReport."""
        from research.models import (
            ResearchPackage,
            HistoricalContext,
            ScientificContext,
            ImportantEntities,
            StoryOpportunities,
            SEOResearch,
            ResearchConfidence,
        )

        dummy_research = ResearchPackage(
            topic="Legacy Topic",
            topic_summary="Legacy Summary",
            historical_context=HistoricalContext(timeline=[], overview=""),
            scientific_context=ScientificContext(explanation="", concepts=[]),
            important_entities=ImportantEntities(people=[], places=[], organizations=[], technologies=[], events=[], objects=[]),
            verified_facts=[],
            common_misconceptions=[],
            unanswered_questions=[],
            visual_opportunities=[],
            story_opportunities=StoryOpportunities("", "", [], "", ""),
            seo_research=SEOResearch([], [], []),
            research_confidence=ResearchConfidence(0.0, [], "moderate"),
        )

        verified_records = []
        for idx, r in enumerate(report.verified_facts, start=1):
            verified_records.append(
                VerifiedFactRecord(
                    fact=r.corrected_version or r.original_fact,
                    status=r.status,
                    original_fact=r.original_fact,
                    suggested_clarification=r.corrected_version or None,
                    confidence_score=r.confidence,
                    reasoning=r.explanation,
                    evidence_level="moderate",
                    source_type="unknown",
                    contradictions_detected=[],
                    is_duplicate=False,
                    importance_score=r.importance_score,
                    category=r.category,
                )
            )

        return VerifiedResearchPackage(
            research_package=dummy_research,
            verified_facts=verified_records,
            verification_warnings=report.warnings,
            overall_confidence_score=report.confidence_score,
            verification_quality_score=report.research_quality_score,
        )
