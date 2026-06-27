from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types

from config import Settings
from core.retry import retry_call
from research.models import ResearchContext
from verification.models import FactVerificationRecord, VerificationReport
from verification.prompts import VERIFICATION_PROMPT_TEMPLATE


class BaseVerificationProvider:
    """Base interface for all verification providers to support future extensions."""

    def verify_research(self, research: ResearchContext) -> VerificationReport:
        raise NotImplementedError("Providers must implement verify_research.")


class GeminiVerificationProvider(BaseVerificationProvider):
    """Fact verification provider powered by Google Gemini API."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def verify_research(self, research: ResearchContext) -> VerificationReport:
        prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            topic=research.topic,
            summary=research.summary,
            facts=json.dumps(research.facts, ensure_ascii=False),
            statistics=json.dumps(research.statistics, ensure_ascii=False),
            timeline=json.dumps(research.timeline, ensure_ascii=False),
            myths=json.dumps(research.myths, ensure_ascii=False),
            controversies=json.dumps(research.controversies, ensure_ascii=False),
        )

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
                    stage="verification",
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
                    stage="verification",
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
            label="Gemini research verification",
        )

        raw_text = response.text or ""
        cleaned = self._clean_json_text(raw_text)
        data = json.loads(cleaned)

        def make_record(item: dict) -> FactVerificationRecord:
            return FactVerificationRecord(
                original_fact=str(item.get("original_fact", "")).strip(),
                status=str(item.get("status", "unverified")).strip(),
                corrected_version=str(item.get("corrected_version", "")).strip(),
                confidence=float(item.get("confidence", 0.0) or 0.0),
                explanation=str(item.get("explanation", "")).strip(),
                importance_score=float(item.get("importance_score", 0.0) or 0.0),
                category=str(item.get("category", "")).strip(),
            )

        verified = [make_record(item) for item in data.get("verified_facts", []) if isinstance(item, dict)]
        rejected = [make_record(item) for item in data.get("rejected_facts", []) if isinstance(item, dict)]
        corrected = [make_record(item) for item in data.get("corrected_facts", []) if isinstance(item, dict)]

        return VerificationReport(
            verified_facts=verified,
            rejected_facts=rejected,
            corrected_facts=corrected,
            warnings=[str(w).strip() for w in data.get("warnings", []) if str(w).strip()],
            confidence_score=float(data.get("confidence_score", 0.0) or 0.0),
            research_quality_score=float(data.get("research_quality_score", 0.0) or 0.0),
        )

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text


class VerificationEngine:
    """Orchestrates validation and fact-checking of research contexts."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        provider: BaseVerificationProvider | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.provider = provider or GeminiVerificationProvider(settings, logger)

    def verify(self, research: ResearchContext) -> VerificationReport:
        self.logger.info("Starting fact verification for research on: %s", research.topic)
        if not research.facts and not research.statistics and not research.summary:
            self.logger.warning("Empty research context provided; returning empty verification report")
            return VerificationReport(
                verified_facts=[],
                rejected_facts=[],
                corrected_facts=[],
                warnings=["Received empty research data"],
                confidence_score=0.0,
                research_quality_score=0.0,
            )

        try:
            return self.provider.verify_research(research)
        except Exception as exc:
            self.logger.error("Verification failed completely: %s. Returning fallback report.", exc)
            fallback_records = [
                FactVerificationRecord(
                    original_fact=fact,
                    status="unverified",
                    corrected_version="",
                    confidence=0.0,
                    explanation=f"Verification failed due to error: {exc}",
                    importance_score=0.5,
                    category="unknown",
                )
                for fact in research.facts
            ]
            return VerificationReport(
                verified_facts=[],
                rejected_facts=fallback_records,
                corrected_facts=[],
                warnings=[f"Verification failure: {exc}"],
                confidence_score=0.0,
                research_quality_score=0.0,
            )
