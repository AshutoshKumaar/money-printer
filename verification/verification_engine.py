from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

from config import Settings
from core.retry import retry_call
from research.models import ResearchPackage, ResearchPackageAdapter
from verification.models import VerifiedResearchPackage, VerifiedFactRecord, FactVerificationRecord
from verification.prompts import VERIFICATION_PROMPT_TEMPLATE


class BaseVerificationProvider:
    """Base interface for all verification providers to support future extensions."""

    def verify_research(self, package: ResearchPackage) -> VerifiedResearchPackage:
        raise NotImplementedError("Providers must implement verify_research.")


class GeminiVerificationProvider(BaseVerificationProvider):
    """Fact verification provider powered by Google Gemini API, with JSON validation and retry handling."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def verify_research(self, package: ResearchPackage) -> VerifiedResearchPackage:
        # Construct facts block
        facts_list = [f.fact for f in package.verified_facts]
        myths_list = [f"Myth: {m.myth} | Truth: {m.verified_fact}" for m in package.common_misconceptions]
        
        prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            topic=package.topic,
            summary=package.topic_summary,
            facts=json.dumps(facts_list, ensure_ascii=False),
            myths=json.dumps(myths_list, ensure_ascii=False),
            timeline=json.dumps(package.historical_context.timeline, ensure_ascii=False),
        )

        from core.telemetry import telemetry_tracker

        attempts = self.settings.retry_attempts
        current_prompt = prompt
        last_exception = None

        for attempt in range(1, attempts + 1):
            t_start = time.time()
            try:
                self.logger.info("Calling Gemini verification attempt %d/%d...", attempt, attempts)
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
                    stage="verification",
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
                self.validate_schema(data)

                # Parse and return VerifiedResearchPackage
                return self.parse_verified_package(package, data)

            except Exception as exc:
                latency = time.time() - t_start
                telemetry_tracker.record(
                    stage="verification",
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
                self.logger.warning("Verification attempt %d failed: %s", attempt, exc)
                
                if attempt < attempts:
                    current_prompt = (
                        f"{prompt}\n\n"
                        f"WARNING: Your previous response failed parsing or validation with error: {exc}\n"
                        f"Please fix this error and output ONLY a valid JSON object matching the verification schema."
                    )
                    time.sleep(self.settings.retry_backoff_seconds * attempt)

        raise ValueError(f"Fact verification failed parsing or validation: {last_exception}") from last_exception

    def validate_schema(self, data: dict[str, Any]) -> None:
        """Validate structure and constraint values of the verification payload."""
        required_keys = {"verified_facts", "verification_warnings", "verification_quality_score"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"Missing required verification schema sections: {missing}")

        facts = data.get("verified_facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError("verified_facts must be a non-empty list of verification records")

        for idx, f in enumerate(facts):
            if not isinstance(f, dict):
                raise ValueError(f"Verification record at index {idx} must be a JSON object")
            required_fact_keys = {
                "fact",
                "status",
                "original_fact",
                "suggested_clarification",
                "confidence_score",
                "reasoning",
                "evidence_level",
                "source_type",
                "contradictions_detected",
                "is_duplicate",
                "importance_score",
                "category",
            }
            missing_fact = required_fact_keys - f.keys()
            if missing_fact:
                raise ValueError(f"Fact record at index {idx} missing keys: {missing_fact}")

            # Status constraints
            status = f.get("status")
            allowed_statuses = {"verified", "partially_verified", "disputed", "insufficient_evidence", "unverified"}
            if status not in allowed_statuses:
                raise ValueError(f"Fact index {idx} has invalid status: '{status}'")

            # Evidence level constraints
            level = f.get("evidence_level")
            if level not in {"strong", "moderate", "weak"}:
                raise ValueError(f"Fact index {idx} has invalid evidence_level: '{level}'")

            # Source type constraints
            stype = f.get("source_type")
            allowed_sources = {
                "scientific consensus",
                "historical consensus",
                "government publication",
                "academic journal",
                "encyclopedia",
                "first-hand account",
                "requires verification",
                "unknown",
            }
            if stype not in allowed_sources:
                raise ValueError(f"Fact index {idx} has invalid source_type: '{stype}'")

            # Contradictions list constraint
            contra = f.get("contradictions_detected")
            if not isinstance(contra, list):
                raise ValueError(f"Fact index {idx} has invalid contradictions_detected type, must be list")

    def parse_verified_package(self, package: ResearchPackage, data: dict[str, Any]) -> VerifiedResearchPackage:
        """Parse structured verification dict into VerifiedResearchPackage dataclass."""
        verified_facts = []
        for f in data.get("verified_facts", []):
            verified_facts.append(
                VerifiedFactRecord(
                    fact=str(f.get("fact", "")).strip(),
                    status=str(f.get("status", "unverified")),
                    original_fact=str(f.get("original_fact", "")).strip(),
                    suggested_clarification=str(f.get("suggested_clarification", "")).strip() or None if f.get("suggested_clarification") else None,
                    confidence_score=float(f.get("confidence_score", 0.0) or 0.0),
                    reasoning=str(f.get("reasoning", "")).strip(),
                    evidence_level=str(f.get("evidence_level", "weak")),
                    source_type=str(f.get("source_type", "unknown")),
                    contradictions_detected=[str(c).strip() for c in f.get("contradictions_detected", []) if str(c).strip()],
                    is_duplicate=bool(f.get("is_duplicate", False)),
                    importance_score=float(f.get("importance_score", 0.5) or 0.5),
                    category=str(f.get("category", "general")).strip(),
                )
            )

        warnings = [str(w).strip() for w in data.get("verification_warnings", []) if str(w).strip()]
        quality_score = float(data.get("verification_quality_score", 0.0) or 0.0)

        # Calculate overall confidence score using weighted importance
        total_importance = sum(f.importance_score for f in verified_facts)
        if total_importance > 0:
            weighted_confidence = sum(f.confidence_score * f.importance_score for f in verified_facts) / total_importance
        else:
            weighted_confidence = 0.0

        return VerifiedResearchPackage(
            research_package=package,
            verified_facts=verified_facts,
            verification_warnings=warnings,
            overall_confidence_score=weighted_confidence,
            verification_quality_score=quality_score,
        )

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text


class VerificationEngine:
    """Orchestrates validation and fact-checking of research packages."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        provider: BaseVerificationProvider | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.provider = provider or GeminiVerificationProvider(settings, logger)

    def verify(self, research: ResearchContext | ResearchPackageAdapter) -> VerifiedResearchPackage:
        self.logger.info("Starting fact verification (V2 Engine) for research on: %s", research.topic)

        # Retrieve or construct the original ResearchPackage
        if isinstance(research, ResearchPackageAdapter):
            package = research._package
        else:
            package = self._dummy_package_from_context(research)

        if not package.verified_facts and not package.topic_summary:
            self.logger.warning("Empty research package provided; returning empty verified package")
            return VerifiedResearchPackage(
                research_package=package,
                verified_facts=[],
                verification_warnings=["Received empty research data"],
                overall_confidence_score=0.0,
                verification_quality_score=0.0,
            )

        try:
            return self.provider.verify_research(package)
        except Exception as exc:
            self.logger.error("Fact verification V2 failed completely: %s. Returning fallback package.", exc)
            fallback_records = []
            for fact_rec in package.verified_facts:
                fallback_records.append(
                    VerifiedFactRecord(
                        fact=fact_rec.fact,
                        status="unverified",
                        original_fact=fact_rec.fact,
                        suggested_clarification=None,
                        confidence_score=0.0,
                        reasoning=f"Verification failed due to error: {exc}",
                        evidence_level="weak",
                        source_type="unknown",
                        contradictions_detected=[],
                        is_duplicate=False,
                        importance_score=0.5,
                        category="unknown",
                    )
                )

            return VerifiedResearchPackage(
                research_package=package,
                verified_facts=fallback_records,
                verification_warnings=[f"Verification failure: {exc}"],
                overall_confidence_score=0.0,
                verification_quality_score=0.0,
            )

    def _dummy_package_from_context(self, context: ResearchContext) -> ResearchPackage:
        """Construct a compatibility ResearchPackage dummy from a legacy ResearchContext."""
        from research.models import (
            HistoricalContext,
            ScientificContext,
            ImportantEntities,
            VerifiedFact,
            StoryOpportunities,
            SEOResearch,
            ResearchConfidence,
        )

        verified_facts = [
            VerifiedFact(
                fact=fact,
                confidence_score=context.confidence_score,
                verification_status="verified",
                reasoning="Legacy import",
                evidence_level="moderate",
                source_type="unknown",
            )
            for fact in context.facts
        ]

        return ResearchPackage(
            topic=context.topic,
            topic_summary=context.summary,
            historical_context=HistoricalContext(timeline=context.timeline, overview=""),
            scientific_context=ScientificContext(explanation="", concepts=[]),
            important_entities=ImportantEntities(
                people=context.people,
                places=context.locations,
                organizations=[],
                technologies=[],
                events=[],
                objects=[],
            ),
            verified_facts=verified_facts,
            common_misconceptions=[],
            unanswered_questions=[],
            visual_opportunities=[],
            story_opportunities=StoryOpportunities(
                strongest_hook="",
                biggest_surprise="",
                emotional_moments=[],
                best_climax="",
                strongest_ending="",
            ),
            seo_research=SEOResearch(primary_keywords=[], secondary_keywords=[], related_concepts=[]),
            research_confidence=ResearchConfidence(
                overall_score=context.confidence_score,
                potential_weak_areas=context.warnings,
                recommended_verification_priority="moderate",
            ),
        )
