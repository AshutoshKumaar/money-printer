from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

from config import Settings
from research.models import (
    ResearchPackage,
    HistoricalContext,
    ScientificContext,
    ImportantEntities,
    VerifiedFact,
    CommonMisconception,
    UnansweredQuestion,
    VisualOpportunity,
    StoryOpportunities,
    SEOResearch,
    ResearchConfidence,
)
from research.prompts import RESEARCH_PROMPT_TEMPLATE


class BaseResearchProvider:
    """Base interface for structured research providers to enable future extensions."""

    def research(self, topic: str) -> ResearchPackage:
        raise NotImplementedError("Providers must implement the research method.")


class GeminiResearchProvider(BaseResearchProvider):
    """Factual research provider powered by Google Gemini API, with strict JSON validation and retry-loop feedback."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def research(self, topic: str) -> ResearchPackage:
        base_prompt = RESEARCH_PROMPT_TEMPLATE.format(topic=topic)
        current_prompt = base_prompt
        attempts = self.settings.retry_attempts
        last_exception = None

        from core.telemetry import telemetry_tracker

        for attempt in range(1, attempts + 1):
            t_start = time.time()
            try:
                self.logger.info("Calling Gemini research attempt %d/%d...", attempt, attempts)
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
                    stage="research",
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

                # Validate schema constraints
                self.validate_schema(data)

                # Construct nested models
                return self.parse_package(topic, data)

            except Exception as exc:
                latency = time.time() - t_start
                telemetry_tracker.record(
                    stage="research",
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
                self.logger.warning("Research attempt %d failed: %s", attempt, exc)
                
                if attempt < attempts:
                    # Enrich next attempt with correction feedback
                    current_prompt = (
                        f"{base_prompt}\n\n"
                        f"WARNING: Your previous response failed parsing or validation with error: {exc}\n"
                        f"Please fix this error and output ONLY a valid JSON object matching the requested schema."
                    )
                    time.sleep(self.settings.retry_backoff_seconds * attempt)

        raise ValueError(f"Factual research failed parsing or validation: {last_exception}") from last_exception

    def validate_schema(self, data: dict[str, Any]) -> None:
        """Validate structure and constraint values of the response payload."""
        required_keys = {
            "topic_summary",
            "historical_context",
            "scientific_context",
            "important_entities",
            "verified_facts",
            "common_misconceptions",
            "unanswered_questions",
            "visual_opportunities",
            "story_opportunities",
            "seo_research",
            "research_confidence",
        }
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"Missing required schema sections: {missing}")

        # Validate verified_facts
        facts = data.get("verified_facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError("verified_facts must be a non-empty list of fact records")

        for idx, f in enumerate(facts):
            if not isinstance(f, dict):
                raise ValueError(f"Fact record at index {idx} must be a JSON object")
            required_fact_keys = {"fact", "confidence_score", "verification_status", "reasoning", "evidence_level", "source_type"}
            missing_fact = required_fact_keys - f.keys()
            if missing_fact:
                raise ValueError(f"Fact record at index {idx} missing keys: {missing_fact}")

            # Status constraints
            status = f.get("verification_status")
            if status not in {"verified", "partially_verified", "unverified"}:
                raise ValueError(f"Fact index {idx} has invalid verification_status: '{status}'")

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

        # Validate visual_opportunities
        vis_opps = data.get("visual_opportunities", [])
        if not isinstance(vis_opps, list):
            raise ValueError("visual_opportunities must be a list")
        for idx, vo in enumerate(vis_opps):
            if not isinstance(vo, dict):
                raise ValueError(f"Visual opportunity at index {idx} must be a JSON object")
            opp_type = vo.get("opportunity_type")
            allowed_types = {"photograph", "map", "diagram", "historical artwork", "reconstruction", "animation"}
            if opp_type not in allowed_types:
                raise ValueError(f"Visual opportunity index {idx} has invalid opportunity_type: '{opp_type}'")

        # Validate unanswered_questions
        questions = data.get("unanswered_questions", [])
        if not isinstance(questions, list):
            raise ValueError("unanswered_questions must be a list")
        for idx, q in enumerate(questions):
            if not isinstance(q, dict):
                raise ValueError(f"Unanswered question at index {idx} must be a JSON object")
            q_type = q.get("uncertainty_type")
            if q_type not in {"scientific", "historical", "general"}:
                raise ValueError(f"Unanswered question index {idx} has invalid uncertainty_type: '{q_type}'")

    def parse_package(self, topic: str, data: dict[str, Any]) -> ResearchPackage:
        """Parse structured dict into strict dataclasses."""
        hist = data.get("historical_context", {})
        historical_context = HistoricalContext(
            timeline=[str(t).strip() for t in hist.get("timeline", []) if str(t).strip()],
            overview=str(hist.get("overview", "")).strip(),
        )

        sci = data.get("scientific_context", {})
        scientific_context = ScientificContext(
            explanation=str(sci.get("explanation", "")).strip(),
            concepts=[str(c).strip() for c in sci.get("concepts", []) if str(c).strip()],
        )

        ent = data.get("important_entities", {})
        important_entities = ImportantEntities(
            people=[str(p).strip() for p in ent.get("people", []) if str(p).strip()],
            places=[str(p).strip() for p in ent.get("places", []) if str(p).strip()],
            organizations=[str(o).strip() for o in ent.get("organizations", []) if str(o).strip()],
            technologies=[str(t).strip() for t in ent.get("technologies", []) if str(t).strip()],
            events=[str(e).strip() for e in ent.get("events", []) if str(e).strip()],
            objects=[str(o).strip() for o in ent.get("objects", []) if str(o).strip()],
        )

        verified_facts = []
        for f in data.get("verified_facts", []):
            verified_facts.append(
                VerifiedFact(
                    fact=str(f.get("fact", "")).strip(),
                    confidence_score=float(f.get("confidence_score", 0.0) or 0.0),
                    verification_status=str(f.get("verification_status", "unverified")),
                    reasoning=str(f.get("reasoning", "")).strip(),
                    evidence_level=str(f.get("evidence_level", "weak")),
                    source_type=str(f.get("source_type", "unknown")),
                )
            )

        common_misconceptions = []
        for cm in data.get("common_misconceptions", []):
            common_misconceptions.append(
                CommonMisconception(
                    myth=str(cm.get("myth", "")).strip(),
                    verified_fact=str(cm.get("verified_fact", "")).strip(),
                )
            )

        unanswered_questions = []
        for uq in data.get("unanswered_questions", []):
            unanswered_questions.append(
                UnansweredQuestion(
                    question=str(uq.get("question", "")).strip(),
                    uncertainty_type=str(uq.get("uncertainty_type", "general")),
                )
            )

        visual_opportunities = []
        for vo in data.get("visual_opportunities", []):
            visual_opportunities.append(
                VisualOpportunity(
                    opportunity_type=str(vo.get("opportunity_type", "photograph")),
                    description=str(vo.get("description", "")).strip(),
                )
            )

        story = data.get("story_opportunities", {})
        story_opportunities = StoryOpportunities(
            strongest_hook=str(story.get("strongest_hook", "")).strip(),
            biggest_surprise=str(story.get("biggest_surprise", "")).strip(),
            emotional_moments=[str(m).strip() for m in story.get("emotional_moments", []) if str(m).strip()],
            best_climax=str(story.get("best_climax", "")).strip(),
            strongest_ending=str(story.get("strongest_ending", "")).strip(),
        )

        seo = data.get("seo_research", {})
        seo_research = SEOResearch(
            primary_keywords=[str(k).strip() for k in seo.get("primary_keywords", []) if str(k).strip()],
            secondary_keywords=[str(k).strip() for k in seo.get("secondary_keywords", []) if str(k).strip()],
            related_concepts=[str(c).strip() for c in seo.get("related_concepts", []) if str(c).strip()],
        )

        conf = data.get("research_confidence", {})
        research_confidence = ResearchConfidence(
            overall_score=float(conf.get("overall_score", 0.0) or 0.0),
            potential_weak_areas=[str(w).strip() for w in conf.get("potential_weak_areas", []) if str(w).strip()],
            recommended_verification_priority=str(conf.get("recommended_verification_priority", "moderate")).strip(),
        )

        return ResearchPackage(
            topic=topic,
            topic_summary=str(data.get("topic_summary", "")).strip(),
            historical_context=historical_context,
            scientific_context=scientific_context,
            important_entities=important_entities,
            verified_facts=verified_facts,
            common_misconceptions=common_misconceptions,
            unanswered_questions=unanswered_questions,
            visual_opportunities=visual_opportunities,
            story_opportunities=story_opportunities,
            seo_research=seo_research,
            research_confidence=research_confidence,
        )

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text


class ResearchEngine:
    """Orchestrates structured knowledge collection for a given topic, producing an immutable ResearchPackage."""

    def __init__(self, settings: Settings, logger: logging.Logger, provider: BaseResearchProvider | None = None) -> None:
        self.settings = settings
        self.logger = logger
        self.provider = provider or GeminiResearchProvider(settings, logger)

    def research_topic(self, topic: str) -> ResearchPackage:
        self.logger.info("Starting factual research gathering (V2 Engine) for topic: %s", topic)
        try:
            return self.provider.research(topic)
        except Exception as exc:
            self.logger.error("Factual research V2 failed completely: %s. Constructing fallback package.", exc)
            return ResearchPackage(
                topic=topic,
                topic_summary=f"Factual research gathering failed. Error: {exc}",
                historical_context=HistoricalContext(timeline=[], overview=""),
                scientific_context=ScientificContext(explanation="", concepts=[]),
                important_entities=ImportantEntities(people=[], places=[], organizations=[], technologies=[], events=[], objects=[]),
                verified_facts=[],
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
                    overall_score=0.0,
                    potential_weak_areas=[f"Research failed: {exc}"],
                    recommended_verification_priority="high",
                ),
            )
