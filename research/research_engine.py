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
from research.prompts import RESEARCH_PROMPT_TEMPLATE


class BaseResearchProvider:
    """Base interface for all factual research providers to enable future extensions."""

    def research(self, topic: str) -> ResearchContext:
        raise NotImplementedError("Providers must implement the research method.")


class GeminiResearchProvider(BaseResearchProvider):
    """Factual research provider powered by Google Gemini API."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def research(self, topic: str) -> ResearchContext:
        prompt = RESEARCH_PROMPT_TEMPLATE.format(topic=topic)
        
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
                    stage="research",
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
                    stage="research",
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
            label="Gemini factual research gathering",
        )
        
        raw_text = response.text or ""
        cleaned = self._clean_json_text(raw_text)
        data = json.loads(cleaned)
        
        return ResearchContext(
            topic=topic,
            summary=str(data.get("summary", "")).strip(),
            facts=[str(f).strip() for f in data.get("facts", []) if str(f).strip()],
            statistics=[str(s).strip() for s in data.get("statistics", []) if str(s).strip()],
            timeline=[str(t).strip() for t in data.get("timeline", []) if str(t).strip()],
            locations=[str(l).strip() for l in data.get("locations", []) if str(l).strip()],
            people=[str(p).strip() for p in data.get("people", []) if str(p).strip()],
            scientific_explanations=[str(s).strip() for s in data.get("scientific_explanations", []) if str(s).strip()],
            myths=[str(m).strip() for m in data.get("myths", []) if str(m).strip()],
            controversies=[str(c).strip() for c in data.get("controversies", []) if str(c).strip()],
            sources=[str(s).strip() for s in data.get("sources", []) if str(s).strip()],
            interesting_hooks=[str(i).strip() for i in data.get("interesting_hooks", []) if str(i).strip()],
            warnings=[str(w).strip() for w in data.get("warnings", []) if str(w).strip()],
            confidence_score=float(data.get("confidence_score", 0.0) or 0.0),
        )

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text


class ResearchEngine:
    """Orchestrates structured knowledge collection for a given topic."""

    def __init__(self, settings: Settings, logger: logging.Logger, provider: BaseResearchProvider | None = None) -> None:
        self.settings = settings
        self.logger = logger
        self.provider = provider or GeminiResearchProvider(settings, logger)

    def research_topic(self, topic: str) -> ResearchContext:
        self.logger.info("Starting factual research gathering for topic: %s", topic)
        try:
            return self.provider.research(topic)
        except Exception as exc:
            self.logger.error("Factual research failed completely after retries: %s", exc)
            return ResearchContext(
                topic=topic,
                summary=f"Factual research gathering failed. Error: {exc}",
                facts=[],
                statistics=[],
                timeline=[],
                locations=[],
                people=[],
                scientific_explanations=[],
                myths=[],
                controversies=[],
                sources=[],
                interesting_hooks=[],
                warnings=[f"Research failed: {exc}"],
                confidence_score=0.0,
            )
