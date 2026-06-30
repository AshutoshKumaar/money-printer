from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from config import Settings
from core.retry import retry_call
from topic.models import TopicDecision
from topic.category_manager import CategoryManager
from topic.topic_history import TopicHistory
from analytics.analytics_engine import AnalyticsEngine


class TopicEngine:
    """Implements the two-stage Topic Selection pipeline: Stage 1 (Python candidate generation) and Stage 2 (LLM refinement)."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        analytics_engine: AnalyticsEngine,
        topic_history: TopicHistory,
        category_manager: CategoryManager
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.analytics_engine = analytics_engine
        self.topic_history = topic_history
        self.category_manager = category_manager
        
        api_key = self.settings.gemini_api_key
        if not api_key:
            self.logger.warning("No GEMINI_API_KEY found in settings. TopicEngine will use fallback selection.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

    def decide_topic(
        self,
        manual_topic: str | None = None,
        topic: str | None = None,
    ) -> TopicDecision:
        """
        Decides the next topic. If a manual topic is provided by the user, wraps it in a TopicDecision.
        Otherwise, runs the Two-Stage Topic Selection pipeline.
        """
        resolved_manual_topic = manual_topic or topic
        if resolved_manual_topic:
            self.logger.info("Manual topic override provided: '%s'", resolved_manual_topic)
            guessed_cat = self.analytics_engine.guess_category(resolved_manual_topic, "")
            return TopicDecision(
                topic=resolved_manual_topic,
                category=guessed_cat,
                is_evergreen=True,
                is_trending=False,
                rationale="Manual topic override by user",
                metadata={"manual": True}
            )

        # Stage 1: Python Only Candidate Generation and Ranking
        self.logger.info("Executing Stage 1: Python Candidate Selection...")
        category = self.category_manager.select_category(self.topic_history, self.analytics_engine)
        
        # Generate candidates from category templates and subjects
        raw_candidates = self.category_manager.generate_candidates(category)
        if not raw_candidates:
            raise ValueError(f"No candidates could be generated for category: '{category}'")

        # Check if learned weights exist
        learning_state_path = self.settings.storage_dir / "learning_state.json"
        learned_topic_weights = {}
        if learning_state_path.exists():
            try:
                state_data = json.loads(learning_state_path.read_text(encoding="utf-8"))
                learned_topic_weights = state_data.get("topic_weights", {})
            except Exception:
                pass

        # Filter duplicates and rank remaining candidates
        ranked_candidates: list[tuple[str, float]] = []
        for cand in raw_candidates:
            # Check for duplicate similarity threshold (0.5)
            dup, max_sim, closest = self.topic_history.is_duplicate(cand, threshold=0.5)
            if dup:
                # Discard duplicate topics
                continue
            
            # Simple scoring: baseline is 10.0 + topic performance score, scaled down by similarity penalty
            base_perf = self.analytics_engine.get_topic_score(cand)
            
            # Incorporate learned topic weight if available
            learned_w = learned_topic_weights.get(cand, 1.0)
            score = (10.0 + (base_perf - 1.0) * 5.0) * (1.0 - max_sim) * learned_w
            ranked_candidates.append((cand, score))

        # Sort candidates descending by score
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top 5-10 candidates
        candidates = [item[0] for item in ranked_candidates[:10]]
        
        if not candidates:
            self.logger.warning("All candidate topics classified as duplicates. Relaxing duplicate filters.")
            candidates = raw_candidates[:5]

        self.logger.info("Stage 1 complete. Generated %d ranked candidates for category '%s'", len(candidates), category)

        # Stage 2: LLM refinement and selection
        if not self.client:
            self.logger.warning("Gemini client unavailable. Falling back to highest-ranked Stage 1 candidate.")
            fallback_topic = candidates[0]
            return TopicDecision(
                topic=fallback_topic,
                category=category,
                is_evergreen=True,
                is_trending=False,
                rationale="Fallback to highest-ranked candidate due to missing Gemini API key"
            )

        self.logger.info("Executing Stage 2: LLM Candidate Selection & Refinement...")
        candidates_text = "\n".join(f"- {c}" for c in candidates)
        
        prompt = f"""
You are the Topic Intelligence model for a Hindi YouTube Shorts educational channel.
Your task is to evaluate and refine the following list of candidate topics from the category '{category}':

Candidates:
{candidates_text}

Guidelines:
1. Evaluate the candidate list and select the single strongest, most curiosity-driven candidate.
2. Refine the wording of the chosen topic to make it highly catchy and optimized for a 60-second vertical Hindi Short.
3. Keep the content strictly factual, educational, and neutral. Do NOT generate opinion-based political content or propaganda.
4. Ensure the topic fits within the chosen category '{category}'. Do not invent completely unrelated topics unless all candidates are rejected (explain why in the rationale).

Respond ONLY with a JSON object containing these keys:
- "topic": Refined final Hindi/English topic (plain text, e.g. "अंतरिक्ष के 3 खौफनाक रहस्य")
- "original_candidate": The candidate topic from the list that you selected
- "rationale": Clear explanation of why you selected and how you refined this topic
- "is_evergreen": true/false (whether the topic is evergreen)
- "is_trending": true/false (whether the topic is trending)
"""

        try:
            response = retry_call(
                lambda: self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                ),
                attempts=self.settings.retry_attempts,
                backoff_seconds=self.settings.retry_backoff_seconds,
                logger=self.logger,
                label="Gemini topic intelligence refinement",
            )
            raw_text = (response.text or "").strip()
            data = json.loads(raw_text)
            
            refined_topic = data.get("topic", "").strip()
            original_candidate = data.get("original_candidate", "").strip()
            rationale = data.get("rationale", "").strip()
            is_evergreen = bool(data.get("is_evergreen", True))
            is_trending = bool(data.get("is_trending", False))

            if not refined_topic:
                raise ValueError("Gemini returned empty topic text")

            self.logger.info("Stage 2 complete. Selected topic: '%s' (Original: '%s')", refined_topic, original_candidate)
            return TopicDecision(
                topic=refined_topic,
                category=category,
                is_evergreen=is_evergreen,
                is_trending=is_trending,
                rationale=rationale,
                metadata={"original_candidate": original_candidate}
            )

        except Exception as exc:
            self.logger.error("Stage 2 LLM topic selection failed: %s. Using highest-ranked Stage 1 candidate.", exc)
            fallback_topic = candidates[0]
            return TopicDecision(
                topic=fallback_topic,
                category=category,
                is_evergreen=True,
                is_trending=False,
                rationale=f"Fallback to highest-ranked candidate due to exception: {exc}"
            )
