from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from config import load_settings
from research.models import (
    ResearchPackage,
    HistoricalContext,
    ScientificContext,
    ImportantEntities,
    StoryOpportunities,
    SEOResearch,
    ResearchConfidence,
)
from verification.models import VerifiedResearchPackage, VerifiedFactRecord
from story.models import NarrativePackage, NarrationSegment, NarrativeQuality, NarrativeAdapter
from story.story_engine import GeminiStoryProvider
import logging


class TestStoryV2(unittest.TestCase):
    """Unit tests for Story Intelligence V2, verifying duration checks, fact traceability, and adapter compatibility."""

    def setUp(self) -> None:
        self.settings = load_settings()
        self.logger = logging.getLogger("TestLogger")

        # Mock research package
        research_pkg = ResearchPackage(
            topic="Test Topic",
            topic_summary="Summary",
            historical_context=HistoricalContext([], ""),
            scientific_context=ScientificContext("", []),
            important_entities=ImportantEntities([], [], [], [], [], []),
            verified_facts=[],
            common_misconceptions=[],
            unanswered_questions=[],
            visual_opportunities=[],
            story_opportunities=StoryOpportunities("", "", [], "", ""),
            seo_research=SEOResearch([], [], []),
            research_confidence=ResearchConfidence(0.0, [], "moderate")
        )

        # Mock verified facts
        self.verified_package = VerifiedResearchPackage(
            research_package=research_pkg,
            verified_facts=[
                VerifiedFactRecord(
                    fact="Factual claim A",
                    status="verified",
                    original_fact="Factual claim A",
                    suggested_clarification=None,
                    confidence_score=0.9,
                    reasoning="Evidence",
                    evidence_level="strong",
                    source_type="scientific consensus",
                    importance_score=0.8,
                    category="science"
                ),
                VerifiedFactRecord(
                    fact="Factual claim B",
                    status="verified",
                    original_fact="Factual claim B",
                    suggested_clarification=None,
                    confidence_score=0.95,
                    reasoning="Evidence",
                    evidence_level="strong",
                    source_type="scientific consensus",
                    importance_score=0.7,
                    category="science"
                )
            ]
        )

    def test_immutability(self) -> None:
        """Verify that NarrativePackage is immutable (frozen=True)."""
        package = NarrativePackage(
            language="hi",
            hook="Grabbing hook",
            context="Context",
            escalation="Escalation",
            climax="Climax",
            ending="Ending",
            narration_segments=[],
            quality=NarrativeQuality(0.9, 0.9, 0.9, 0.9, 0.9, [1.0])
        )
        with self.assertRaises(FrozenInstanceError):
            package.hook = "New Hook"  # type: ignore

    def test_duration_and_traceability_validation(self) -> None:
        """Verify duration range constraints (35-55s) and fact ID traceability checks."""
        provider = GeminiStoryProvider(self.settings, self.logger)

        # 1. Total duration too short (30s)
        invalid_data_1 = {
            "language": "hi",
            "hook": "Hook",
            "context": "Context",
            "escalation": "Escalation",
            "climax": "Climax",
            "ending": "Ending",
            "narration_segments": [
                {
                    "index": 1,
                    "narration_text": "Claim text",
                    "estimated_duration": 30.0,  # Total = 30s
                    "target_start": 0.0,
                    "target_end": 30.0,
                    "emotion": "curiosity",
                    "purpose": "Evidence hook",
                    "verified_fact_ids": ["fact_1"],
                    "beat_type": "setup"
                }
            ],
            "quality": {
                "retention_score": 0.9,
                "pacing_score": 0.9,
                "curiosity_score": 0.9,
                "clarity_score": 0.9,
                "emotional_score": 0.9,
                "estimated_retention_curve": [1.0]
            }
        }
        with self.assertRaises(ValueError) as ctx:
            provider.validate_schema(invalid_data_1, self.verified_package)
        self.assertIn("duration", str(ctx.exception))
        self.assertIn("outside target range", str(ctx.exception))

        # 2. Invalid fact ID reference (references fact_3 which doesn't exist)
        invalid_data_2 = {
            "language": "hi",
            "hook": "Hook",
            "context": "Context",
            "escalation": "Escalation",
            "climax": "Climax",
            "ending": "Ending",
            "narration_segments": [
                {
                    "index": 1,
                    "narration_text": "Claim text",
                    "estimated_duration": 40.0,  # Total = 40s (valid)
                    "target_start": 0.0,
                    "target_end": 40.0,
                    "emotion": "curiosity",
                    "purpose": "Evidence hook",
                    "verified_fact_ids": ["fact_3"],  # Invalid! Only fact_1 and fact_2 exist
                    "beat_type": "setup"
                }
            ],
            "quality": {
                "retention_score": 0.9,
                "pacing_score": 0.9,
                "curiosity_score": 0.9,
                "clarity_score": 0.9,
                "emotional_score": 0.9,
                "estimated_retention_curve": [1.0]
            }
        }
        with self.assertRaises(ValueError) as ctx:
            provider.validate_schema(invalid_data_2, self.verified_package)
        self.assertIn("references invalid fact ID", str(ctx.exception))

    def test_adapter_compatibility(self) -> None:
        """Verify that NarrativeAdapter maps NarrativePackage properties correctly to legacy fields."""
        segments = [
            NarrationSegment(
                index=1,
                narration_text="Hook spoken text",
                estimated_duration=10.0,
                target_start=0.0,
                target_end=10.0,
                emotion="suspense",
                purpose="Hook curiosity",
                verified_fact_ids=["fact_1"],
                beat_type="hook"
            ),
            NarrationSegment(
                index=2,
                narration_text="Context spoken text",
                estimated_duration=30.0,
                target_start=10.0,
                target_end=40.0,
                emotion="wonder",
                purpose="Context setup",
                verified_fact_ids=["fact_2"],
                beat_type="setup"
            )
        ]
        
        quality = NarrativeQuality(
            retention_score=0.92,
            pacing_score=0.88,
            curiosity_score=0.95,
            clarity_score=0.90,
            emotional_score=0.85,
            estimated_retention_curve=[1.0, 0.9, 0.8]
        )

        pkg = NarrativePackage(
            language="hi",
            hook="Grabbing Hook",
            context="Introduction Context",
            escalation="Escalation",
            climax="Climax Reveal",
            ending="Concluding wrap-up",
            narration_segments=segments,
            quality=quality
        )

        adapter = NarrativeAdapter(pkg)

        self.assertEqual(adapter.hook, "Grabbing Hook")
        self.assertEqual(adapter.context, "Introduction Context")
        self.assertEqual(adapter.ending, "Concluding wrap-up")
        self.assertEqual(adapter.estimated_duration, 40.0)
        self.assertEqual(len(adapter.segments), 2)
        
        # Verify first segment
        s0 = adapter.segments[0]
        self.assertEqual(s0.index, 1)
        self.assertEqual(s0.spoken_hindi, "Hook spoken text")
        self.assertEqual(s0.caption_keywords, "")  # Fallback
        self.assertEqual(s0.visual_concept, "Visual representation of narration.")  # Fallback
        
        # Verify emotion curve mapping (suspense -> curiosity=0.8, urgency=0.6)
        self.assertEqual(s0.emotion_curve.curiosity, 0.8)
        self.assertEqual(s0.emotion_curve.urgency, 0.6)

        # Verify dict representation keys
        d = adapter.to_dict()
        self.assertIn("hook", d)
        self.assertIn("context", d)
        self.assertIn("segments", d)
        self.assertEqual(d["estimated_duration"], 40.0)
        self.assertEqual(d["retention_score"], 0.92)


if __name__ == "__main__":
    unittest.main()
