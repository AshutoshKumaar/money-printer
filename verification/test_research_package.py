from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
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
    ResearchPackageAdapter,
)
from research.research_engine import GeminiResearchProvider
from config import load_settings
import logging


class TestResearchPackage(unittest.TestCase):
    """Tests the immutability, schema validation, and legacy adapter of the Research Intelligence V2."""

    def setUp(self) -> None:
        self.settings = load_settings()
        self.logger = logging.getLogger("TestLogger")
        
        # Construct a valid dummy ResearchPackage structure
        self.package = ResearchPackage(
            topic="Test Topic",
            topic_summary="This is a test topic summary.",
            historical_context=HistoricalContext(
                timeline=["1999: Start event", "2005: End event"],
                overview="Historical overview text."
            ),
            scientific_context=ScientificContext(
                explanation="This is how the test system works.",
                concepts=["Concept A", "Concept B"]
            ),
            important_entities=ImportantEntities(
                people=["Person A: Lead creator"],
                places=["Location X"],
                organizations=["Org Y"],
                technologies=["Tech Z"],
                events=["Event E"],
                objects=["Object O"]
            ),
            verified_facts=[
                VerifiedFact(
                    fact="Water is H2O.",
                    confidence_score=0.99,
                    verification_status="verified",
                    reasoning="Chemical composition analysis.",
                    evidence_level="strong",
                    source_type="scientific consensus"
                )
            ],
            common_misconceptions=[
                CommonMisconception(
                    myth="Water is pure blue.",
                    verified_fact="Water is transparent, showing blue in mass reflection."
                )
            ],
            unanswered_questions=[
                UnansweredQuestion(
                    question="Why is this mystery unsolved?",
                    uncertainty_type="general"
                )
            ],
            visual_opportunities=[
                VisualOpportunity(
                    opportunity_type="diagram",
                    description="A diagram showing chemical bonds."
                )
            ],
            story_opportunities=StoryOpportunities(
                strongest_hook="Did you know this about water?",
                biggest_surprise="Surprising detail.",
                emotional_moments=["Awe of nature"],
                best_climax="Peak reveal.",
                strongest_ending="Lingering question."
            ),
            seo_research=SEOResearch(
                primary_keywords=["water", "h2o"],
                secondary_keywords=["chemistry"],
                related_concepts=["liquid"]
            ),
            research_confidence=ResearchConfidence(
                overall_score=0.99,
                potential_weak_areas=[],
                recommended_verification_priority="low"
            )
        )

    def test_immutability(self) -> None:
        """Verify that ResearchPackage is immutable (frozen=True)."""
        with self.assertRaises(FrozenInstanceError):
            # Attempting to mutate property should raise FrozenInstanceError
            self.package.topic = "New Topic"  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            # Attempting to reassign nested property should raise FrozenInstanceError
            self.package.historical_context.timeline = []  # type: ignore

    def test_package_adapter(self) -> None:
        """Verify that ResearchPackageAdapter maps all legacy ResearchContext fields correctly."""
        adapter = ResearchPackageAdapter(self.package)

        self.assertEqual(adapter.topic, "Test Topic")
        self.assertEqual(adapter.summary, "This is a test topic summary.")
        self.assertEqual(adapter.facts, ["Water is H2O."])
        self.assertEqual(adapter.timeline, ["1999: Start event", "2005: End event"])
        self.assertEqual(adapter.locations, ["Location X"])
        self.assertEqual(adapter.people, ["Person A: Lead creator"])
        self.assertEqual(adapter.scientific_explanations, ["This is how the test system works."])
        self.assertEqual(adapter.myths, ["Myth: Water is pure blue. | Truth: Water is transparent, showing blue in mass reflection."])
        self.assertEqual(adapter.controversies, ["Why is this mystery unsolved?"])
        self.assertEqual(adapter.interesting_hooks, ["Did you know this about water?"])
        self.assertEqual(adapter.warnings, [])
        self.assertEqual(adapter.confidence_score, 0.99)
        self.assertEqual(adapter.sources, [])
        self.assertEqual(adapter.statistics, [])

        # Verify serialization contains the adapted schema keys
        d = adapter.to_dict()
        self.assertIn("summary", d)
        self.assertIn("facts", d)
        self.assertIn("timeline", d)
        self.assertIn("myths", d)

    def test_validation_rules(self) -> None:
        """Verify that schema validation correctly identifies malformed payloads."""
        provider = GeminiResearchProvider(self.settings, self.logger)

        # 1. Missing main sections
        invalid_data_1 = {
            "topic_summary": "Incomplete data"
        }
        with self.assertRaises(ValueError) as ctx:
            provider.validate_schema(invalid_data_1)  # type: ignore
        self.assertIn("Missing required schema sections", str(ctx.exception))

        # 2. Empty facts list
        invalid_data_2 = {
            "topic_summary": "Test Summary",
            "historical_context": {},
            "scientific_context": {},
            "important_entities": {},
            "verified_facts": [],  # Empty
            "common_misconceptions": [],
            "unanswered_questions": [],
            "visual_opportunities": [],
            "story_opportunities": {},
            "seo_research": {},
            "research_confidence": {}
        }
        with self.assertRaises(ValueError) as ctx:
            provider.validate_schema(invalid_data_2)  # type: ignore
        self.assertIn("verified_facts must be a non-empty list", str(ctx.exception))

        # 3. Invalid enum constraint in fact status
        invalid_data_3 = {
            "topic_summary": "Test Summary",
            "historical_context": {},
            "scientific_context": {},
            "important_entities": {},
            "verified_facts": [
                {
                    "fact": "Water is H2O",
                    "confidence_score": 0.99,
                    "verification_status": "super-verified",  # Invalid enum value
                    "reasoning": "Tested",
                    "evidence_level": "strong",
                    "source_type": "scientific consensus"
                }
            ],
            "common_misconceptions": [],
            "unanswered_questions": [],
            "visual_opportunities": [],
            "story_opportunities": {},
            "seo_research": {},
            "research_confidence": {}
        }
        with self.assertRaises(ValueError) as ctx:
            provider.validate_schema(invalid_data_3)  # type: ignore
        self.assertIn("invalid verification_status", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
