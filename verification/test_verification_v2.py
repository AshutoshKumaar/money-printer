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
from verification.models import VerifiedResearchPackage, VerifiedFactRecord, VerificationAdapter
from verification.verification_engine import GeminiVerificationProvider
import logging


class TestVerificationV2(unittest.TestCase):
    """Unit tests for Verification V2, verifying contradictions, duplicates, weighted confidence, and adapter compatibility."""

    def setUp(self) -> None:
        self.settings = load_settings()
        self.logger = logging.getLogger("TestLogger")
        
        # Base dummy ResearchPackage
        self.research_package = ResearchPackage(
            topic="Test Topic",
            topic_summary="Summary",
            historical_context=HistoricalContext(timeline=[], overview=""),
            scientific_context=ScientificContext(explanation="", concepts=[]),
            important_entities=ImportantEntities(people=[], places=[], organizations=[], technologies=[], events=[], objects=[]),
            verified_facts=[],
            common_misconceptions=[],
            unanswered_questions=[],
            visual_opportunities=[],
            story_opportunities=StoryOpportunities("", "", [], "", ""),
            seo_research=SEOResearch([], [], []),
            research_confidence=ResearchConfidence(0.0, [], "moderate")
        )

    def test_immutability(self) -> None:
        """Verify that VerifiedResearchPackage is immutable (frozen=True)."""
        verified_package = VerifiedResearchPackage(
            research_package=self.research_package,
            verified_facts=[]
        )
        with self.assertRaises(FrozenInstanceError):
            verified_package.overall_confidence_score = 0.95  # type: ignore

    def test_weighted_confidence_calculation(self) -> None:
        """Verify that overall confidence is computed as importance-weighted average."""
        facts = [
            VerifiedFactRecord(
                fact="Fact A",
                status="verified",
                original_fact="Fact A",
                suggested_clarification=None,
                confidence_score=0.8,
                reasoning="Reasoning A",
                evidence_level="moderate",
                source_type="academic journal",
                importance_score=0.9,
                category="general"
            ),
            VerifiedFactRecord(
                fact="Fact B",
                status="partially_verified",
                original_fact="Fact B",
                suggested_clarification="Fact B (clarified)",
                confidence_score=0.5,
                reasoning="Reasoning B",
                evidence_level="weak",
                source_type="encyclopedia",
                importance_score=0.4,
                category="general"
            )
        ]

        provider = GeminiVerificationProvider(self.settings, self.logger)
        
        # Calculate expected: (0.8 * 0.9 + 0.5 * 0.4) / (0.9 + 0.4) = (0.72 + 0.2) / 1.3 = 0.92 / 1.3 = 0.707692...
        expected_weighted_conf = (0.8 * 0.9 + 0.5 * 0.4) / (0.9 + 0.4)
        
        verified_pkg = provider.parse_verified_package(
            self.research_package,
            {
                "verified_facts": [
                    {
                        "fact": f.fact,
                        "status": f.status,
                        "original_fact": f.original_fact,
                        "suggested_clarification": f.suggested_clarification,
                        "confidence_score": f.confidence_score,
                        "reasoning": f.reasoning,
                        "evidence_level": f.evidence_level,
                        "source_type": f.source_type,
                        "contradictions_detected": [],
                        "is_duplicate": f.is_duplicate,
                        "importance_score": f.importance_score,
                        "category": f.category
                    }
                    for f in facts
                ],
                "verification_warnings": [],
                "verification_quality_score": 0.90
            }
        )

        self.assertAlmostEqual(verified_pkg.overall_confidence_score, expected_weighted_conf, places=5)

    def test_duplicate_and_contradiction_handling_in_adapter(self) -> None:
        """Verify duplicate exclusion, contradiction mapping, and adapter backward compatibility."""
        facts = [
            VerifiedFactRecord(
                fact="Primary claim about topic",
                status="verified",
                original_fact="Primary claim about topic",
                suggested_clarification=None,
                confidence_score=0.9,
                reasoning="Solid evidence",
                evidence_level="strong",
                source_type="scientific consensus",
                is_duplicate=False,
                importance_score=0.8,
                category="science"
            ),
            VerifiedFactRecord(
                fact="Primary claim about topic",  # Duplicate fact
                status="verified",
                original_fact="Primary claim about topic",
                suggested_clarification=None,
                confidence_score=0.9,
                reasoning="Duplicate",
                evidence_level="strong",
                source_type="scientific consensus",
                is_duplicate=True,  # Flagged as duplicate
                importance_score=0.8,
                category="science"
            ),
            VerifiedFactRecord(
                fact="Conflicting statement about the event",
                status="disputed",  # Contradictory claim
                original_fact="Conflicting statement about the event",
                suggested_clarification=None,
                confidence_score=0.4,
                reasoning="Conflicting accounts exist.",
                evidence_level="moderate",
                source_type="unknown",
                contradictions_detected=["Contradicts primary source"],
                is_duplicate=False,
                importance_score=0.5,
                category="history"
            )
        ]

        verified_pkg = VerifiedResearchPackage(
            research_package=self.research_package,
            verified_facts=facts,
            verification_warnings=["Potential bias detected"],
            overall_confidence_score=0.75,
            verification_quality_score=0.85
        )

        adapter = VerificationAdapter(verified_pkg)

        # 1. Verify duplicate filtering
        # The duplicate fact (index 1) should be excluded from verified_facts
        self.assertEqual(len(adapter.verified_facts), 1)
        self.assertEqual(adapter.verified_facts[0].original_fact, "Primary claim about topic")
        
        # 2. Verify contradiction status mapping
        # Disputed maps to contradictory in legacy report
        self.assertEqual(len(adapter.rejected_facts), 1)
        self.assertEqual(adapter.rejected_facts[0].original_fact, "Conflicting statement about the event")
        self.assertEqual(adapter.rejected_facts[0].status, "contradictory")

        # 3. Adapter dictionary keys check
        d = adapter.to_dict()
        self.assertIn("verified_facts", d)
        self.assertIn("rejected_facts", d)
        self.assertIn("corrected_facts", d)
        self.assertEqual(d["confidence_score"], 0.75)
        self.assertEqual(d["research_quality_score"], 0.85)

    def test_validation_schema(self) -> None:
        """Verify that Verification provider correctly identifies schema violations."""
        provider = GeminiVerificationProvider(self.settings, self.logger)

        # Missing required keys
        invalid_data_1 = {
            "verified_facts": []
        }
        with self.assertRaises(ValueError) as ctx:
            provider.validate_schema(invalid_data_1)  # type: ignore
        self.assertIn("Missing required verification schema sections", str(ctx.exception))

        # Invalid outcome status enum
        invalid_data_2 = {
            "verified_facts": [
                {
                    "fact": "Fact A",
                    "status": "super-verified",  # Invalid enum value
                    "original_fact": "Fact A",
                    "suggested_clarification": None,
                    "confidence_score": 0.9,
                    "reasoning": "Reason",
                    "evidence_level": "strong",
                    "source_type": "scientific consensus",
                    "contradictions_detected": [],
                    "is_duplicate": False,
                    "importance_score": 0.8,
                    "category": "science"
                }
            ],
            "verification_warnings": [],
            "verification_quality_score": 0.90
        }
        with self.assertRaises(ValueError) as ctx:
            provider.validate_schema(invalid_data_2)
        self.assertIn("invalid status", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
