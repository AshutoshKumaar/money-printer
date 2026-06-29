from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from config import load_settings
from learning.models import LearningState
from learning.learning_engine import LearningEngine
import logging


class TestLearningEngine(unittest.TestCase):
    """Unit tests for Learning Engine, verifying weight learning, normalizations, and self-healing recovery."""

    def setUp(self) -> None:
        self.settings = load_settings()
        self.logger = logging.getLogger("TestLogger")
        self.temp_dir = TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name)
        self.history_path = self.storage_dir / "analytics_history.json"
        self.state_path = self.storage_dir / "learning_state.json"
        
        # Override storage path settings
        object.__setattr__(self.settings, "storage_dir", self.storage_dir)
        self.engine = LearningEngine(self.settings, self.logger)

        # Seed categories topics directory to support category detection
        topics_dir = self.storage_dir / "topics"
        topics_dir.mkdir(parents=True, exist_ok=True)
        (topics_dir / "history.json").write_text('{"category": "history"}', encoding="utf-8")
        (topics_dir / "ocean.json").write_text('{"category": "ocean"}', encoding="utf-8")
        (topics_dir / "space.json").write_text('{"category": "space"}', encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_state(self) -> None:
        """Verify default state generation when no data exists."""
        state = self.engine.get_default_state()
        self.assertIn("history", state.category_weights)
        self.assertAlmostEqual(sum(state.category_weights.values()), 1.0)
        self.assertEqual(state.confidence_scores["overall"], 0.0)

    def test_insufficient_data(self) -> None:
        """Verify that training with insufficient valid data returns previous/default weights."""
        # Seed only 1 record
        records = [
            {
                "run_id": "run_1",
                "category": "history",
                "views": 100,
                "performance_score": 80.0,
            }
        ]
        self.history_path.write_text(json.dumps(records), encoding="utf-8")
        state = self.engine.train()
        # Should be default since sample count < 3
        self.assertEqual(state.confidence_scores["overall"], 0.0)

    def test_rolling_averages_and_normalization(self) -> None:
        """Verify rolling average calculation and normalization of category weights."""
        records = [
            {"run_id": "run_1", "category": "history", "views": 100, "performance_score": 80.0},
            {"run_id": "run_2", "category": "history", "views": 200, "performance_score": 90.0},
            {"run_id": "run_3", "category": "ocean", "views": 500, "performance_score": 50.0},
        ]
        self.history_path.write_text(json.dumps(records), encoding="utf-8")
        state = self.engine.train()

        # Check weights sum to 1.0
        self.assertAlmostEqual(sum(state.category_weights.values()), 1.0)
        
        # history has average performance 85.0, ocean has 50.0.
        # space was seeded but had no records, so it has baseline/default.
        # Therefore, history weight should be higher than ocean weight.
        self.assertTrue(state.category_weights["history"] > state.category_weights["ocean"])
        self.assertTrue(state.confidence_scores["overall"] > 0.0)

    def test_determinism(self) -> None:
        """Verify that training is fully deterministic and reproducible."""
        records = [
            {"run_id": "run_1", "category": "history", "views": 100, "performance_score": 80.0},
            {"run_id": "run_2", "category": "history", "views": 200, "performance_score": 90.0},
            {"run_id": "run_3", "category": "ocean", "views": 500, "performance_score": 50.0},
        ]
        self.history_path.write_text(json.dumps(records), encoding="utf-8")
        state1 = self.engine.train()
        state2 = self.engine.train()
        
        self.assertEqual(state1.category_weights, state2.category_weights)
        self.assertEqual(state1.topic_weights, state2.topic_weights)
        self.assertEqual(state1.confidence_scores, state2.confidence_scores)

    def test_corrupted_history_recovery(self) -> None:
        """Verify recovery logic when history file contains corrupted or invalid JSON lines."""
        corrupted_content = """
        [
          {"run_id": "run_1", "category": "history", "views": 100, "performance_score": 80.0},
          {"run_id": "run_2", "category": "history", "views": 200, "performance_score": 90.0},
          INVALID_GARBAGE_JSON_LINE,
          {"run_id": "run_3", "category": "ocean", "views": 500, "performance_score": 50.0}
        ]
        """
        self.history_path.write_text(corrupted_content, encoding="utf-8")
        
        # Should not crash, and should successfully recover the 3 valid JSON lines
        state = self.engine.train()
        self.assertTrue(state.confidence_scores["overall"] > 0.0)
        self.assertAlmostEqual(sum(state.category_weights.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
