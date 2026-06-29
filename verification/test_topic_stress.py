from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# reconfigure stdout for unicode output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add workspace root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import load_settings
from analytics.analytics_engine import AnalyticsEngine
from topic.topic_history import TopicHistory
from topic.category_manager import CategoryManager
from topic.topic_engine import TopicEngine
from topic.models import TopicDecision, generate_fingerprint, compute_similarity
from analytics.feedback_engine import FeedbackEngine


class TestTopicStress(unittest.TestCase):
    """Stress testing the Topic Engine across 100+ iterations for duplicate avoidance and category rotation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_settings()
        cls.logger = logging.getLogger("StressTest")
        cls.logger.setLevel(logging.INFO)
        
        # Initialize engines
        cls.analytics = AnalyticsEngine(cls.settings, cls.logger)
        cls.history = TopicHistory(cls.settings, cls.logger, cls.analytics)
        cls.category_manager = CategoryManager(cls.settings, cls.logger)
        
        # Inject 30 unique subjects into each category to prevent candidate exhaustion in 100 runs
        for cat, data in cls.category_manager.categories.items():
            if "subjects" in data:
                for i in range(30):
                    data["subjects"].append(f"UniqueSub{cat.capitalize()}{i}")

        cls.engine = TopicEngine(cls.settings, cls.logger, cls.analytics, cls.history, cls.category_manager)
        cls.feedback = FeedbackEngine(cls.settings, cls.logger)

        # Mock the Gemini client to avoid external API calls during stress testing
        cls.engine.client = MagicMock()
        
    def test_stress_topic_pipeline(self) -> None:
        print("\n--- Starting Topic Engine 100-Iteration Stress Test ---")
        
        # Save original history file contents to restore it later
        original_history_content = ""
        if self.analytics.history_path.exists():
            original_history_content = self.analytics.history_path.read_text(encoding="utf-8")
            
        # Temporary clear history file to start fresh for stress test
        self.analytics.history_path.write_text("[]", encoding="utf-8")
        
        selected_categories: list[str] = []
        selected_topics: list[str] = []
        
        # Generate and save topics across 100 iterations
        try:
            for i in range(1, 101):
                # Mock Gemini to choose the first candidate with minor prefix refinement
                def mock_generate_content(*args, **kwargs):
                    prompt = kwargs.get("contents", args[0] if args else "")
                    # Extract list of candidates from prompt
                    candidates = []
                    for line in prompt.split("\n"):
                        if line.startswith("- "):
                            candidates.append(line[2:])
                    
                    chosen = candidates[0] if candidates else f"Mock Topic {i}"
                    # Return mock response
                    mock_resp = MagicMock()
                    mock_resp.text = json.dumps({
                        "topic": f"{chosen} (Refined)",
                        "original_candidate": chosen,
                        "rationale": "Mock selection for stress testing",
                        "is_evergreen": True,
                        "is_trending": False
                    })
                    return mock_resp
                
                import json
                self.engine.client.models.generate_content = mock_generate_content
                
                # Execute decision
                decision = self.engine.decide_topic()
                selected_categories.append(decision.category)
                selected_topics.append(decision.topic)
                
                # Save run performance in history to affect future selections
                mock_video = MagicMock()
                mock_video.youtube_url = f"https://youtube.com/watch?v=mock_{i}"
                mock_video.metadata_path = Path(f"storage/metadata/mock_{i}.json")
                decision.run_id = f"mock_{i}"
                
                self.feedback.save_run_performance(mock_video, decision)
                
            print(f"Successfully ran 100 iterations!")
            
            # --- Verification 1: Category Rotation & Avoidance ---
            category_counts = {}
            for cat in selected_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
            print("\nCategory Distribution Summary:")
            for cat, count in category_counts.items():
                print(f"  {cat}: {count} times")
                
            # Verify that we have a distribution across all categories
            self.assertGreater(len(category_counts), 3, "Categories are not rotating properly.")
            
            # Check rotation logic: same category shouldn't be selected in immediate succession
            successive_repeats = 0
            for idx in range(len(selected_categories) - 1):
                if selected_categories[idx] == selected_categories[idx + 1]:
                    successive_repeats += 1
            
            print(f"Successive category repeats: {successive_repeats}")
            self.assertLessEqual(successive_repeats, 5, "Too many successive category repeats. Rotation is failing.")

            # --- Verification 2: Duplicate Topic Avoidance ---
            # Compare all generated topics pairwise to make sure none exceed the similarity threshold (0.5)
            duplicate_pairs = []
            for i in range(len(selected_topics)):
                f1 = generate_fingerprint(selected_topics[i])
                for j in range(i + 1, len(selected_topics)):
                    f2 = generate_fingerprint(selected_topics[j])
                    sim = compute_similarity(f1, f2)
                    if sim >= 0.5:
                        duplicate_pairs.append((selected_topics[i], selected_topics[j], sim))
                        
            print(f"Duplicate pairs detected: {len(duplicate_pairs)}")
            for p in duplicate_pairs[:5]:
                print(f"  Similarity {p[2]:.2f}: '{p[0]}' vs '{p[1]}'")
                
            self.assertEqual(len(duplicate_pairs), 0, f"Duplicate topics were generated! Count: {len(duplicate_pairs)}")
            print("No duplicate topics were generated! All topic fingerprints are distinct.")
            print("Topic Engine stress tests passed successfully!")
            
        finally:
            # Restore original history content
            if original_history_content:
                self.analytics.history_path.write_text(original_history_content, encoding="utf-8")
            else:
                # If it was empty or missing, delete it
                if self.analytics.history_path.exists():
                    self.analytics.history_path.unlink()


if __name__ == "__main__":
    unittest.main()
