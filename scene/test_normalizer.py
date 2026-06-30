from __future__ import annotations

import logging
import unittest
from scene.normalizer import SceneNormalizer


class TestSceneNormalizer(unittest.TestCase):
    """Unit tests for Scene Planner Enum Normalization Layer."""

    def setUp(self) -> None:
        self.logger = logging.getLogger("TestNormalizerLogger")
        self.input_json = {
            "estimated_total_duration": 15.0,
            "pacing_score": 0.8,
            "visual_variety_score": 0.85,
            "scenes": [
                {
                    "scene_id": "scene_1",
                    "narration_segment_id": 1,
                    "target_start": 0.0,
                    "target_end": 5.0,
                    "visual_type": "stock_video",
                    "visual_priority": " highest ",  # maps to critical
                    "transition": "fade in",  # maps to fade
                    "overlay": {
                        "overlay_type": "text",
                        "position": " upper ",  # maps to top
                        "animation": "slide in"  # maps to slide
                    },
                    "shots": [
                        {
                            "shot_id": "shot_1_1",
                            "visual_goal": "Wide shot of temple",
                            "camera_motion": {
                                "motion_type": "push-in",  # maps to zoom_in
                                "speed": None  # maps to medium
                            },
                            "duration": 5.0,
                            "transition_to_next": {
                                "transition_type": "cross fade"  # maps to crossfade
                            },
                            "visual_source_strategy": " ai ",  # maps to ai_preferred
                            "shot_type": " cu "  # maps to close_up
                        }
                    ],
                    "continuity_group": "temple_sequence"
                }
            ]
        }

    def test_recursive_and_alias_normalization(self) -> None:
        """Verify normalizer traverses recursively and converts aliases correctly."""
        normalized = SceneNormalizer.normalize(self.input_json, self.logger)
        
        scene = normalized["scenes"][0]
        self.assertEqual(scene["visual_priority"], "critical")
        self.assertEqual(scene["transition"]["transition_type"], "fade")
        self.assertEqual(scene["overlay"]["position"], "top")
        self.assertEqual(scene["overlay"]["animation"], "slide")
        
        shot = scene["shots"][0]
        self.assertEqual(shot["camera_motion"]["motion_type"], "zoom_in")
        self.assertEqual(shot["camera_motion"]["speed"], "medium")
        self.assertEqual(shot["transition_to_next"]["transition_type"], "crossfade")
        self.assertEqual(shot["visual_source_strategy"], "ai_preferred")
        self.assertEqual(shot["shot_type"], "close_up")

    def test_whitespace_and_punctuation_cleaning(self) -> None:
        """Verify normalizer handles multiple spaces, hyphens, and casing combinations."""
        raw = {
            "scenes": [
                {
                    "visual_priority": "  hi-gh  ",
                    "transition": {
                        "transition_type": "  CROSS_fade  "
                    },
                    "shots": [
                        {
                            "shot_type": "  extreme_wide  "
                        }
                    ]
                }
            ]
        }
        normalized = SceneNormalizer.normalize(raw)
        scene = normalized["scenes"][0]
        self.assertEqual(scene["visual_priority"], "high")
        self.assertEqual(scene["transition"]["transition_type"], "crossfade")
        self.assertEqual(scene["shots"][0]["shot_type"], "establishing")

    def test_idempotency(self) -> None:
        """Verify that running normalize multiple times yields the exact same structure without side-effects."""
        first_pass = SceneNormalizer.normalize(self.input_json)
        second_pass = SceneNormalizer.normalize(first_pass)
        
        # The fixes count on second pass should be 0 because it is already normalized
        self.assertEqual(second_pass["_normalization_fixes"], 0)
        
        # Structural equality
        self.assertEqual(first_pass["scenes"][0]["visual_priority"], second_pass["scenes"][0]["visual_priority"])
        self.assertEqual(
            first_pass["scenes"][0]["shots"][0]["shot_type"],
            second_pass["scenes"][0]["shots"][0]["shot_type"]
        )

    def test_strict_unknown_preservation(self) -> None:
        """Verify that unknown values are left completely untouched so the strict validator fails."""
        raw = {
            "scenes": [
                {
                    "visual_priority": "invalid_priority_value",
                    "transition": {
                        "transition_type": "weird_transition"
                    }
                }
            ]
        }
        normalized = SceneNormalizer.normalize(raw)
        scene = normalized["scenes"][0]
        self.assertEqual(scene["visual_priority"], "invalid_priority_value")
        self.assertEqual(scene["transition"]["transition_type"], "weird_transition")


if __name__ == "__main__":
    unittest.main()
