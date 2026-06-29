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
from story.models import NarrativePackage, NarrationSegment, NarrativeQuality
from scene.models import (
    ScenePackage,
    Scene,
    Shot,
    CameraInstruction,
    TransitionInstruction,
    OverlayInstruction,
    ScenePackageAdapter,
)
from scene.scene_engine import GeminiScenePlannerProvider
import logging


class TestSceneV2(unittest.TestCase):
    """Unit tests for Scene Planner V2, verifying duration matching, sequencing checks, and adapter flattening."""

    def setUp(self) -> None:
        self.settings = load_settings()
        self.logger = logging.getLogger("TestLogger")

        # Mock NarrativePackage with segment of 10s
        self.narrative = NarrativePackage(
            language="hi",
            hook="Hook",
            context="Context",
            escalation="Escalation",
            climax="Climax",
            ending="Ending",
            narration_segments=[
                NarrationSegment(
                    index=1,
                    narration_text="Narrative segment text",
                    estimated_duration=10.0,
                    target_start=0.0,
                    target_end=10.0,
                    emotion="curiosity",
                    purpose="Hook segment",
                    verified_fact_ids=["fact_1"],
                    beat_type="hook"
                )
            ],
            quality=NarrativeQuality(0.9, 0.9, 0.9, 0.9, 0.9, [1.0])
        )

    def test_immutability(self) -> None:
        """Verify that ScenePackage is immutable (frozen=True)."""
        package = ScenePackage(
            scenes=[],
            estimated_total_duration=0.0,
            pacing_score=0.9,
            visual_variety_score=0.9
        )
        with self.assertRaises(FrozenInstanceError):
            package.estimated_total_duration = 10.0  # type: ignore

    def test_duration_alignment_validation(self) -> None:
        """Verify that validation fails if sum of shot durations does not match scene duration."""
        provider = GeminiScenePlannerProvider(self.settings, self.logger)

        # Scene duration is 10.0s, but shots sum to 8.0s (mismatch)
        invalid_data_1 = {
            "scenes": [
                {
                    "scene_id": "scene_1",
                    "narration_segment_id": 1,
                    "target_start": 0.0,
                    "target_end": 10.0,
                    "visual_type": "stock_video",
                    "visual_priority": "high",
                    "transition": {
                        "transition_type": "fade"
                    },
                    "overlay": {
                        "overlay_type": "text",
                        "text": "Overlay text",
                        "position": "center",
                        "style": "default",
                        "animation": "none",
                        "duration": 10.0
                    },
                    "continuity_group": "default",
                    "shots": [
                        {
                            "shot_id": "shot_1_1",
                            "visual_goal": "Goal description",
                            "camera_motion": {
                                "motion_type": "static",
                                "speed": "medium"
                            },
                            "duration": 8.0,  # 8s != 10s
                            "transition_to_next": {
                                "transition_type": "none"
                            },
                            "visual_reference": None,
                            "visual_source_strategy": "stock_only",
                            "shot_type": "medium",
                            "aspect_ratio_hint": "9:16",
                            "safe_crop_region": None,
                            "focus_subject": None
                        }
                    ]
                }
            ],
            "estimated_total_duration": 10.0,
            "pacing_score": 0.9,
            "visual_variety_score": 0.9
        }
        with self.assertRaises(ValueError) as ctx:
            provider.validate_schema(invalid_data_1, self.narrative)
        self.assertIn("does not match sum of shot durations", str(ctx.exception))

    def test_sequencing_gaps_validation(self) -> None:
        """Verify that validation fails if there is a gap or overlap in scene timing."""
        provider = GeminiScenePlannerProvider(self.settings, self.logger)

        # Gap between scene 1 (ends at 5.0) and scene 2 (starts at 6.0)
        invalid_data_2 = {
            "scenes": [
                {
                    "scene_id": "scene_1",
                    "narration_segment_id": 1,
                    "target_start": 0.0,
                    "target_end": 5.0,
                    "visual_type": "stock_video",
                    "visual_priority": "high",
                    "transition": {"transition_type": "none"},
                    "overlay": {"overlay_type": "none", "text": None},
                    "continuity_group": "default",
                    "shots": [
                        {
                            "shot_id": "shot_1_1",
                            "visual_goal": "Goal 1",
                            "camera_motion": {"motion_type": "static", "speed": "medium"},
                            "duration": 5.0,
                            "transition_to_next": {"transition_type": "none"},
                            "visual_reference": None,
                            "visual_source_strategy": "stock_only"
                        }
                    ]
                },
                {
                    "scene_id": "scene_2",
                    "narration_segment_id": 2,
                    "target_start": 6.0,  # Gap of 1.0s (ends at 5.0s)
                    "target_end": 10.0,
                    "visual_type": "stock_video",
                    "visual_priority": "high",
                    "transition": {"transition_type": "none"},
                    "overlay": {"overlay_type": "none", "text": None},
                    "continuity_group": "default",
                    "shots": [
                        {
                            "shot_id": "shot_2_1",
                            "visual_goal": "Goal 2",
                            "camera_motion": {"motion_type": "static", "speed": "medium"},
                            "duration": 4.0,
                            "transition_to_next": {"transition_type": "none"},
                            "visual_reference": None,
                            "visual_source_strategy": "stock_only"
                        }
                    ]
                }
            ],
            "estimated_total_duration": 10.0,
            "pacing_score": 0.9,
            "visual_variety_score": 0.9
        }
        with self.assertRaises(ValueError) as ctx:
            provider.validate_schema(invalid_data_2, self.narrative)
        self.assertIn("timing gap or overlap detected", str(ctx.exception))

    def test_adapter_compatibility_and_flattening(self) -> None:
        """Verify that ScenePackageAdapter flattens nested shots and maps legacy SceneShot attributes."""
        shots_1 = [
            Shot("shot_1_1", "Goal 1", CameraInstruction("zoom_in", "slow"), 4.0, TransitionInstruction("none"), "ref_1", "stock_only"),
            Shot("shot_1_2", "Goal 2", CameraInstruction("static", "medium"), 6.0, TransitionInstruction("fade"), "ref_2", "ai_preferred")
        ]
        
        scenes = [
            Scene(
                scene_id="scene_1",
                narration_segment_id=1,
                target_start=0.0,
                target_end=10.0,
                visual_type="stock_video",
                visual_priority="high",
                transition=TransitionInstruction("fade"),
                overlay=OverlayInstruction("text", "Text", "center", "default", "none", 10.0),
                shots=shots_1,
                continuity_group="contin_1"
            )
        ]

        pkg = ScenePackage(scenes, 10.0, 0.9, 0.9)
        adapter = ScenePackageAdapter(pkg)

        self.assertEqual(adapter.scene_count, 1)
        self.assertEqual(adapter.estimated_runtime, 10.0)
        
        # Flatted shots count should be 2
        self.assertEqual(len(adapter.scenes), 2)

        # Verify Shot 1
        s0 = adapter.scenes[0]
        self.assertEqual(s0.scene_index, 1)
        self.assertEqual(s0.shot_index, 1)
        self.assertEqual(s0.duration_seconds, 4.0)
        self.assertEqual(s0.priority, "HIGH")
        self.assertEqual(s0.visual_description, "Goal 1")
        self.assertEqual(s0.transition_in, "fade")
        self.assertEqual(s0.transition_out, "none")
        self.assertIn("Goal 1", s0.ai_image_prompt)
        self.assertIn("contin_1", s0.ai_image_prompt)

        # Verify Shot 2
        s1 = adapter.scenes[1]
        self.assertEqual(s1.scene_index, 1)
        self.assertEqual(s1.shot_index, 2)
        self.assertEqual(s1.duration_seconds, 6.0)
        self.assertEqual(s1.transition_out, "fade")

        # Verify serializability
        d = adapter.to_dict()
        self.assertIn("scenes", d)
        self.assertEqual(len(d["scenes"]), 2)
        self.assertEqual(d["scenes"][0]["scene_index"], 1)
        self.assertEqual(d["scenes"][0]["shot_index"], 1)


if __name__ == "__main__":
    unittest.main()
