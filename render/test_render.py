from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from config import load_settings
from story.models import NarrativePackage, NarrationSegment, NarrativeQuality
from scene.models import ScenePackage, Scene, Shot, CameraInstruction, TransitionInstruction, OverlayInstruction
from visual.models import VisualPackage, NewVisualAsset, AssetMetadata
from render.models import RenderPackage, RenderPackageAdapter
from render.render_engine import RenderEngine
import logging


class TestRenderEngine(unittest.TestCase):
    """Unit tests for Render Engine, verifying determinism, timeline logic, and transitions."""

    def setUp(self) -> None:
        self.settings = load_settings()
        self.logger = logging.getLogger("TestLogger")
        self.engine = RenderEngine(self.settings, self.logger)

        # Mock NarrativePackage
        self.narrative = NarrativePackage(
            language="hi",
            hook="Hook text",
            context="Context text",
            escalation="Escalation text",
            climax="Climax text",
            ending="Ending text",
            narration_segments=[
                NarrationSegment(
                    index=1,
                    narration_text="Yeh ek short video ka narration segment hai.",
                    estimated_duration=4.5,
                    target_start=0.0,
                    target_end=4.5,
                    emotion="curiosity",
                    purpose="Hook",
                    verified_fact_ids=["fact_1"],
                    beat_type="hook"
                )
            ],
            quality=NarrativeQuality(0.9, 0.9, 0.9, 0.9, 0.9, [1.0])
        )

        # Mock ScenePackage
        self.scenes = [
            Scene(
                scene_id="scene_1",
                narration_segment_id=1,
                target_start=0.0,
                target_end=4.5,
                visual_type="stock_video",
                visual_priority="high",
                transition=TransitionInstruction("fade"),
                overlay=OverlayInstruction("none"),
                shots=[
                    Shot(
                        shot_id="shot_1_1",
                        visual_goal="Climax visual of temple",
                        camera_motion=CameraInstruction("zoom_in", "slow"),
                        duration=4.5,
                        transition_to_next=TransitionInstruction("crossfade"),
                        visual_reference="temple_opp",
                        visual_source_strategy="stock_only"
                    )
                ],
                continuity_group="temple_sequence"
            )
        ]
        self.scene_pkg = ScenePackage(self.scenes, 4.5, 0.9, 0.9)

        # Mock VisualPackage
        metadata = AssetMetadata()
        self.assets = [
            NewVisualAsset(
                shot_id="shot_1_1",
                asset_id="asset_temple",
                source="stock",
                confidence=0.9,
                cache_hit=False,
                continuity_group="temple_sequence",
                quality_score=0.9,
                resolution="1080x1920",
                orientation="vertical",
                local_path="path/to/temple.jpg",
                metadata=metadata
            )
        ]
        self.visual_pkg = VisualPackage(self.assets, [], 4.5, 0, 0.0)

        self.voice_paths = ["path/to/voice_1.mp3"]

    def test_immutability(self) -> None:
        """Verify that RenderPackage is immutable."""
        package = self.engine.generate_package(self.narrative, self.scene_pkg, self.visual_pkg, self.voice_paths)
        with self.assertRaises(FrozenInstanceError):
            package.total_duration = 5.0  # type: ignore

    def test_timeline_generation_and_determinism(self) -> None:
        """Verify deterministic timeline creation, Ken Burns interpolation, and audio track scheduling."""
        pkg1 = self.engine.generate_package(self.narrative, self.scene_pkg, self.visual_pkg, self.voice_paths)
        pkg2 = self.engine.generate_package(self.narrative, self.scene_pkg, self.visual_pkg, self.voice_paths)

        # Assert full reproducibility (determinism)
        self.assertEqual(pkg1.total_duration, pkg2.total_duration)
        self.assertEqual(pkg1.ffmpeg_filter_graph, pkg2.ffmpeg_filter_graph)
        self.assertEqual(len(pkg1.clips), len(pkg2.clips))
        self.assertEqual(pkg1.clips[0].ken_burns_zoom_start, pkg2.clips[0].ken_burns_zoom_start)

        # Assert correct scheduling values
        self.assertEqual(pkg1.total_duration, 4.5)
        self.assertEqual(len(pkg1.clips), 1)
        self.assertEqual(pkg1.clips[0].clip_id, "clip_shot_1_1")
        self.assertEqual(pkg1.clips[0].transition_in, "fade")
        self.assertEqual(pkg1.clips[0].transition_out, "crossfade")

        # Subtitle timings
        self.assertTrue(len(pkg1.subtitles) > 0)
        self.assertEqual(pkg1.subtitles[0].start_time, 0.0)

    def test_adapter_compatibility(self) -> None:
        """Verify RenderPackageAdapter maps total duration and properties for legacy pipelines."""
        pkg = self.engine.generate_package(self.narrative, self.scene_pkg, self.visual_pkg, self.voice_paths)
        adapter = RenderPackageAdapter(pkg)
        self.assertEqual(adapter.total_duration, 4.5)
        self.assertEqual(adapter.resolution, self.settings.video_resolution)


if __name__ == "__main__":
    unittest.main()
