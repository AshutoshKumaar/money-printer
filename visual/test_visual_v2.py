from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from config import load_settings
from scene.models import ScenePackage, Scene, Shot, CameraInstruction, TransitionInstruction, OverlayInstruction
from visual.models import VisualPackage, NewVisualAsset, AssetMetadata, AssetDecision, VisualPackageAdapter
from visual.visual_engine import VisualEngine
import logging
import tempfile
from pathlib import Path


class TestVisualV2(unittest.TestCase):
    """Unit tests for Visual Decision Engine V2, verifying source priority, reuse policy, and telemetry."""

    def setUp(self) -> None:
        self.settings = load_settings()
        self.logger = logging.getLogger("TestLogger")

        # Mock ScenePackage with multiple shots and continuity group
        self.scenes = [
            Scene(
                scene_id="scene_1",
                narration_segment_id=1,
                target_start=0.0,
                target_end=5.0,
                visual_type="stock_video",
                visual_priority="high",
                transition=TransitionInstruction("none"),
                overlay=OverlayInstruction("none"),
                shots=[
                    Shot(
                        shot_id="shot_1_1",
                        visual_goal="Ocean waves crashing on rocks",
                        camera_motion=CameraInstruction("static", "medium"),
                        duration=5.0,
                        transition_to_next=TransitionInstruction("none"),
                        visual_reference="unique_ocean_waves_test_key_xyz",
                        visual_source_strategy="stock_only",
                    )
                ],
                continuity_group="ocean_sequence",
            ),
            Scene(
                scene_id="scene_2",
                narration_segment_id=2,
                target_start=5.0,
                target_end=10.0,
                visual_type="stock_video",
                visual_priority="high",
                transition=TransitionInstruction("none"),
                overlay=OverlayInstruction("none"),
                shots=[
                    Shot(
                        shot_id="shot_2_1",
                        visual_goal="Different view of ocean waves on rocks",
                        camera_motion=CameraInstruction("static", "medium"),
                        duration=5.0,
                        transition_to_next=TransitionInstruction("none"),
                        visual_reference="unique_ocean_waves_test_key_xyz",  # Matches previous
                        visual_source_strategy="stock_only",
                    )
                ],
                continuity_group="ocean_sequence",  # Matches previous
            )
        ]
        self.package = ScenePackage(self.scenes, 10.0, 0.9, 0.9)


    def test_immutability(self) -> None:
        """Verify that VisualPackage and NewVisualAsset are immutable."""
        metadata = AssetMetadata()
        asset = NewVisualAsset(
            shot_id="shot_1",
            asset_id="asset_1",
            source="cache",
            confidence=0.9,
            cache_hit=True,
            continuity_group="default",
            quality_score=1.0,
            resolution="1080x1920",
            orientation="vertical",
            local_path="path/to/file",
            metadata=metadata,
        )
        with self.assertRaises(FrozenInstanceError):
            asset.confidence = 0.55  # type: ignore

    def test_cache_reuse_and_continuity_preservation(self) -> None:
        """Verify that visual resolution reuses assets from the same continuity group or cache."""
        engine = VisualEngine(self.settings, self.logger, providers=[])
        # Mock VisualCache.get to always return None to force a cache miss
        engine.cache.get = lambda key: None
        
        # Run visual engine to resolve package
        res = engine.resolve_assets(self.package)
        
        # Because providers is empty, it evaluates cache (miss), continuity (miss for first, hit for second), and fallbacks
        self.assertEqual(len(res.assets), 2)
        
        # Shot 1 should fall back to fallback (since no cache/stock/AI provider)
        a0 = res.assets[0]
        self.assertEqual(a0.shot_id, "shot_1_1")
        self.assertEqual(a0.source, "fallback")
        self.assertFalse(a0.cache_hit)

        # Shot 2 should hit continuity reuse on the same continuity group
        a1 = res.assets[1]
        self.assertEqual(a1.shot_id, "shot_2_1")
        self.assertEqual(a1.source, "existing")
        self.assertTrue(a1.cache_hit)
        self.assertEqual(a1.local_path, a0.local_path)
        self.assertEqual(a1.metadata.perceptual_hash, a0.metadata.perceptual_hash)

    def test_adapter_compatibility(self) -> None:
        """Verify that VisualPackageAdapter maps shot IDs to legacy scene indices correctly."""
        metadata = AssetMetadata()
        assets = [
            NewVisualAsset("shot_1_1", "asset_1", "ai", 0.9, False, "group1", 0.8, "1080x1920", "vertical", "path1.jpg", metadata),
            NewVisualAsset("shot_2_1", "asset_2", "stock", 0.85, True, "group1", 0.9, "1080x1920", "vertical", "path2.jpg", metadata),
        ]
        pkg = VisualPackage(assets, [], 10.0, 1, 0.5)
        adapter = VisualPackageAdapter(pkg)

        self.assertEqual(len(adapter.assets), 2)
        
        # Verify Shot 1 maps to scene_index 1
        s0 = adapter.assets[0]
        self.assertEqual(s0.scene_index, 1)
        self.assertEqual(s0.provider, "aiimage")
        self.assertEqual(s0.file_path, "path1.jpg")

        # Verify Shot 2 maps to scene_index 2
        s1 = adapter.assets[1]
        self.assertEqual(s1.scene_index, 2)
        self.assertEqual(s1.provider, "pexels")
        self.assertEqual(s1.file_path, "path2.jpg")


if __name__ == "__main__":
    unittest.main()
