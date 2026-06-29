from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from config import load_settings
from analytics.importer import (
    YouTubeAnalyticsImporter,
    AnalyticsNormalizer,
    AnalyticsStore,
    AnalyticsSyncService,
)
import logging


class TestAnalyticsImporter(unittest.TestCase):
    """Unit tests for YouTube Analytics Importer, Normalizer, and Store."""

    def setUp(self) -> None:
        self.settings = load_settings()
        self.logger = logging.getLogger("TestLogger")
        self.temp_dir = TemporaryDirectory()
        self.history_dir = Path(self.temp_dir.name)
        self.history_path = self.history_dir / "analytics_history.json"
        
        # Override storage path settings
        object.__setattr__(self.settings, "storage_dir", self.history_dir)

        # Seed initial upload records
        self.seed_records = [
            {
                "run_id": "20260628-120000",
                "topic": "Mysteries of Mariana Trench",
                "category": "ocean",
                "upload_date": "2026-06-28T12:00:00Z",
                "status": "success",
                "views": None,
                "retention_rate": None,
                "engagement_rate": None,
                "youtube_url": "https://www.youtube.com/watch?v=abc123xyz99",
                "fingerprint": "mariana_fingerprint"
            }
        ]
        self.history_path.write_text(json.dumps(self.seed_records, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_api_parsing(self) -> None:
        """Verify video ID extraction from multiple YouTube URL formats."""
        urls = [
            "https://www.youtube.com/watch?v=abc123xyz99",
            "https://youtu.be/abc123xyz99",
            "https://www.youtube.com/watch?v=abc123xyz99&feature=shared",
        ]
        for url in urls:
            vid = YouTubeAnalyticsImporter.extract_video_id(url)
            self.assertEqual(vid, "abc123xyz99")

    def test_normalizer_and_derived_metrics(self) -> None:
        """Verify derived metrics calculation and value bounds validation."""
        raw_metrics = {
            "views": 100,
            "watch_time_sec": 300.0,
            "average_view_duration_sec": 3.0,
            "average_percentage_viewed": 50.0,
            "likes": 10,
            "comments": 2,
            "shares": 3,
            "impressions": 1000,
            "ctr": 5.0,
            "subscribers_gained": 1,
            "publish_time": "2026-06-28T12:00:00Z",
        }
        normalized = AnalyticsNormalizer.normalize(raw_metrics)
        
        # Test derived fields
        self.assertEqual(normalized["completion_rate"], 50.0)
        self.assertEqual(normalized["engagement_rate"], 15.0)  # (10+2+3)/100 * 100
        self.assertEqual(normalized["retention_score"], 0.5)
        self.assertTrue(normalized["velocity_score"] > 0)
        self.assertTrue(normalized["performance_score"] > 0)

    def test_missing_metrics_fallback(self) -> None:
        """Verify fallback handling when raw metrics are empty or missing."""
        empty_metrics = {}
        normalized = AnalyticsNormalizer.normalize(empty_metrics)
        self.assertEqual(normalized["views"], 0)
        self.assertEqual(normalized["likes"], 0)
        self.assertEqual(normalized["completion_rate"], 0.0)
        self.assertEqual(normalized["engagement_rate"], 0.0)

    def test_historical_preservation_and_duplicate_deduplication(self) -> None:
        """Verify append-only updates and historical record preservation."""
        store = AnalyticsStore(self.settings, self.logger)
        sync_service = AnalyticsSyncService(self.settings, self.logger)

        # 1. First synchronization pass
        sync_service.sync_all(simulate=True)
        
        # Verify first observation appended
        history = store.load_history()
        self.assertEqual(len(history), 2)  # Seed record + 1st Sync observation
        self.assertEqual(history[0]["views"], None)  # Seed record preserved!
        self.assertEqual(history[1]["views"], 1200)   # 1st Sync view value

        # 2. Second synchronization pass (simulating a subsequent periodic fetch)
        sync_service.sync_all(simulate=True)
        
        # Verify second observation appended without overwriting previous snapshots
        history2 = store.load_history()
        self.assertEqual(len(history2), 3)  # Seed + 1st Sync + 2nd Sync
        self.assertEqual(history2[0]["views"], None)
        self.assertEqual(history2[1]["views"], 1200)
        self.assertEqual(history2[2]["views"], 1200)
        self.assertTrue("sync_timestamp" in history2[1])
        self.assertTrue("sync_timestamp" in history2[2])


if __name__ == "__main__":
    unittest.main()
