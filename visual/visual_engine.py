from __future__ import annotations

import logging
import time
import re
from pathlib import Path
from typing import Any

from config import Settings
from scene.models import ScenePackage, ScenePackageAdapter, Scene, Shot
from visual.cache import VisualCache
from visual.image_quality import ImageQualityChecker
from visual.models import (
    VisualPackage,
    NewVisualAsset,
    AssetMetadata,
    AssetDecision,
)
from visual.providers import (
    AIImageProvider,
    BaseVisualProvider,
    PexelsProvider,
    PixabayProvider,
)


class ShotProviderWrapper:
    """Wraps a granular Shot in a legacy-compatible interface for visual providers."""

    def __init__(self, shot: Shot, scene: Scene, index: int) -> None:
        self.scene_index = index
        self.ai_image_prompt = shot.visual_goal
        self.stock_search_query = shot.visual_reference or shot.visual_goal
        self.search_query = self.stock_search_query
        self.priority = scene.visual_priority.upper()


class VisualEngine:
    """Orchestrates structured visual asset search, generation, and quality control."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        providers: list[BaseVisualProvider] | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.cache = VisualCache(settings.visual_cache_dir)
        self.checker = ImageQualityChecker(logger)

        if providers is not None:
            self.providers = providers
        else:
            self.providers = []
            if settings.enable_ai_images:
                self.providers.append(AIImageProvider(settings, logger))
            self.providers.append(PexelsProvider(settings, logger))
            self.providers.append(PixabayProvider(settings, logger))

    def resolve_assets(self, scene_plan: ScenePackage | ScenePackageAdapter) -> VisualPackage:
        self.logger.info("Resolving visual assets for scene plan...")

        # Extract/unwrap ScenePackage
        if isinstance(scene_plan, ScenePackageAdapter):
            package = scene_plan._package
        elif hasattr(scene_plan, "_package"):
            package = scene_plan._package
        else:
            package = scene_plan

        resolved_assets: list[NewVisualAsset] = []
        decisions: list[AssetDecision] = []

        # Trackers for reuse policy and budget
        ai_generation_count = 0
        used_urls = set()
        used_paths = set()
        continuity_group_assets: dict[str, NewVisualAsset] = {}
        total_latency = 0.0

        ai_provider = next((p for p in self.providers if p.__class__.__name__ == "AIImageProvider"), None)
        pexels_provider = next((p for p in self.providers if p.__class__.__name__ == "PexelsProvider"), None)
        pixabay_provider = next((p for p in self.providers if p.__class__.__name__ == "PixabayProvider"), None)

        shot_count = 0
        for s_idx, scene in enumerate(package.scenes, start=1):
            for sh_idx, shot in enumerate(scene.shots, start=1):
                shot_count += 1
                t_start = time.time()
                fallback_triggered = False
                retry_count = 0
                retry_occurred = False

                # Generate clean cache key
                search_query = shot.visual_reference or shot.visual_goal
                clean = re.sub(r'[^\w\s]', ' ', search_query.lower())
                words = clean.split()
                stopwords = {"a", "an", "the", "at", "in", "on", "of", "and", "or", "for", "with", "by", "to", "from", "is", "was", "were", "are"}
                filtered = [w for w in words if w not in stopwords]
                filtered.sort()
                cache_key = " ".join(filtered)

                # Prioritization Sequence
                selected_source = None
                local_path = ""
                cache_hit = False
                quality_score = 0.0
                confidence = 0.0
                reasoning = ""
                evaluated_sources = []

                # Wrapper for legacy provider calls
                wrapper = ShotProviderWrapper(shot, scene, s_idx)

                # 1. Evaluate Local Cache (First priority)
                evaluated_sources.append("cache")
                cached_path = self.cache.get(cache_key)
                if cached_path and str(cached_path) not in used_paths:
                    selected_source = "cache"
                    local_path = str(cached_path)
                    cache_hit = True
                    confidence = 0.99
                    quality_score = 1.0
                    reasoning = f"Cache hit for key: '{cache_key}'"
                    used_paths.add(local_path)
                    self.logger.info("Shot %s: Cache hit! Reusing %s", shot.shot_id, cached_path)
                    try:
                        from core.telemetry import telemetry_tracker
                        telemetry_tracker.record_fallback("cached_image")
                    except Exception:
                        pass

                # 2. Evaluate Existing Generated Assets/Continuity Group Reuse (Second priority)
                if not selected_source and scene.continuity_group != "default":
                    evaluated_sources.append("existing_continuity")
                    if scene.continuity_group in continuity_group_assets:
                        existing = continuity_group_assets[scene.continuity_group]
                        selected_source = "existing"
                        local_path = existing.local_path
                        cache_hit = True
                        confidence = existing.confidence
                        quality_score = existing.quality_score
                        reasoning = f"Reused asset from continuity group: '{scene.continuity_group}'"
                        used_paths.add(local_path)
                        self.logger.info("Shot %s: Continuity group '%s' reuse hit! Reusing %s", shot.shot_id, scene.continuity_group, local_path)
                        try:
                            from core.telemetry import telemetry_tracker
                            telemetry_tracker.record_fallback("cached_image")
                        except Exception:
                            pass

                # 3. Evaluate Stock Providers (Third priority)
                if not selected_source:
                    evaluated_sources.append("stock")
                    run_temp_dir = self.settings.image_dir / "run_temp"
                    run_temp_dir.mkdir(parents=True, exist_ok=True)
                    output_path = run_temp_dir / f"shot_{shot.shot_id}.jpg"

                    stock_providers = [p for p in [pexels_provider, pixabay_provider] if p is not None]
                    for prov in stock_providers:
                        prov_name = prov.__class__.__name__.replace("Provider", "").lower()
                        self.logger.info("Shot %s: Attempting stock provider %s", shot.shot_id, prov_name)
                        try:
                            res = prov.acquire(wrapper, output_path, exclude_urls=used_urls)
                            if res:
                                asset_type, conf, latency = res
                                quality_res = self.checker.evaluate(str(output_path))
                                if quality_res["is_valid"]:
                                    selected_source = "stock"
                                    local_path = str(self.cache.store(cache_key, output_path))
                                    confidence = conf
                                    quality_score = float(quality_res["overall_quality_score"])
                                    reasoning = f"Successfully resolved high-quality stock asset from {prov_name}"
                                    used_paths.add(local_path)
                                    if getattr(prov, "last_acquired_url", None):
                                        used_urls.add(prov.last_acquired_url)
                                    break
                                else:
                                    self.logger.warning("Shot %s: Rejecting low-quality stock asset", shot.shot_id)
                                    if output_path.exists():
                                        output_path.unlink()
                        except Exception as e:
                            self.logger.warning("Shot %s: Stock provider failed: %s", shot.shot_id, e)
                            try:
                                from core.telemetry import telemetry_tracker
                                telemetry_tracker.record_retry("Visual", str(e), recovered=False, fallback=False)
                                retry_occurred = True
                            except Exception:
                                pass

                # 4. Evaluate AI Image Generation (Fourth priority)
                if not selected_source and ai_provider:
                    evaluated_sources.append("ai")
                    output_path = run_temp_dir / f"shot_{shot.shot_id}.jpg"
                    self.logger.info("Shot %s: Attempting AI image generation", shot.shot_id)
                    try:
                        res = ai_provider.acquire(wrapper, output_path)
                        if res:
                            asset_type, conf, latency = res
                            quality_res = self.checker.evaluate(str(output_path))
                            if quality_res["is_valid"]:
                                selected_source = "ai"
                                local_path = str(self.cache.store(cache_key, output_path))
                                confidence = conf
                                quality_score = float(quality_res["overall_quality_score"])
                                reasoning = "Successfully generated cinematic AI image"
                                used_paths.add(local_path)
                                ai_generation_count += 1
                            else:
                                self.logger.warning("Shot %s: Rejecting low-quality AI asset", shot.shot_id)
                                if output_path.exists():
                                    output_path.unlink()
                        else:
                            try:
                                from core.telemetry import telemetry_tracker
                                telemetry_tracker.record_retry("Visual", "AI provider returned no result", recovered=False, fallback=False)
                                retry_occurred = True
                            except Exception:
                                pass
                    except Exception as e:
                        self.logger.warning("Shot %s: AI generation failed: %s", shot.shot_id, e)
                        try:
                            from core.telemetry import telemetry_tracker
                            telemetry_tracker.record_retry("Visual", str(e), recovered=False, fallback=False)
                            retry_occurred = True
                        except Exception:
                            pass

                # 5. Placeholder Fallback (Fifth priority)
                if not selected_source:
                    evaluated_sources.append("fallback")
                    fallback_triggered = True
                    selected_source = "fallback"
                    # Generate a basic placeholder file
                    output_path = run_temp_dir / f"shot_{shot.shot_id}.jpg"
                    # Write tiny dummy JPEG bytes
                    output_path.write_bytes(
                        b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xFF\xDB\x00C\x00\x08\x06\x06'
                        b'\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
                        b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xFF\xC0\x00\x0b\x08\x00'
                        b'\x01\x00\x01\x01\x01\x11\x00\xFF\xC4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
                        b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xFF\xDA\x00\x08\x01\x01\x00'
                        b'\x00\x3F\x00\xB7\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                        b'\xFF\xD9'
                    )
                    local_path = str(output_path)
                    confidence = 0.1
                    quality_score = 0.5
                    reasoning = "All visual resolution strategies failed. Falling back to dummy placeholder."
                    used_paths.add(local_path)
                    try:
                        from core.telemetry import telemetry_tracker
                        telemetry_tracker.record_fallback("placeholder_image")
                        if retry_occurred:
                            telemetry_tracker.retries["Visual"]["fallback"] = True
                    except Exception:
                        pass

                if retry_occurred and selected_source in ("stock", "ai"):
                    try:
                        from core.telemetry import telemetry_tracker
                        telemetry_tracker.retries["Visual"]["recovered"] = True
                    except Exception:
                        pass

                decision_time_ms = (time.time() - t_start) * 1000.0
                total_latency += decision_time_ms

                # Dominant subjects extraction from shot details
                subjects = []
                if shot.focus_subject:
                    subjects.append(shot.focus_subject)

                metadata = AssetMetadata(
                    perceptual_hash=f"phash_{cache_key}",
                    dominant_colors=["#000000"],
                    detected_subjects=subjects,
                    creation_timestamp=time.time(),
                    provider_metadata={"cache_key": cache_key, "shot_type": shot.shot_type},
                )

                resolved = NewVisualAsset(
                    shot_id=shot.shot_id,
                    asset_id=f"asset_{shot.shot_id}",
                    source=selected_source,
                    confidence=confidence,
                    cache_hit=cache_hit,
                    continuity_group=scene.continuity_group,
                    quality_score=quality_score,
                    resolution="1080x1920",
                    orientation="vertical",
                    local_path=local_path,
                    metadata=metadata,
                )
                resolved_assets.append(resolved)

                # Track for continuity reuse
                if scene.continuity_group != "default" and scene.continuity_group not in continuity_group_assets:
                    continuity_group_assets[scene.continuity_group] = resolved

                decisions.append(
                    AssetDecision(
                        shot_id=shot.shot_id,
                        evaluated_sources=evaluated_sources,
                        selected_source=selected_source,
                        reasoning=reasoning,
                        decision_time_ms=decision_time_ms,
                        fallback_triggered=fallback_triggered,
                        retry_count=retry_count,
                    )
                )

        # Cache efficiency
        cache_hits = sum(1 for a in resolved_assets if a.cache_hit)
        efficiency = (cache_hits / len(resolved_assets)) if resolved_assets else 0.0

        return VisualPackage(
            assets=resolved_assets,
            decisions=decisions,
            total_duration=sum(shot.duration for scene in package.scenes for shot in scene.shots),
            ai_generation_count=ai_generation_count,
            cache_efficiency=efficiency,
        )
