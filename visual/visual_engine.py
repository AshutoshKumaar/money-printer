from __future__ import annotations

import logging
import time
from pathlib import Path

from config import Settings
from scene.models import ScenePlanManifest, SceneShot
from visual.cache import VisualCache
from visual.image_quality import ImageQualityChecker
from visual.models import VisualAsset, VisualAssetManifest
from visual.providers import (
    AIImageProvider,
    BaseVisualProvider,
    PexelsProvider,
    PixabayProvider,
)


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
        
        if providers:
            self.providers = providers
        else:
            self.providers = []
            if settings.enable_ai_images:
                self.providers.append(AIImageProvider(settings, logger))
            self.providers.append(PexelsProvider(settings, logger))
            self.providers.append(PixabayProvider(settings, logger))

    def resolve_assets(self, scene_plan: ScenePlanManifest) -> VisualAssetManifest:
        self.logger.info("Resolving visual assets for scene plan containing %s scenes", scene_plan.scene_count)
        
        assets: list[VisualAsset] = []
        ai_images_used = 0
        used_urls = set()
        used_paths = set()
        
        # Determine cost optimization thresholds
        is_short = scene_plan.estimated_runtime <= 60.0
        limit = self.settings.ai_image_limit_per_short if is_short else self.settings.ai_image_limit_per_long
        max_allowed_ai = int(scene_plan.scene_count * (self.settings.max_ai_percentage / 100.0))
        effective_limit = min(limit, max_allowed_ai)
        
        ai_provider = next((p for p in self.providers if p.__class__.__name__ == "AIImageProvider"), None)
        pexels_provider = next((p for p in self.providers if p.__class__.__name__ == "PexelsProvider"), None)
        pixabay_provider = next((p for p in self.providers if p.__class__.__name__ == "PixabayProvider"), None)
        
        previous_cache_key = None
        previous_prompt = None
        
        for scene in scene_plan.scenes:
            priority = getattr(scene, "priority", "MEDIUM").upper()
            ai_prompt = scene.ai_image_prompt or scene.visual_description
            stock_query = scene.stock_search_query
            
            # Clean and normalize the cache key deterministically
            import re
            clean = re.sub(r'[^\w\s]', ' ', stock_query.lower())
            words = clean.split()
            stopwords = {"a", "an", "the", "at", "in", "on", "of", "and", "or", "for", "with", "by", "to", "from", "is", "was", "were", "are"}
            filtered = [w for w in words if w not in stopwords]
            filtered.sort()
            cache_key = " ".join(filtered)
            
            # Prevent consecutive identical prompts/cache keys by adding variation
            if cache_key == previous_cache_key:
                cache_key = f"{cache_key} variation {scene.scene_index}"
                ai_prompt = f"{ai_prompt}, variation {scene.scene_index}"
                stock_query = f"{stock_query} variation {scene.scene_index}"
                scene.cache_key = cache_key
                scene.ai_image_prompt = ai_prompt
                scene.stock_search_query = stock_query
            else:
                scene.cache_key = cache_key
                
            # 1. Try cache first for non-CRITICAL scenes
            if priority != "CRITICAL":
                cached_path = self.cache.get(cache_key)
                if cached_path and str(cached_path) not in used_paths:
                    self.logger.info("Cache hit for scene %s! Reusing %s", scene.scene_index, cached_path)
                    used_paths.add(str(cached_path))
                    assets.append(
                        VisualAsset(
                            scene_index=scene.scene_index,
                            provider="cache",
                            asset_type="image",
                            prompt=ai_prompt,
                            file_path=str(cached_path),
                            quality_score=1.0,
                            confidence=0.99,
                            cache_hit=True,
                            generation_time=0.0,
                            cache_key=cache_key,
                        )
                    )
                    from core.telemetry import telemetry_tracker
                    telemetry_tracker.record(
                        stage="visual",
                        provider="cache",
                        model="cache",
                        endpoint="cache",
                        cache_hit=True,
                        scene_index=scene.scene_index,
                    )
                    previous_cache_key = cache_key
                    previous_prompt = ai_prompt
                    continue
            
            # 2. Cache missed or CRITICAL
            run_temp_dir = self.settings.image_dir / "run_temp"
            run_temp_dir.mkdir(parents=True, exist_ok=True)
            output_path = run_temp_dir / f"scene_{scene.scene_index:02d}.jpg"
            
            resolved_asset = None
            
            # Determine provider sequence based on priority policy & budget
            providers_to_try = []
            
            if priority == "CRITICAL":
                providers_to_try = [ai_provider, pexels_provider, pixabay_provider]
            elif priority == "HIGH":
                # AI if budget remains, else fallback to Cache/Stock
                if ai_images_used < effective_limit:
                    providers_to_try = [ai_provider, pexels_provider, pixabay_provider]
                else:
                    self.logger.info("AI image budget exhausted. Falling back to stock for HIGH priority scene %s", scene.scene_index)
                    providers_to_try = [pexels_provider, pixabay_provider, ai_provider]
            elif priority == "MEDIUM":
                providers_to_try = [pexels_provider, pixabay_provider, ai_provider]
            elif priority == "LOW":
                providers_to_try = [pexels_provider, pixabay_provider]
                
            # Filter out None values
            providers_to_try = [p for p in providers_to_try if p is not None]
            
            for provider in providers_to_try:
                provider_name = provider.__class__.__name__.replace("Provider", "").lower()
                self.logger.info("Attempting provider: %s for scene %s (%s)", provider_name, scene.scene_index, priority)
                
                try:
                    if provider_name in ["pexels", "pixabay"]:
                        res = provider.acquire(scene, output_path, exclude_urls=used_urls)
                    else:
                        res = provider.acquire(scene, output_path)
                except Exception as e:
                    self.logger.warning("Provider %s failed during acquire: %s", provider_name, e)
                    res = None
                    
                if res is None:
                    continue
                    
                asset_type, confidence, latency = res
                
                # Check quality
                quality_res = self.checker.evaluate(str(output_path))
                if quality_res["is_valid"]:
                    quality_score = float(quality_res["overall_quality_score"])
                    self.logger.info(
                        "Asset resolved and verified! Provider: %s, Score: %s",
                        provider_name, quality_score
                    )
                    
                    cached_file = self.cache.store(cache_key, output_path)
                    used_paths.add(str(cached_file))
                    
                    resolved_asset = VisualAsset(
                        scene_index=scene.scene_index,
                        provider=provider_name,
                        asset_type=asset_type,
                        prompt=ai_prompt if provider_name == "aiimage" else stock_query,
                        file_path=str(cached_file),
                        quality_score=quality_score,
                        confidence=confidence,
                        cache_hit=False,
                        generation_time=round(latency, 2),
                        cache_key=cache_key,
                    )
                    
                    # Update trackers
                    if provider_name == "aiimage":
                        ai_images_used += 1
                    elif provider_name in ["pexels", "pixabay"]:
                        if getattr(provider, "last_acquired_url", None):
                            used_urls.add(provider.last_acquired_url)
                            
                    break
                else:
                    self.logger.warning(
                        "Quality check rejected image from provider %s. Continuing fallback...",
                        provider_name
                    )
                    if output_path.exists():
                        output_path.unlink()
                        
            if resolved_asset is not None:
                assets.append(resolved_asset)
                previous_cache_key = cache_key
                previous_prompt = ai_prompt
            else:
                raise RuntimeError(
                    f"All visual providers failed to resolve a high-quality asset for scene {scene.scene_index}."
                )
                
        return VisualAssetManifest(assets=assets)
