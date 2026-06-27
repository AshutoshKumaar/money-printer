from __future__ import annotations

import logging
import os
import random
import time
import requests
import urllib.parse
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image

from config import Settings
from core.retry import retry_call
from scene.models import SceneShot


class BaseVisualProvider:
    """Base interface for all visual asset providers."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.last_acquired_url: str | None = None

    def acquire(self, scene: SceneShot, output_path: Path, exclude_urls: set[str] | None = None) -> tuple[str, float, float] | None:
        """
        Attempts to acquire visual asset for the scene.
        Returns tuple: (asset_type, confidence, latency) or None if fails.
        """
        raise NotImplementedError


class AIImageProvider(BaseVisualProvider):
    """Google Gemini Imagen visual asset provider."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        super().__init__(settings, logger)
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def acquire(self, scene: SceneShot, output_path: Path) -> tuple[str, float, float] | None:
        if not self.settings.gemini_api_key:
            self.logger.warning("Gemini API key is not configured for AI Image Provider")
            return None
        
        prompt = scene.ai_image_prompt or scene.visual_description
        model_name = self.settings.gemini_image_model
        
        vertical_prompt = (
            f"{prompt}. Vertical 9:16 composition, cinematic, realistic, high contrast, "
            "sharp subject, no watermark, no subtitles, no UI text, photorealistic, "
            "professional cinematography, movie quality."
        )

        self.logger.info("Generating Gemini AI image using model: %s", model_name)
        from core.telemetry import telemetry_tracker
        start_time = time.time()
        try:
            result = self.client.models.generate_images(
                model=model_name,
                prompt=vertical_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio="9:16",
                )
            )
            latency = time.time() - start_time
            if not result.generated_images:
                raise RuntimeError("No images returned from Gemini API")
            
            image_bytes = result.generated_images[0].image.image_bytes
            output_path.write_bytes(image_bytes)
            self._ensure_vertical(output_path)
            
            telemetry_tracker.record(
                stage="visual",
                provider="Google",
                model=model_name,
                endpoint="models.generate_images",
                images_requested=1,
                images_returned=1,
                status_code=200,
                latency=latency,
                response_size_bytes=len(image_bytes),
                scene_index=scene.scene_index,
            )
            
            return "image", 0.95, latency
        except Exception as exc:
            latency = time.time() - start_time
            telemetry_tracker.record(
                stage="visual",
                provider="Google",
                model=model_name,
                endpoint="models.generate_images",
                images_requested=1,
                images_returned=0,
                status_code=getattr(exc, "status_code", 500) or 500,
                latency=latency,
                scene_index=scene.scene_index,
            )
            self.logger.error("Gemini AI image generation failed: %s (latency: %.2fs)", exc, latency)
            return None

    def _ensure_vertical(self, image_path: Path) -> None:
        with Image.open(image_path) as image:
            target_w, target_h = self.settings.video_resolution
            image = image.convert("RGB")
            source_w, source_h = image.size
            scale = max(target_w / source_w, target_h / source_h)
            resized = image.resize((int(source_w * scale), int(source_h * scale)), Image.Resampling.LANCZOS)
            left = max(0, (resized.width - target_w) // 2)
            top = max(0, (resized.height - target_h) // 2)
            resized.crop((left, top, left + target_w, top + target_h)).save(image_path, quality=92)


class PexelsProvider(BaseVisualProvider):
    """Pexels stock image asset provider."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        super().__init__(settings, logger)

    def acquire(self, scene: SceneShot, output_path: Path, exclude_urls: set[str] | None = None) -> tuple[str, float, float] | None:
        api_key = self.settings.pexels_api_key
        if not api_key:
            self.logger.warning("Pexels API key is not configured")
            return None

        query = getattr(scene, "stock_search_query", None) or scene.search_query or scene.visual_description
        self.logger.info("Searching Pexels for query: %s", query)
        start_time = time.time()
        
        url = (
            "https://api.pexels.com/v1/search?"
            f"query={urllib.parse.quote(query)}&per_page=12&orientation=portrait"
        )
        from core.telemetry import telemetry_tracker
        try:
            response = requests.get(
                url,
                headers={"Authorization": api_key},
                timeout=self.settings.request_timeout_seconds,
            )
            
            # Record Pexels search API call
            telemetry_tracker.record(
                stage="visual",
                provider="Pexels",
                model="pexels-search",
                endpoint="/v1/search",
                images_requested=1,
                images_returned=len(response.json().get("photos", [])) if response.ok else 0,
                status_code=response.status_code,
                latency=time.time() - start_time,
                response_size_bytes=len(response.content) if response.ok else 0,
                scene_index=scene.scene_index,
            )
            
            response.raise_for_status()
            photos = response.json().get("photos", [])
            valid_photos = photos
            if exclude_urls:
                valid_photos = []
                for p in photos:
                    src = p.get("src", {})
                    url = src.get("portrait") or src.get("large2x") or src.get("large")
                    if url not in exclude_urls:
                        valid_photos.append(p)
                if not valid_photos:
                    self.logger.warning("All Pexels photos filtered by exclude_urls; returning None to fallback.")
                    return None

            if not valid_photos:
                self.logger.warning("No Pexels photos found for query: %s", query)
                return None
                
            photo = random.choice(valid_photos)
            src = photo.get("src", {})
            image_url = src.get("portrait") or src.get("large2x") or src.get("large")
            if not image_url:
                return None
            self.last_acquired_url = image_url

            t_download = time.time()
            img_res = requests.get(image_url, timeout=self.settings.request_timeout_seconds)
            
            # Record Pexels image download API call
            telemetry_tracker.record(
                stage="visual",
                provider="Pexels",
                model="pexels-download",
                endpoint="image-download",
                images_requested=1,
                images_returned=1 if img_res.ok else 0,
                status_code=img_res.status_code,
                latency=time.time() - t_download,
                response_size_bytes=len(img_res.content) if img_res.ok else 0,
                scene_index=scene.scene_index,
            )
            
            img_res.raise_for_status()
            
            output_path.write_bytes(img_res.content)
            self._ensure_vertical(output_path)
            
            latency = time.time() - start_time
            return "image", 0.80, latency
        except Exception as exc:
            latency = time.time() - start_time
            telemetry_tracker.record(
                stage="visual",
                provider="Pexels",
                model="pexels-search",
                endpoint="/v1/search",
                images_requested=1,
                images_returned=0,
                status_code=500,
                latency=latency,
                scene_index=scene.scene_index,
            )
            self.logger.error("Pexels download failed: %s (latency: %.2fs)", exc, latency)
            return None

    def _ensure_vertical(self, image_path: Path) -> None:
        with Image.open(image_path) as image:
            target_w, target_h = self.settings.video_resolution
            image = image.convert("RGB")
            source_w, source_h = image.size
            scale = max(target_w / source_w, target_h / source_h)
            resized = image.resize((int(source_w * scale), int(source_h * scale)), Image.Resampling.LANCZOS)
            left = max(0, (resized.width - target_w) // 2)
            top = max(0, (resized.height - target_h) // 2)
            resized.crop((left, top, left + target_w, top + target_h)).save(image_path, quality=92)


class PixabayProvider(BaseVisualProvider):
    """Pixabay stock image asset provider."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        super().__init__(settings, logger)

    def acquire(self, scene: SceneShot, output_path: Path, exclude_urls: set[str] | None = None) -> tuple[str, float, float] | None:
        api_key = os.getenv("PIXABAY_API_KEY")
        if not api_key:
            self.logger.warning("PIXABAY_API_KEY is not configured in environment")
            return None

        query = getattr(scene, "stock_search_query", None) or scene.search_query or scene.visual_description
        self.logger.info("Searching Pixabay for query: %s", query)
        start_time = time.time()
        
        url = (
            "https://pixabay.com/api/?"
            f"key={api_key}&q={urllib.parse.quote(query)}&image_type=photo&orientation=vertical&per_page=10"
        )
        from core.telemetry import telemetry_tracker
        try:
            response = requests.get(url, timeout=self.settings.request_timeout_seconds)
            
            # Record Pixabay search API call
            telemetry_tracker.record(
                stage="visual",
                provider="Pixabay",
                model="pixabay-search",
                endpoint="/api/",
                images_requested=1,
                images_returned=len(response.json().get("hits", [])) if response.ok else 0,
                status_code=response.status_code,
                latency=time.time() - start_time,
                response_size_bytes=len(response.content) if response.ok else 0,
                scene_index=scene.scene_index,
            )
            
            response.raise_for_status()
            hits = response.json().get("hits", [])
            valid_hits = hits
            if exclude_urls:
                valid_hits = []
                for h in hits:
                    url = h.get("largeImageURL") or h.get("webformatURL")
                    if url not in exclude_urls:
                        valid_hits.append(h)
                if not valid_hits:
                    self.logger.warning("All Pixabay hits filtered by exclude_urls; returning None to fallback.")
                    return None

            if not valid_hits:
                self.logger.warning("No Pixabay hits found for query: %s", query)
                return None
                
            selected = random.choice(valid_hits)
            image_url = selected.get("largeImageURL") or selected.get("webformatURL")
            if not image_url:
                return None
            self.last_acquired_url = image_url

            t_download = time.time()
            img_res = requests.get(image_url, timeout=self.settings.request_timeout_seconds)
            
            # Record Pixabay image download API call
            telemetry_tracker.record(
                stage="visual",
                provider="Pixabay",
                model="pixabay-download",
                endpoint="image-download",
                images_requested=1,
                images_returned=1 if img_res.ok else 0,
                status_code=img_res.status_code,
                latency=time.time() - t_download,
                response_size_bytes=len(img_res.content) if img_res.ok else 0,
                scene_index=scene.scene_index,
            )
            
            img_res.raise_for_status()
            
            output_path.write_bytes(img_res.content)
            self._ensure_vertical(output_path)
            
            latency = time.time() - start_time
            return "image", 0.75, latency
        except Exception as exc:
            latency = time.time() - start_time
            telemetry_tracker.record(
                stage="visual",
                provider="Pixabay",
                model="pixabay-search",
                endpoint="/api/",
                images_requested=1,
                images_returned=0,
                status_code=500,
                latency=latency,
                scene_index=scene.scene_index,
            )
            self.logger.error("Pixabay download failed: %s (latency: %.2fs)", exc, latency)
            return None

    def _ensure_vertical(self, image_path: Path) -> None:
        with Image.open(image_path) as image:
            target_w, target_h = self.settings.video_resolution
            image = image.convert("RGB")
            source_w, source_h = image.size
            scale = max(target_w / source_w, target_h / source_h)
            resized = image.resize((int(source_w * scale), int(source_h * scale)), Image.Resampling.LANCZOS)
            left = max(0, (resized.width - target_w) // 2)
            top = max(0, (resized.height - target_h) // 2)
            resized.crop((left, top, left + target_w, top + target_h)).save(image_path, quality=92)
