from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import urllib.parse
from pathlib import Path

import requests
from google import genai
from PIL import Image, ImageDraw, ImageFilter

from config import Settings
from core.models import Script, Segment
from services.visual_matcher import VisualMatcher, VisualPlan
from storage import RunPaths


class ImageService:
    """Selects and creates narration-relevant visuals using a hybrid provider chain."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.matcher = VisualMatcher()
        self._hf_images_used = 0
        self._hf_disabled = False
        self._gemini_image_available = True
        self._used_image_digests: set[str] = set()
        self._used_pexels_ids: set[int] = set()
        self.settings.visual_cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_images(self, script: Script, paths: RunPaths, use_existing: bool = False) -> list[Path]:
        self._used_image_digests.clear()
        self._used_pexels_ids.clear()
        image_paths: list[Path] = []
        visual_manifest: list[dict] = []

        for index, segment in enumerate(script.segments, start=1):
            output_path = paths.image_dir / f"scene_{index:02d}.jpg"
            plan = self.matcher.plan(segment, script.topic)
            self.logger.info(
                "Visual plan %s/%s: category=%s confidence=%.2f query=%s",
                index,
                len(script.segments),
                plan.category,
                plan.confidence,
                plan.query,
            )

            selected_path, provider, confidence = self._select_visual(
                segment,
                plan,
                output_path,
                use_existing=use_existing,
            )
            segment.visual_category = plan.category
            segment.visual_provider = provider
            segment.visual_confidence = confidence
            self._ensure_vertical(selected_path)
            image_paths.append(selected_path)
            visual_manifest.append(
                {
                    "scene": index,
                    "provider": provider,
                    "path": str(selected_path),
                    "confidence": confidence,
                    "visual_plan": plan.to_dict(),
                }
            )

        manifest_path = paths.image_dir / "visual_manifest.json"
        manifest_path.write_text(json.dumps(visual_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return image_paths

    def _select_visual(
        self,
        segment: Segment,
        plan: VisualPlan,
        output_path: Path,
        *,
        use_existing: bool,
    ) -> tuple[Path, str, float]:
        if use_existing:
            existing = self._cached_image(output_path)
            if existing:
                self._ensure_vertical(existing)
                if self._claim_unique(existing):
                    return existing, "existing", max(plan.confidence, 0.72)
                self.logger.warning("Existing visual was already used in this video; replacing it")

        cached = self._relevance_cache_path(plan)
        if cached.exists() and cached.stat().st_size > 0:
            shutil.copy2(cached, output_path)
            self._ensure_vertical(output_path)
            if self._claim_unique(output_path):
                return output_path, "cache", max(plan.confidence, 0.82)
            self.logger.warning("Cached visual was already used in this video; fetching a unique replacement")

        attempts: list[tuple[str, float]] = []
        providers = self._provider_order()
        for provider in providers:
            try:
                if provider == "huggingface":
                    if self._hf_disabled or self._hf_images_used >= self.settings.hf_max_images_per_video:
                        continue
                    generated = self._huggingface_image(plan.prompt, output_path)
                    self._hf_images_used += 1
                    confidence = min(0.98, plan.confidence + 0.08)
                elif provider == "gemini":
                    if not self._gemini_image_available:
                        continue
                    generated = self._gemini_image(plan.prompt, output_path)
                    confidence = min(0.98, plan.confidence + 0.08)
                elif provider == "pexels":
                    generated = self._pexels_image(plan, output_path)
                    confidence = min(0.96, plan.confidence + 0.12)
                elif provider == "pollinations":
                    generated = self._pollinations_image(plan.prompt, output_path)
                    confidence = plan.confidence
                else:
                    continue

                if confidence < self.settings.visual_min_confidence:
                    attempts.append((provider, confidence))
                    self.logger.warning(
                        "Visual provider %s produced low confidence %.2f; trying a better match",
                        provider,
                        confidence,
                    )
                    continue

                self._ensure_vertical(generated)
                if not self._claim_unique(generated):
                    attempts.append((f"{provider}_duplicate", 0.0))
                    self.logger.warning(
                        "%s returned a visual already used in this video; trying another provider",
                        provider,
                    )
                    continue
                shutil.copy2(generated, cached)
                return generated, provider, confidence
            except Exception as exc:
                attempts.append((provider, 0.0))
                if provider == "huggingface" and ("402" in str(exc) or "depleted" in str(exc).lower()):
                    self._hf_disabled = True
                    self.logger.warning("Hugging Face credits unavailable; disabling HF for this run")
                elif provider == "gemini" and ("NOT_FOUND" in str(exc) or "not supported" in str(exc)):
                    self._gemini_image_available = False
                self.logger.warning("%s visual failed for category %s: %s", provider, plan.category, exc)

        local = self._local_fallback_image(segment, plan, output_path)
        local_confidence = max(0.76, plan.confidence)
        self._ensure_vertical(local)
        self._claim_unique(local)
        shutil.copy2(local, cached)
        self.logger.info(
            "Using relevant local fallback for %s after provider attempts: %s",
            plan.category,
            attempts,
        )
        return local, f"local_{plan.category}", local_confidence

    def _provider_order(self) -> list[str]:
        provider = self.settings.image_provider
        if provider == "hybrid":
            return ["huggingface", "pexels", "pollinations"]
        if provider == "huggingface":
            return ["huggingface", "pexels", "pollinations"]
        if provider == "gemini":
            return ["gemini", "pexels", "pollinations"]
        if provider == "pexels":
            return ["pexels", "pollinations"]
        return ["pollinations", "pexels"]

    def _gemini_image(self, prompt: str, output_path: Path) -> Path:
        response = self.client.models.generate_content(
            model=self.settings.gemini_image_model,
            contents=self._vertical_prompt(prompt),
        )
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None)
                if data:
                    output_path.write_bytes(data)
                    return output_path
        raise RuntimeError("Gemini image model returned no inline image")

    def _huggingface_image(self, prompt: str, output_path: Path) -> Path:
        if not self.settings.hf_token:
            raise RuntimeError("HF_TOKEN is missing")
        from huggingface_hub import InferenceClient

        kwargs = {"api_key": self.settings.hf_token}
        if self.settings.hf_provider:
            kwargs["provider"] = self.settings.hf_provider
        client = InferenceClient(**kwargs)
        image = client.text_to_image(
            self._vertical_prompt(prompt),
            model=self.settings.hf_image_model,
            width=self.settings.hf_image_width,
            height=self.settings.hf_image_height,
            negative_prompt=(
                "unrelated subject, random portrait, text, watermark, logo, subtitles, blurry, "
                "low quality, distorted, bad anatomy, cropped subject"
            ),
        )
        image.convert("RGB").save(output_path, quality=92)
        return output_path

    def _pexels_image(self, plan: VisualPlan, output_path: Path) -> Path:
        if not self.settings.pexels_api_key:
            raise RuntimeError("PEXELS_API_KEY is missing")
        url = (
            "https://api.pexels.com/v1/search?"
            f"query={urllib.parse.quote(plan.query)}&per_page={self.settings.pexels_max_results}&orientation=portrait"
        )
        response = requests.get(
            url,
            headers={"Authorization": self.settings.pexels_api_key},
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        photos = response.json().get("photos", [])
        if not photos:
            raise RuntimeError(f"No Pexels results for relevant query: {plan.query}")

        available = [
            photo
            for photo in photos
            if int(photo.get("id", 0) or 0) not in self._used_pexels_ids
        ]
        if not available:
            raise RuntimeError(f"Pexels returned only photos already used in this video: {plan.query}")

        ranked = sorted(
            available,
            key=lambda photo: (
                self._pexels_relevance(photo, plan),
                int(photo.get("height", 0) or 0),
            ),
            reverse=True,
        )
        best = ranked[0]
        score = self._pexels_relevance(best, plan)
        if score < 1:
            raise RuntimeError(f"Pexels results had low relevance for {plan.category}")
        photo_id = int(best.get("id", 0) or 0)
        if photo_id:
            self._used_pexels_ids.add(photo_id)
        src = best.get("src", {})
        image_url = src.get("portrait") or src.get("large2x") or src.get("large") or src.get("original")
        if not image_url:
            raise RuntimeError("Pexels result had no downloadable image")
        image_response = requests.get(image_url, timeout=self.settings.request_timeout_seconds)
        image_response.raise_for_status()
        output_path.write_bytes(image_response.content)
        return output_path

    def _pexels_relevance(self, photo: dict, plan: VisualPlan) -> int:
        text = " ".join(
            [
                str(photo.get("alt", "")),
                str(photo.get("photographer", "")),
            ]
        ).lower()
        category_terms = [plan.category, *plan.keywords[:6]]
        return sum(1 for term in category_terms if term.lower() in text)

    def _pollinations_image(self, prompt: str, output_path: Path) -> Path:
        encoded_prompt = urllib.parse.quote(self._vertical_prompt(prompt))
        seed = random.randint(1, 999999)
        url = (
            f"{self.settings.pollinations_base_url}/{encoded_prompt}"
            f"?width=1080&height=1920&nologo=true&seed={seed}"
        )
        response = requests.get(
            url,
            headers={"User-Agent": "HindiShortsAutomation/1.0"},
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return output_path

    def _local_fallback_image(self, segment: Segment, plan: VisualPlan, output_path: Path) -> Path:
        width, height = self.settings.video_resolution
        palettes = {
            "space": ((3, 6, 20), (32, 25, 80), (92, 180, 255)),
            "ocean": ((1, 12, 28), (0, 54, 82), (70, 220, 240)),
            "history": ((25, 15, 8), (96, 60, 24), (235, 184, 92)),
            "person": ((12, 12, 16), (54, 46, 60), (255, 196, 150)),
            "technology": ((2, 12, 18), (0, 68, 72), (40, 255, 210)),
            "nature": ((5, 20, 12), (28, 74, 32), (130, 220, 90)),
            "horror": ((10, 4, 8), (50, 10, 24), (230, 40, 72)),
        }
        top, bottom, accent = palettes.get(plan.category, palettes["horror"])
        image = Image.new("RGB", (width, height), top)
        draw = ImageDraw.Draw(image, "RGBA")

        for y in range(height):
            ratio = y / max(height - 1, 1)
            color = tuple(int(top[i] + (bottom[i] - top[i]) * ratio) for i in range(3))
            draw.line([(0, y), (width, y)], fill=(*color, 255))

        random.seed(f"{plan.cache_key()}|{segment.text}")
        self._draw_category_subject(draw, plan.category, width, height, accent)
        self._draw_particles(draw, plan.category, width, height, accent)

        glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow, "RGBA")
        center_x = width // 2 + random.randint(-120, 120)
        center_y = int(height * 0.42) + random.randint(-160, 160)
        for radius in range(480, 40, -24):
            alpha = int(70 * (radius / 480) ** 2)
            glow_draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                fill=(*accent, alpha),
            )
        glow = glow.filter(ImageFilter.GaussianBlur(38))
        image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")
        image.save(output_path, quality=92)
        return output_path

    def _draw_category_subject(
        self,
        draw: ImageDraw.ImageDraw,
        category: str,
        width: int,
        height: int,
        accent: tuple[int, int, int],
    ) -> None:
        if category == "space":
            draw.ellipse([170, 280, 900, 1010], fill=(*accent, 80), outline=(230, 240, 255, 170), width=8)
            draw.ellipse([305, 420, 765, 880], fill=(8, 12, 28, 255))
            draw.arc([120, 480, 960, 960], 195, 345, fill=(255, 220, 150, 180), width=22)
        elif category == "ocean":
            for offset in range(0, 600, 70):
                y = 280 + offset
                draw.arc([-100, y, width + 100, y + 240], 190, 350, fill=(*accent, 90), width=18)
            draw.polygon([(210, 1370), (540, 940), (870, 1370)], fill=(0, 8, 15, 210))
        elif category == "history":
            draw.rectangle([210, 730, 870, 1390], fill=(38, 22, 10, 180))
            for x in range(250, 850, 145):
                draw.rectangle([x, 560, x + 70, 1390], fill=(*accent, 120))
            draw.polygon([(170, 560), (540, 300), (910, 560)], fill=(*accent, 95))
        elif category == "technology":
            for x in range(140, width, 160):
                draw.line([(x, 280), (x, 1540)], fill=(*accent, 100), width=8)
            for y in range(360, 1540, 170):
                draw.line([(130, y), (950, y)], fill=(*accent, 90), width=8)
            draw.ellipse([310, 610, 770, 1070], outline=(*accent, 220), width=18)
        elif category == "nature":
            draw.polygon([(0, 1400), (330, 760), (560, 1300), (760, 620), (1080, 1400)], fill=(5, 35, 20, 230))
            draw.ellipse([620, 260, 900, 540], fill=(*accent, 130))
        elif category == "person":
            draw.ellipse([390, 430, 690, 730], fill=(*accent, 105))
            draw.rounded_rectangle([300, 700, 780, 1500], radius=170, fill=(5, 6, 10, 230))
        else:
            draw.polygon([(180, 1440), (540, 560), (900, 1440)], fill=(5, 2, 8, 220))
            draw.ellipse([430, 760, 650, 980], fill=(*accent, 120))

    def _draw_particles(
        self,
        draw: ImageDraw.ImageDraw,
        category: str,
        width: int,
        height: int,
        accent: tuple[int, int, int],
    ) -> None:
        count = 360 if category == "space" else 130
        for _ in range(count):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            radius = random.choice([1, 1, 2, 3])
            alpha = random.randint(70, 210)
            draw.ellipse([x, y, x + radius, y + radius], fill=(*accent, alpha))

    def _vertical_prompt(self, prompt: str) -> str:
        return (
            f"{prompt}. Vertical 9:16 composition, cinematic, realistic, high contrast, "
            "sharp subject, narration-relevant composition, no watermark, no subtitles, no UI text."
        )

    def _ensure_vertical(self, image_path: Path) -> None:
        with Image.open(image_path) as image:
            target_w, target_h = self.settings.video_resolution
            image = image.convert("RGB")
            source_w, source_h = image.size
            scale = max(target_w / source_w, target_h / source_h)
            resized = image.resize((int(source_w * scale), int(source_h * scale)), Image.LANCZOS)
            left = max(0, (resized.width - target_w) // 2)
            top = max(0, (resized.height - target_h) // 2)
            resized.crop((left, top, left + target_w, top + target_h)).save(image_path, quality=92)

    def _relevance_cache_path(self, plan: VisualPlan) -> Path:
        category_dir = self.settings.visual_cache_dir / plan.category
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir / f"{plan.cache_key()}.jpg"

    def _cached_image(self, preferred_path: Path) -> Path | None:
        if preferred_path.exists() and preferred_path.stat().st_size > 0:
            return preferred_path
        return None

    def _claim_unique(self, image_path: Path) -> bool:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in self._used_image_digests:
            return False
        self._used_image_digests.add(digest)
        return True
