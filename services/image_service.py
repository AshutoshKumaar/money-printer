from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
import urllib.parse
from pathlib import Path

import requests
from google import genai
from google.genai import types
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
            
            selected_path, provider, confidence, selected_id, selected_url, used_query = self._select_visual(
                index,
                segment,
                plan,
                script.topic,
                output_path,
                use_existing=use_existing,
            )
            
            # Post-selection diversity score for logging
            diversity_score = self._calculate_diversity_score(index)
            
            # Log required by the user (Requirement 8)
            self.logger.info("--- Visual Diversity Log ---")
            self.logger.info("Scene Number: %s", index)
            self.logger.info("Query: %s", used_query)
            self.logger.info("Selected Image ID: %s", selected_id)
            self.logger.info("Selected Image URL: %s", selected_url)
            self.logger.info("Source (Pexels/Fallback): %s", "Pexels" if provider == "pexels" else "Fallback")
            self.logger.info("Visual Diversity Score: %.1f%%", diversity_score)
            self.logger.info("----------------------------")
            
            segment.visual_category = plan.category
            segment.visual_provider = provider
            segment.visual_confidence = confidence
            image_paths.append(selected_path)
            
            visual_manifest.append(
                {
                    "scene": index,
                    "provider": provider,
                    "path": str(selected_path),
                    "confidence": confidence,
                    "selected_id": selected_id,
                    "selected_url": selected_url,
                    "visual_plan": plan.to_dict(),
                    "diversity_score": diversity_score,
                }
            )

        manifest_path = paths.image_dir / "visual_manifest.json"
        manifest_path.write_text(json.dumps(visual_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return image_paths

    def _select_visual(
        self,
        index: int,
        segment: Segment,
        plan: VisualPlan,
        topic: str,
        output_path: Path,
        *,
        use_existing: bool,
    ) -> tuple[Path, str, float, str, str, str]:
        # Calculate visual diversity score (based on previous scenes)
        diversity_score = self._calculate_diversity_score(index - 1)
        # If diversity score drops below 80%, force a new search (Requirement 9)
        force_fresh = diversity_score < 80.0
        
        if force_fresh:
            self.logger.warning("Visual Diversity Score is %.1f%% (<80%%). Forcing fresh search!", diversity_score)

        if use_existing and not force_fresh:
            existing = self._cached_image(output_path)
            if existing:
                self._ensure_vertical(existing)
                if self._claim_unique(existing):
                    return existing, "existing", max(plan.confidence, 0.72), "N/A", "N/A", plan.query
                self.logger.warning("Existing visual was already used in this video; replacing it")

        # Generate exactly 3 alternative queries (Requirement 5)
        alts = self._generate_alternative_queries(plan, segment, topic)
        self.logger.info("Generated alternative queries: %s", alts)

        # 1. Pexels (Primary)
        try:
            self.logger.info("Attempting primary provider: Pexels")
            # Retry searches using alternative queries before using fallback (Requirement 6)
            photo_id, image_url, generated, used_q = self._pexels_image(
                plan, output_path, force_fresh=force_fresh, alternative_queries=alts
            )
            self._ensure_vertical(generated)
            if self._claim_unique(generated):
                cached = self._relevance_cache_path(plan)
                shutil.copy2(generated, cached)
                return generated, "pexels", min(0.96, plan.confidence + 0.12), str(photo_id), image_url, used_q
            else:
                self.logger.warning("Pexels returned a duplicate photo; trying next fallbacks")
        except Exception as exc:
            self.logger.warning("Pexels failed: %s", exc)

        # 2. Imagen 4 (Fallback if enabled)
        if self.settings.enable_ai_images and not force_fresh:
            if self._gemini_image_available:
                try:
                    self.logger.info("Attempting optional provider: Gemini Imagen 4")
                    generated = self._gemini_image(plan.prompt, output_path)
                    self._ensure_vertical(generated)
                    if self._claim_unique(generated):
                        cached = self._relevance_cache_path(plan)
                        shutil.copy2(generated, cached)
                        return generated, "gemini", min(0.98, plan.confidence + 0.08), "AI", "N/A", plan.query
                except Exception as exc:
                    if "NOT_FOUND" in str(exc) or "not supported" in str(exc):
                        self._gemini_image_available = False
                    self.logger.warning("Gemini visual failed: %s", exc)

        # 3. Local Cache (Fallback)
        if not force_fresh:
            cached = self._relevance_cache_path(plan)
            if cached.exists() and cached.stat().st_size > 0:
                try:
                    self.logger.info("Attempting fallback: Local Cache")
                    shutil.copy2(cached, output_path)
                    self._ensure_vertical(output_path)
                    if self._claim_unique(output_path):
                        return output_path, "cache", max(plan.confidence, 0.82), "N/A", "N/A", plan.query
                except Exception as exc:
                    self.logger.warning("Local Cache failed: %s", exc)

        # 4. Local Fallback (guaranteed unique) (Requirement 7)
        attempts = 0
        while attempts < 10:
            self._local_fallback_image(segment, plan, output_path, attempts)
            self._ensure_vertical(output_path)
            if self._claim_unique(output_path):
                break
            attempts += 1
            
        return output_path, f"local_{plan.category}", max(0.76, plan.confidence), "Fallback", "N/A", plan.query

    def _calculate_diversity_score(self, count: int) -> float:
        if count <= 0:
            return 100.0
        return (len(self._used_image_digests) / count) * 100.0

    def _clean_query(self, query: str) -> str:
        # Keep only English words (alphanumeric and dashes) and remove stopwords
        words = re.findall(r"[a-zA-Z0-9-]+", query)
        clean = []
        for w in words:
            lowered = w.lower()
            if lowered not in self.matcher.STOPWORDS and len(lowered) >= 3:
                clean.append(lowered)
        return " ".join(clean)

    def _generate_alternative_queries(self, plan: VisualPlan, segment: Segment, topic: str) -> list[str]:
        alts = []
        
        # Helper to clean and format a query string
        def sanitize(q: str) -> str:
            cleaned = self._clean_query(q)
            return cleaned.strip()

        # Alt 1: Cleaned version of segment.search_query
        alt1 = sanitize(segment.search_query)
        if alt1 and alt1 != plan.query:
            alts.append(alt1)

        # Alt 2: English keywords + category
        english_kws = [k for k in plan.keywords if re.fullmatch(r"[a-zA-Z0-9-]+", k)]
        if len(english_kws) > 2:
            alt2 = sanitize(f"{' '.join(english_kws[2:5])} {plan.category}")
        else:
            alt2 = sanitize(f"{' '.join(english_kws)} {plan.category}")
        if alt2 and alt2 != plan.query and alt2 not in alts:
            alts.append(alt2)

        # Alt 3: Cleaned topic + category default terms
        default_q = self.matcher.CATEGORY_RULES.get(plan.category, {}).get("query", "")
        alt3 = sanitize(f"{topic} {default_q}")
        if alt3 and alt3 != plan.query and alt3 not in alts:
            alts.append(alt3)

        # Fallback additions if we don't have 3 unique alternatives:
        fallbacks = [
            f"{plan.category} close up",
            f"cinematic {plan.category}",
            f"{topic} cinematic",
            default_q
        ]
        for fb in fallbacks:
            fb_cleaned = sanitize(fb)
            if len(alts) >= 3:
                break
            if fb_cleaned and fb_cleaned != plan.query and fb_cleaned not in alts:
                alts.append(fb_cleaned)

        # Ensure we always return exactly 3
        while len(alts) < 3:
            alts.append(sanitize(default_q))
            
        return alts[:3]

    def _pexels_search_and_select(self, query: str, plan: VisualPlan, output_path: Path) -> tuple[int, str, Path] | None:
        if not self.settings.pexels_api_key:
            self.logger.warning("PEXELS_API_KEY is missing")
            return None

        # Fetch at least 20 results per query (Requirement 3)
        per_page = max(20, self.settings.pexels_max_results)
        url = (
            "https://api.pexels.com/v1/search?"
            f"query={urllib.parse.quote(query)}&per_page={per_page}&orientation=portrait"
        )
        try:
            response = requests.get(
                url,
                headers={"Authorization": self.settings.pexels_api_key},
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
            photos = response.json().get("photos", [])
        except Exception as exc:
            self.logger.warning("Pexels query '%s' failed: %s", query, exc)
            return None

        if not photos:
            self.logger.warning("No Pexels results for query: %s", query)
            return None

        # Filter out already used photos (Requirement 1 & 4)
        available = [
            photo
            for photo in photos
            if int(photo.get("id", 0) or 0) not in self._used_pexels_ids
        ]
        
        if not available:
            self.logger.warning("Query '%s' returned only photos already used in this video", query)
            return None

        # Filter for relevance >= 1 (Requirement 4)
        relevant = []
        for photo in available:
            score = self._pexels_relevance(photo, plan)
            if score >= 1:
                relevant.append((photo, score))

        if not relevant:
            self.logger.warning("Query '%s' returned no photos with relevance >= 1", query)
            return None

        # Randomly select from the relevant unused results (Requirement 4)
        random.shuffle(relevant)
        for selected_photo, _ in relevant:
            photo_id = int(selected_photo.get("id", 0) or 0)
            src = selected_photo.get("src", {})
            image_url = src.get("portrait") or src.get("large2x") or src.get("large") or src.get("original")
            if not image_url:
                continue

            try:
                image_response = requests.get(image_url, timeout=self.settings.request_timeout_seconds)
                image_response.raise_for_status()
                
                # Check SHA256 of the content
                digest = hashlib.sha256(image_response.content).hexdigest()
                if digest in self._used_image_digests:
                    self.logger.warning("Pexels photo %s content hash already used, trying another photo", photo_id)
                    continue
                
                output_path.write_bytes(image_response.content)
                return photo_id, image_url, output_path
            except Exception as exc:
                self.logger.error("Failed to download image from Pexels URL %s: %s", image_url, exc)
                continue

        return None

    def _pexels_image(
        self,
        plan: VisualPlan,
        output_path: Path,
        force_fresh: bool = False,
        alternative_queries: list[str] = None
    ) -> tuple[int, str, Path, str]:
        if not self.settings.pexels_api_key:
            raise RuntimeError("PEXELS_API_KEY is missing")
            
        queries = [plan.query]
        if alternative_queries:
            queries.extend(alternative_queries)
            
        if force_fresh:
            # Mutate queries to force a new search (Requirement 9)
            category_kws = self.matcher.CATEGORY_RULES.get(plan.category, {}).get("keywords", [])
            mutated_queries = []
            for q in queries:
                if category_kws:
                    mutated_queries.append(f"{q} {random.choice(category_kws)}")
                else:
                    mutated_queries.append(f"{q} fresh")
            queries = mutated_queries

        for q in queries:
            result = self._pexels_search_and_select(q, plan, output_path)
            if result is not None:
                photo_id, image_url, path = result
                self._used_pexels_ids.add(photo_id)
                return photo_id, image_url, path, q
                
        raise RuntimeError(f"All Pexels search queries failed to find unused relevant images for category {plan.category}")

    def _pexels_relevance(self, photo: dict, plan: VisualPlan) -> int:
        text = " ".join(
            [
                str(photo.get("alt", "")),
                str(photo.get("photographer", "")),
            ]
        ).lower()
        category_terms = [plan.category, *plan.keywords[:6]]
        return sum(1 for term in category_terms if term.lower() in text)

    def _gemini_image(self, prompt: str, output_path: Path) -> Path:
        import time
        model_name = self.settings.gemini_image_model
        self.logger.info("Generating Gemini AI image using model: %s", model_name)
        start_time = time.time()
        try:
            result = self.client.models.generate_images(
                model=model_name,
                prompt=self._vertical_prompt(prompt),
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
            self.logger.info("Gemini AI image generated successfully. Latency: %.3fs", latency)
            return output_path
        except Exception as exc:
            latency = time.time() - start_time
            self.logger.error("Gemini AI image generation failed. Latency: %.3fs. Error: %s", latency, exc)
            raise

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

    def _pollinations_image(self, prompt: str, output_path: Path) -> Path:
        encoded_prompt = urllib.parse.quote(self._vertical_prompt(prompt))
        seed = random.randint(1, 999999)
        url = (
            f"{self.settings.pollinations_base_url}/{encoded_prompt}"
            f"?width={self.settings.video_resolution[0]}&height={self.settings.video_resolution[1]}"
            f"&nologo=true&seed={seed}"
        )
        response = requests.get(
            url,
            headers={"User-Agent": "HindiShortsAutomation/1.0"},
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return output_path

    def _local_fallback_image(self, segment: Segment, plan: VisualPlan, output_path: Path, attempt: int = 0) -> Path:
        width, height = self.settings.video_resolution
        palettes = {
            "everyday_science": ((14, 16, 18), (48, 52, 56), (255, 210, 72)),
            "space": ((3, 6, 20), (32, 25, 80), (92, 180, 255)),
            "ocean": ((1, 12, 28), (0, 54, 82), (70, 220, 240)),
            "history": ((25, 15, 8), (96, 60, 24), (235, 184, 92)),
            "person": ((12, 12, 16), (54, 46, 60), (255, 196, 150)),
            "technology": ((2, 12, 18), (0, 68, 72), (40, 255, 210)),
            "nature": ((5, 20, 12), (28, 74, 32), (130, 220, 90)),
            "horror": ((10, 4, 8), (50, 10, 24), (230, 40, 72)),
        }
        top, bottom, accent = palettes.get(plan.category, palettes["everyday_science"])
        image = Image.new("RGB", (width, height), top)
        draw = ImageDraw.Draw(image, "RGBA")

        for y in range(height):
            ratio = y / max(height - 1, 1)
            color = tuple(int(top[i] + (bottom[i] - top[i]) * ratio) for i in range(3))
            draw.line([(0, y), (width, y)], fill=(*color, 255))

        random.seed(f"{plan.cache_key()}|{segment.text}|{output_path.name}|{attempt}")
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

    def _draw_category_subject(
        self,
        draw: ImageDraw.ImageDraw,
        category: str,
        width: int,
        height: int,
        accent: tuple[int, int, int],
    ) -> None:
        if category == "everyday_science":
            table_y = int(height * 0.70)
            draw.rectangle([80, table_y, width - 80, table_y + 70], fill=(220, 220, 210, 120))
            draw.rounded_rectangle(
                [int(width * 0.24), int(height * 0.42), int(width * 0.76), int(height * 0.56)],
                radius=22,
                fill=(245, 245, 235, 210),
                outline=(*accent, 230),
                width=8,
            )
            for x in range(int(width * 0.30), int(width * 0.74), max(20, width // 24)):
                draw.line([(x, int(height * 0.44)), (x + 30, int(height * 0.54))], fill=(30, 35, 40, 170), width=5)
            center_x = width // 2
            center_y = int(height * 0.38)
            for radius in range(max(60, width // 10), max(220, width // 3), max(28, width // 28)):
                draw.ellipse(
                    [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                    outline=(*accent, 70),
                    width=5,
                )
        elif category == "space":
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
