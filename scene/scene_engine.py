from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types

from config import Settings
from core.retry import retry_call
from story.models import NarrativeScript
from scene.models import ScenePlanManifest, SceneShot
from scene.prompts import SCENE_PLAN_PROMPT_TEMPLATE


class BaseScenePlannerProvider:
    """Base interface for cinematic scene planning providers to support future extensions."""

    def plan_scenes(self, script: NarrativeScript) -> ScenePlanManifest:
        raise NotImplementedError("Providers must implement plan_scenes.")


class GeminiScenePlannerProvider(BaseScenePlannerProvider):
    """Cinematic scene planner powered by Google Gemini API."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def plan_scenes(self, script: NarrativeScript) -> ScenePlanManifest:
        segments_list = []
        for segment in script.segments:
            segments_list.append(
                f"Segment {segment.index}:\n"
                f"- Narration: {segment.spoken_hindi}\n"
                f"- Subtitle: {getattr(segment, 'caption_keywords', getattr(segment, 'caption_hinglish', ''))}\n"
                f"- Concept: {segment.visual_concept}\n"
            )
        segments_block = "\n\n".join(segments_list)

        prompt = SCENE_PLAN_PROMPT_TEMPLATE.format(
            hook=script.hook,
            context=script.context,
            segments=segments_block,
            ending=script.ending,
            cta=script.cta,
        )

        import time
        from core.telemetry import telemetry_tracker

        attempt_tracker = {"count": 0}

        def run_call():
            attempt_tracker["count"] += 1
            t_start = time.time()
            try:
                res = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                latency = time.time() - t_start
                input_tokens = 0
                output_tokens = 0
                if res.usage_metadata:
                    input_tokens = res.usage_metadata.prompt_token_count or 0
                    output_tokens = res.usage_metadata.candidates_token_count or 0

                telemetry_tracker.record(
                    stage="scene",
                    provider="Google",
                    model="gemini-2.5-flash",
                    endpoint="models.generate_content",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attempt_number=attempt_tracker["count"],
                    retry_count=attempt_tracker["count"] - 1,
                    status_code=200,
                    latency=latency,
                    response_size_bytes=len(res.text or "") if res.text else 0,
                )
                return res
            except Exception as e:
                latency = time.time() - t_start
                telemetry_tracker.record(
                    stage="scene",
                    provider="Google",
                    model="gemini-2.5-flash",
                    endpoint="models.generate_content",
                    attempt_number=attempt_tracker["count"],
                    retry_count=attempt_tracker["count"] - 1,
                    status_code=getattr(e, "status_code", 500) or 500,
                    latency=latency,
                    response_size_bytes=0,
                )
                raise e

        response = retry_call(
            run_call,
            attempts=self.settings.retry_attempts,
            backoff_seconds=self.settings.retry_backoff_seconds,
            logger=self.logger,
            label="Gemini scene planning",
        )

        raw_text = response.text or ""
        cleaned = self._clean_json_text(raw_text)
        data = json.loads(cleaned)

        def make_shot(item: dict) -> SceneShot:
            ai_image_prompt = str(item.get("ai_image_prompt", "")).strip()
            stock_search_query = str(item.get("stock_search_query", item.get("search_query", ""))).strip()
            
            return SceneShot(
                scene_index=int(item.get("scene_index", 1)),
                shot_index=int(item.get("shot_index", 1)),
                duration_seconds=float(item.get("duration_seconds", 5.0) or 5.0),
                purpose=str(item.get("purpose", "")).strip(),
                visual_description=str(item.get("visual_description", "")).strip(),
                camera_angle=str(item.get("camera_angle", "Wide")).strip(),
                camera_motion=str(item.get("camera_motion", "Static")).strip(),
                lens_type=str(item.get("lens_type", "")).strip(),
                lighting=str(item.get("lighting", "")).strip(),
                environment=str(item.get("environment", "")).strip(),
                time_of_day=str(item.get("time_of_day", "")).strip(),
                color_palette=str(item.get("color_palette", "")).strip(),
                emotion=str(item.get("emotion", "curiosity")).strip(),
                transition_in=str(item.get("transition_in", "Cut")).strip(),
                transition_out=str(item.get("transition_out", "Cut")).strip(),
                caption_style=str(item.get("caption_style", "")).strip(),
                search_query=stock_search_query,
                ai_image_prompt=ai_image_prompt,
                stock_video_query=str(item.get("stock_video_query", "")).strip(),
                sound_effects=str(item.get("sound_effects", "")).strip(),
                background_music_mood=str(item.get("background_music_mood", "suspenseful")).strip(),
                priority=str(item.get("priority", "MEDIUM")).strip().upper(),
                stock_search_query=stock_search_query,
                cache_key="",
            )

        scenes = [make_shot(item) for item in data.get("scenes", []) if isinstance(item, dict)]

        return ScenePlanManifest(
            overall_style=str(data.get("overall_style", "Cinematic")).strip(),
            scene_count=int(data.get("scene_count", len(scenes))),
            estimated_runtime=float(data.get("estimated_runtime", sum(s.duration_seconds for s in scenes))),
            scenes=scenes,
        )

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text


class ScenePlanner:
    """Orchestrates cinematic planning from narrative story scripts."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        provider: BaseScenePlannerProvider | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.provider = provider or GeminiScenePlannerProvider(settings, logger)

    def plan(self, script: NarrativeScript) -> ScenePlanManifest:
        self.logger.info("Starting cinematic scene planning from script...")
        if not script.segments:
            self.logger.warning("Empty narrative script provided; returning a default scene plan manifest")
            return self._fallback_manifest()

        try:
            manifest = self.provider.plan_scenes(script)
        except Exception as exc:
            self.logger.error("Scene planning failed: %s. Returning fallback manifest.", exc)
            manifest = self._fallback_manifest()
            
        # Enforce strict 1-to-1 mapping by filtering and aligning scenes to segments
        unique_scenes = []
        seen_indices = set()
        for scene in manifest.scenes:
            s_idx = scene.scene_index
            if 1 <= s_idx <= len(script.segments):
                if s_idx not in seen_indices:
                    seen_indices.add(s_idx)
                    unique_scenes.append(scene)

        final_scenes = []
        for i in range(1, len(script.segments) + 1):
            existing = next((s for s in unique_scenes if s.scene_index == i), None)
            if existing:
                existing.shot_index = 1
                final_scenes.append(existing)
            else:
                seg = script.segments[i - 1]
                default_shot = SceneShot(
                    scene_index=i,
                    shot_index=1,
                    duration_seconds=5.0,
                    purpose=f"Scene {i} fallback",
                    visual_description=seg.visual_concept or "Cinematic scene",
                    camera_angle="Wide",
                    camera_motion="Static",
                    lens_type="35mm lens",
                    lighting="Cinematic lighting",
                    environment="Visual landscape",
                    time_of_day="Day",
                    color_palette="Natural colors",
                    emotion="curiosity",
                    transition_in="Cut",
                    transition_out="Cut",
                    caption_style="",
                    search_query=seg.search_query or "cinematic mystery landscape",
                    ai_image_prompt=f"vertical 9:16 cinematic {seg.visual_concept or 'desolate snowy landscape'}, movie quality, ultra realistic, professional cinematography, no text, no watermark",
                    stock_video_query=seg.search_query or "cinematic",
                    sound_effects="",
                    background_music_mood="suspenseful",
                    priority="MEDIUM",
                    stock_search_query=seg.search_query or "cinematic mystery landscape",
                    cache_key="",
                )
                final_scenes.append(default_shot)

        manifest.scenes = final_scenes
        manifest.scene_count = len(manifest.scenes)
        manifest.estimated_runtime = sum(s.duration_seconds for s in manifest.scenes)

        # Post-process priorities for all scenes in manifest
        for i, scene in enumerate(manifest.scenes):
            purpose_lower = scene.purpose.lower()
            desc_lower = scene.visual_description.lower()
            emotion_lower = scene.emotion.lower()
            
            # CRITICAL: Hook, Climax, Twist, Final reveal
            is_hook = (scene.scene_index == 1) or ("hook" in purpose_lower)
            is_climax_twist_reveal = any(word in purpose_lower or word in desc_lower for word in ["climax", "twist", "reveal", "discovery", "shocking"])
            is_final_reveal = (i == len(manifest.scenes) - 2 and len(manifest.scenes) > 3) or ("final" in purpose_lower)
            
            # HIGH: Character introduction, Important evidence, Emotional moments
            is_char_intro = any(word in purpose_lower or word in desc_lower for word in ["introduce", "introduction", "character", "protagonist", "person", "subject", "face", "portrait"])
            is_evidence = any(word in purpose_lower or word in desc_lower for word in ["evidence", "proof", "clue", "track", "document", "artifact", "photo"])
            is_emotional = any(word in emotion_lower or word in purpose_lower for word in ["fear", "surprise", "dread", "terror", "shock", "urgency", "emotion", "sadness", "anger"])
            
            # LOW: Transition, Atmosphere, Generic filler
            is_transition = "transition" in purpose_lower or "transition" in desc_lower
            is_atmosphere = "atmosphere" in purpose_lower or "atmosphere" in desc_lower or "ambient" in purpose_lower
            is_filler = "filler" in purpose_lower or "filler" in desc_lower or "generic" in purpose_lower or "cta" in purpose_lower or "call to action" in purpose_lower or i == len(manifest.scenes) - 1
            
            if is_hook or is_climax_twist_reveal or is_final_reveal:
                scene.priority = "CRITICAL"
            elif is_char_intro or is_evidence or is_emotional:
                scene.priority = "HIGH"
            elif is_transition or is_atmosphere or is_filler:
                scene.priority = "LOW"
            else:
                scene.priority = "MEDIUM"
                
        return manifest

    def _fallback_manifest(self) -> ScenePlanManifest:
        fallback_shot = SceneShot(
            scene_index=1,
            shot_index=1,
            duration_seconds=12.0,
            purpose="Fallback representation of narrative",
            visual_description="Cinematic stars swirling in deep space",
            camera_angle="Wide",
            camera_motion="Slow Zoom",
            lens_type="35mm anamorphic lens",
            lighting="Dramatic stars glow",
            environment="Deep space",
            time_of_day="Night",
            color_palette="Cool blues and stellar white",
            emotion="wonder",
            transition_in="Fade in",
            transition_out="Fade out",
            caption_style="Large highlighted overlay text",
            search_query="deep space stars cinematic spinning",
            ai_image_prompt="vertical 9:16 cinematic deep space stars, movie quality, ultra realistic, professional cinematography, no text, no watermark",
            stock_video_query="deep space stars",
            sound_effects="Cosmic rumble",
            background_music_mood="majestic",
            priority="CRITICAL",
            stock_search_query="deep space stars cinematic spinning",
            cache_key="cinematic deep space spinning stars",
        )
        return ScenePlanManifest(
            overall_style="Cinematic Cosmic Explainer",
            scene_count=1,
            estimated_runtime=12.0,
            scenes=[fallback_shot],
        )
