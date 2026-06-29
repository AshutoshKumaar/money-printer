from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

from config import Settings
from research.models import ResearchPackage, ResearchPackageAdapter
from verification.models import VerifiedResearchPackage, VerificationAdapter
from story.models import NarrativePackage, NarrativeAdapter, NarrativeScript
from scene.models import (
    ScenePackage,
    Scene,
    Shot,
    CameraInstruction,
    TransitionInstruction,
    OverlayInstruction,
)
from scene.prompts import SCENE_PLAN_PROMPT_TEMPLATE


class BaseScenePlannerProvider:
    """Base interface for cinematic scene planning providers to support future extensions."""

    def plan_scenes(
        self,
        research: ResearchPackage,
        verified: VerifiedResearchPackage,
        narrative: NarrativePackage,
    ) -> ScenePackage:
        raise NotImplementedError("Providers must implement plan_scenes.")


class GeminiScenePlannerProvider(BaseScenePlannerProvider):
    """Cinematic scene planner powered by Google Gemini API, with JSON validation and retry loop."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def plan_scenes(
        self,
        research: ResearchPackage,
        verified: VerifiedResearchPackage,
        narrative: NarrativePackage,
    ) -> ScenePackage:
        # Format verified facts
        facts_list = []
        for idx, f in enumerate(verified.verified_facts, start=1):
            facts_list.append(f"Fact ID: fact_{idx} | claim: {f.fact} | category: {f.category} | importance: {f.importance_score}")
        facts_block = "\n".join(facts_list)

        # Format narrative segments
        segments_list = []
        for s in narrative.narration_segments:
            segments_list.append(
                f"Segment ID: seg_{s.index}\n"
                f"- Narration: {s.narration_text}\n"
                f"- Duration: {s.estimated_duration:.1f}s ({s.target_start:.1f}s to {s.target_end:.1f}s)\n"
                f"- Emotion: {s.emotion}\n"
                f"- Beat Type: {s.beat_type}\n"
                f"- Linked Facts: {s.verified_fact_ids}\n"
            )
        segments_block = "\n".join(segments_list)

        prompt = SCENE_PLAN_PROMPT_TEMPLATE.format(
            research_package=json.dumps(research.to_dict(), ensure_ascii=False),
            verified_facts=facts_block,
            narrative_package=segments_block,
        )

        from core.telemetry import telemetry_tracker

        attempts = self.settings.retry_attempts
        current_prompt = prompt
        last_exception = None

        for attempt in range(1, attempts + 1):
            t_start = time.time()
            try:
                self.logger.info("Calling Gemini scene planning attempt %d/%d...", attempt, attempts)
                res = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=current_prompt,
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
                    attempt_number=attempt,
                    retry_count=attempt - 1,
                    status_code=200,
                    latency=latency,
                    response_size_bytes=len(res.text or "") if res.text else 0,
                )

                raw_text = res.text or ""
                cleaned = self._clean_json_text(raw_text)
                data = json.loads(cleaned)

                # Validate JSON schema and value constraints
                self.validate_schema(data, narrative)

                # Parse and return ScenePackage
                return self.parse_scene_package(data)

            except Exception as exc:
                latency = time.time() - t_start
                telemetry_tracker.record(
                    stage="scene",
                    provider="Google",
                    model="gemini-2.5-flash",
                    endpoint="models.generate_content",
                    attempt_number=attempt,
                    retry_count=attempt - 1,
                    status_code=getattr(exc, "status_code", 500) or 500,
                    latency=latency,
                    response_size_bytes=0,
                )
                last_exception = exc
                self.logger.warning("Scene planning attempt %d failed: %s", attempt, exc)
                
                if attempt < attempts:
                    current_prompt = (
                        f"{prompt}\n\n"
                        f"WARNING: Your previous response failed parsing or validation with error: {exc}\n"
                        f"Please fix this error and output ONLY a valid JSON object matching the requested scene plan schema."
                    )
                    time.sleep(self.settings.retry_backoff_seconds * attempt)

        raise ValueError(f"Scene planning failed validation: {last_exception}") from last_exception

    def validate_schema(self, data: dict[str, Any], narrative: NarrativePackage) -> None:
        """Validate structure and alignment constraints of the scene plan payload."""
        required_keys = {"scenes", "estimated_total_duration", "pacing_score", "visual_variety_score"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"Missing required scene plan schema sections: {missing}")

        scenes = data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("scenes must be a non-empty list of scene records")

        valid_priorities = {"critical", "high", "medium", "low"}
        valid_motions = {"zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "tilt", "static"}
        valid_speeds = {"slow", "medium", "fast"}
        valid_transitions = {"fade", "crossfade", "dissolve", "zoom", "wipe", "slide", "none"}
        valid_overlays = {"text", "subtitle", "diagram", "map", "label", "none"}
        valid_positions = {"top", "center", "bottom"}
        valid_strategies = {"stock_only", "ai_preferred", "ai_required", "archival", "map", "diagram", "hybrid"}
        valid_shot_types = {"establishing", "close_up", "medium", "aerial", "macro", "diagram", "map", "archive", "reconstruction"}

        # Validate timing gaps & overlaps
        sorted_scenes = sorted(scenes, key=lambda s: float(s.get("target_start", 0.0) or 0.0))
        previous_end = 0.0

        for idx, s in enumerate(sorted_scenes):
            if not isinstance(s, dict):
                raise ValueError(f"Scene at index {idx} must be a JSON object")
            required_scene_keys = {
                "scene_id",
                "narration_segment_id",
                "target_start",
                "target_end",
                "visual_type",
                "visual_priority",
                "transition",
                "overlay",
                "shots",
                "continuity_group",
            }
            missing_scene = required_scene_keys - s.keys()
            if missing_scene:
                raise ValueError(f"Scene at index {idx} missing keys: {missing_scene}")

            t_start = float(s.get("target_start", 0.0) or 0.0)
            t_end = float(s.get("target_end", 0.0) or 0.0)
            scene_dur = t_end - t_start

            # Check overlaps/gaps (allow small floating point tolerance of 0.15s)
            if idx > 0 and abs(t_start - previous_end) > 0.15:
                raise ValueError(f"Scene sequence timing gap or overlap detected at target_start {t_start}s (previous end was {previous_end}s)")
            previous_end = t_end

            # Priority check
            priority = s.get("visual_priority")
            if priority not in valid_priorities:
                raise ValueError(f"Scene {s.get('scene_id')} has invalid visual_priority: '{priority}'")

            # Transition check
            trans = s.get("transition", {})
            t_type = trans.get("transition_type")
            if t_type not in valid_transitions:
                raise ValueError(f"Scene {s.get('scene_id')} has invalid transition type: '{t_type}'")

            # Overlay check
            overlay = s.get("overlay", {})
            o_type = overlay.get("overlay_type")
            if o_type not in valid_overlays:
                raise ValueError(f"Scene {s.get('scene_id')} has invalid overlay type: '{o_type}'")
            o_pos = overlay.get("position", "center")
            if o_pos not in valid_positions:
                raise ValueError(f"Scene {s.get('scene_id')} has invalid overlay position: '{o_pos}'")

            # Shots list check
            shots = s.get("shots")
            if not isinstance(shots, list) or not shots:
                raise ValueError(f"Scene {s.get('scene_id')} shots must be a non-empty list")

            # Shot duration alignment check
            shots_dur = 0.0
            for sh_idx, shot in enumerate(shots):
                if not isinstance(shot, dict):
                    raise ValueError(f"Shot at index {sh_idx} in scene {s.get('scene_id')} must be a JSON object")
                required_shot_keys = {
                    "shot_id",
                    "visual_goal",
                    "camera_motion",
                    "duration",
                    "transition_to_next",
                    "visual_reference",
                    "visual_source_strategy",
                }
                missing_shot = required_shot_keys - shot.keys()
                if missing_shot:
                    raise ValueError(f"Shot at index {sh_idx} in scene {s.get('scene_id')} missing keys: {missing_shot}")

                shots_dur += float(shot.get("duration", 0.0) or 0.0)

                # Camera motion check
                cam = shot.get("camera_motion", {})
                m_type = cam.get("motion_type")
                if m_type not in valid_motions:
                    raise ValueError(f"Shot {shot.get('shot_id')} has invalid camera motion type: '{m_type}'")
                speed = cam.get("speed")
                if speed not in valid_speeds:
                    raise ValueError(f"Shot {shot.get('shot_id')} has invalid camera speed: '{speed}'")

                # Transition next check
                trans_next = shot.get("transition_to_next", {})
                tn_type = trans_next.get("transition_type")
                if tn_type not in valid_transitions:
                    raise ValueError(f"Shot {shot.get('shot_id')} has invalid transition_to_next: '{tn_type}'")

                # Visual strategy check
                strat = shot.get("visual_source_strategy")
                if strat not in valid_strategies:
                    raise ValueError(f"Shot {shot.get('shot_id')} has invalid visual_source_strategy: '{strat}'")

                # Shot type check (can be null/none)
                st = shot.get("shot_type")
                if st is not None and st not in valid_shot_types:
                    raise ValueError(f"Shot {shot.get('shot_id')} has invalid shot_type: '{st}'")

            # Tolerance check for shot duration alignment (0.15s)
            if abs(scene_dur - shots_dur) > 0.15:
                raise ValueError(f"Scene {s.get('scene_id')} duration ({scene_dur:.2f}s) does not match sum of shot durations ({shots_dur:.2f}s)")

    def parse_scene_package(self, data: dict[str, Any]) -> ScenePackage:
        """Parse structured scene plan dict into ScenePackage dataclass."""
        scenes = []
        for s in data.get("scenes", []):
            trans_data = s.get("transition", {})
            transition = TransitionInstruction(transition_type=str(trans_data.get("transition_type", "none")))

            overlay_data = s.get("overlay", {})
            overlay = OverlayInstruction(
                overlay_type=str(overlay_data.get("overlay_type", "none")),
                text=overlay_data.get("text") or None if overlay_data.get("text") else None,
                position=str(overlay_data.get("position", "center")),
                style=str(overlay_data.get("style", "default")),
                animation=str(overlay_data.get("animation", "none")),
                duration=float(overlay_data.get("duration", 0.0) or 0.0),
            )

            shots = []
            for shot in s.get("shots", []):
                cam_data = shot.get("camera_motion", {})
                camera_motion = CameraInstruction(
                    motion_type=str(cam_data.get("motion_type", "static")),
                    speed=str(cam_data.get("speed", "medium")),
                )

                trans_next_data = shot.get("transition_to_next", {})
                transition_to_next = TransitionInstruction(
                    transition_type=str(trans_next_data.get("transition_type", "none"))
                )

                shots.append(
                    Shot(
                        shot_id=str(shot.get("shot_id", "")).strip(),
                        visual_goal=str(shot.get("visual_goal", "")).strip(),
                        camera_motion=camera_motion,
                        duration=float(shot.get("duration", 0.0) or 0.0),
                        transition_to_next=transition_to_next,
                        visual_reference=shot.get("visual_reference") or None if shot.get("visual_reference") else None,
                        visual_source_strategy=str(shot.get("visual_source_strategy", "stock_only")),
                        shot_type=shot.get("shot_type") or None if shot.get("shot_type") else None,
                        aspect_ratio_hint=str(shot.get("aspect_ratio_hint", "9:16")),
                        safe_crop_region=shot.get("safe_crop_region") or None,
                        focus_subject=shot.get("focus_subject") or None if shot.get("focus_subject") else None,
                    )
                )

            scenes.append(
                Scene(
                    scene_id=str(s.get("scene_id", "")).strip(),
                    narration_segment_id=int(s.get("narration_segment_id", 1)),
                    target_start=float(s.get("target_start", 0.0) or 0.0),
                    target_end=float(s.get("target_end", 0.0) or 0.0),
                    visual_type=str(s.get("visual_type", "stock_video")),
                    visual_priority=str(s.get("visual_priority", "medium")),
                    transition=transition,
                    overlay=overlay,
                    shots=shots,
                    continuity_group=str(s.get("continuity_group", "default")).strip(),
                )
            )

        return ScenePackage(
            scenes=scenes,
            estimated_total_duration=float(data.get("estimated_total_duration", 0.0) or 0.0),
            pacing_score=float(data.get("pacing_score", 0.0) or 0.0),
            visual_variety_score=float(data.get("visual_variety_score", 0.0) or 0.0),
        )

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text


class ScenePlanner:
    """Orchestrates cinematic scene plans from narrative structures and verified research databases."""

    def __init__(self, settings: Settings, logger: logging.Logger, provider: BaseScenePlannerProvider | None = None) -> None:
        self.settings = settings
        self.logger = logger
        self.provider = provider or GeminiScenePlannerProvider(settings, logger)

    def plan(
        self,
        script: NarrativeScript | NarrativeAdapter,
        research: ResearchPackage | ResearchPackageAdapter | None = None,
        verified: VerifiedResearchPackage | VerificationAdapter | None = None,
    ) -> ScenePackage:
        self.logger.info("Starting cinematic scene planning (V2 Engine)...")

        # Unwrap packages
        if isinstance(script, NarrativeAdapter):
            narrative_pkg = script._package
        else:
            narrative_pkg = self._dummy_narrative_from_script(script)

        if research is None:
            research_pkg = self._dummy_research(narrative_pkg)
        elif isinstance(research, ResearchPackageAdapter):
            research_pkg = research._package
        else:
            research_pkg = research

        if verified is None:
            verified_pkg = self._dummy_verified(research_pkg)
        elif isinstance(verified, VerificationAdapter):
            verified_pkg = verified._package
        else:
            verified_pkg = verified

        try:
            return self.provider.plan_scenes(research_pkg, verified_pkg, narrative_pkg)
        except Exception as exc:
            self.logger.error("Scene planning V2 failed completely: %s. Constructing fallback package.", exc)
            
            # Simple fallback scene plan matching narrative segments
            fallback_scenes = []
            for s in narrative_pkg.narration_segments:
                scene_id = f"scene_{s.index}"
                fallback_scenes.append(
                    Scene(
                        scene_id=scene_id,
                        narration_segment_id=s.index,
                        target_start=s.target_start,
                        target_end=s.target_end,
                        visual_type="stock_video",
                        visual_priority="medium",
                        transition=TransitionInstruction(transition_type="none"),
                        overlay=OverlayInstruction(overlay_type="none", text=None),
                        shots=[
                            Shot(
                                shot_id=f"shot_{s.index}_1",
                                visual_goal=f"Cinematic video representing narration: {s.narration_text[:40]}",
                                camera_motion=CameraInstruction(motion_type="static", speed="medium"),
                                duration=s.estimated_duration,
                                transition_to_next=TransitionInstruction(transition_type="none"),
                                visual_reference=None,
                                visual_source_strategy="stock_only",
                            )
                        ],
                        continuity_group="default",
                    )
                )

            return ScenePackage(
                scenes=fallback_scenes,
                estimated_total_duration=narrative_pkg.estimated_total_duration if hasattr(narrative_pkg, "estimated_total_duration") else sum(s.estimated_duration for s in narrative_pkg.narration_segments),
                pacing_score=0.5,
                visual_variety_score=0.5,
            )

    def _dummy_narrative_from_script(self, script: NarrativeScript) -> NarrativePackage:
        """Construct a compatibility NarrativePackage from a legacy NarrativeScript."""
        segments = []
        current_time = 0.0
        for seg in script.segments:
            # Approximate duration based on word count (legacy fallback)
            dur = seg.duration_seconds if hasattr(seg, "duration_seconds") else float(len(seg.spoken_hindi.split()) * 0.4)
            if dur < 1.0:
                dur = 4.0
            segments.append(
                NarrationSegment(
                    index=seg.index,
                    narration_text=seg.spoken_hindi,
                    estimated_duration=dur,
                    target_start=current_time,
                    target_end=current_time + dur,
                    emotion="neutral",
                    purpose="Legacy segment",
                    verified_fact_ids=[],
                    beat_type="evidence",
                )
            )
            current_time += dur

        return NarrativePackage(
            language="hi",
            hook=script.hook,
            context=script.context,
            escalation="",
            climax="",
            ending=script.ending,
            narration_segments=segments,
            quality=NarrativeQuality(
                retention_score=script.retention_score,
                pacing_score=0.8,
                curiosity_score=script.retention_score,
                clarity_score=0.8,
                emotional_score=0.8,
                estimated_retention_curve=[1.0, script.retention_score],
            ),
        )

    def _dummy_research(self, narrative: NarrativePackage) -> ResearchPackage:
        """Construct a fallback empty ResearchPackage."""
        from research.models import (
            HistoricalContext,
            ScientificContext,
            ImportantEntities,
            StoryOpportunities,
            SEOResearch,
            ResearchConfidence,
        )
        return ResearchPackage(
            topic="Fallback Topic",
            topic_summary="Fallback Summary",
            historical_context=HistoricalContext([], ""),
            scientific_context=ScientificContext("", []),
            important_entities=ImportantEntities([], [], [], [], [], []),
            verified_facts=[],
            common_misconceptions=[],
            unanswered_questions=[],
            visual_opportunities=[],
            story_opportunities=StoryOpportunities("", "", [], "", ""),
            seo_research=SEOResearch([], [], []),
            research_confidence=ResearchConfidence(0.0, [], "moderate"),
        )

    def _dummy_verified(self, research: ResearchPackage) -> VerifiedResearchPackage:
        """Construct a fallback empty VerifiedResearchPackage."""
        return VerifiedResearchPackage(
            research_package=research,
            verified_facts=[],
            verification_warnings=[],
            overall_confidence_score=0.0,
            verification_quality_score=0.0,
        )
