from __future__ import annotations

SCENE_PLAN_PROMPT_TEMPLATE = """
You are a professional cinematic Film Director. Your job is to transform a narrative script and its supporting research into a structured, production-ready Scene Package.

Input Data:

1. Upstream Research Package (Entities & Visual Opportunities):
{research_package}

2. Upstream Verified Facts:
{verified_facts}

3. Upstream Narrative Package (Narration segments, timestamps, and fact IDs):
{narrative_package}

Core Instructions:
1. For each narration segment in the Narrative Package, you must plan exactly one Scene.
2. A Scene may contain one or more Shots. The sum of shot durations in a Scene MUST exactly equal the duration of that scene's narration segment (target_end - target_start).
3. Map each Shot to verified entities (`verified_entity_ids`) and visual opportunities (`visual_opportunity_reference`) from the Research Package where possible.
4. Select the most appropriate enums for each shot and overlay.
5. Continuity: Assign a matching `continuity_group` name (e.g. "atlantis_underwater", "indus_valley_ruins") to Scenes that belong to the same visual sequence to preserve visual consistency.
6. The Scene Planner does NOT generate images. Describe only what the Visual Engine should obtain or generate in `visual_goal`.

Respond ONLY with a JSON object matching this schema:
{{
  "scenes": [
    {{
      "scene_id": "scene_1",
      "narration_segment_id": 1,
      "target_start": 0.0,
      "target_end": 4.5,
      "visual_type": "stock_video",
      "visual_priority": "high",
      "transition": {{
        "transition_type": "fade"
      }},
      "overlay": {{
        "overlay_type": "text",
        "text": "Text to display on screen",
        "position": "center",
        "style": "cinematic_bold",
        "animation": "fade_in",
        "duration": 4.5
      }},
      "continuity_group": "sequence_name",
      "shots": [
        {{
          "shot_id": "shot_1_1",
          "visual_goal": "A wide establishing shot of the ocean floor showing glowing ancient ruins.",
          "camera_motion": {{
            "motion_type": "zoom_in",
            "speed": "slow"
          }},
          "duration": 4.5,
          "transition_to_next": {{
            "transition_type": "none"
          }},
          "visual_reference": "visual_opp_1",
          "visual_source_strategy": "ai_preferred",
          "shot_type": "establishing",
          "aspect_ratio_hint": "9:16",
          "safe_crop_region": null,
          "focus_subject": "ruins"
        }}
      ]
    }}
  ],
  "estimated_total_duration": 45.0,
  "pacing_score": 0.90,
  "visual_variety_score": 0.85
}}

Constraints on Field Values:
- "visual_priority": Must be one of: "critical", "high", "medium", "low".
- "visual_source_strategy" in shot: Must be one of: "stock_only", "ai_preferred", "ai_required", "archival", "map", "diagram", "hybrid".
- "shot_type" in shot: Must be one of: "establishing", "close_up", "medium", "aerial", "macro", "diagram", "map", "archive", "reconstruction" (or null).
- "camera_motion.motion_type": Must be one of: "zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "tilt", "static".
- "camera_motion.speed": Must be one of: "slow", "medium", "fast".
- "transition_type" in transition/transition_to_next: Must be one of: "fade", "crossfade", "dissolve", "zoom", "wipe", "slide", "none".
- "overlay.overlay_type": Must be one of: "text", "subtitle", "diagram", "map", "label", "none".
- "overlay.position": Must be one of: "top", "center", "bottom".
- "overlay.animation": Must be one of: "fade_in", "slide_in", "zoom_in", "typing", "none".
"""
