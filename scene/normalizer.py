from __future__ import annotations

import json
import logging
import re
from typing import Any

# ==========================================
# Alias Dictionaries
# ==========================================

SHOT_TYPE_MAP = {
    "wide": "establishing",
    "wide shot": "establishing",
    "long shot": "establishing",
    "extreme wide": "establishing",
    "close": "close_up",
    "close shot": "close_up",
    "close up": "close_up",
    "cu": "close_up",
    "medium shot": "medium",
    "ms": "medium",
    "tracking shot": "medium",
    "pan shot": "medium",
    "zoom shot": "medium",
    "establishing": "establishing",
    "close_up": "close_up",
    "medium": "medium",
    "aerial": "aerial",
    "macro": "macro",
    "diagram": "diagram",
    "map": "map",
    "archive": "archive",
    "reconstruction": "reconstruction",
}

TRANSITION_MAP = {
    "fade in": "fade",
    "fade out": "fade",
    "fade": "fade",
    "hard cut": "none",
    "cut": "none",
    "crossfade": "crossfade",
    "cross fade": "crossfade",
    "dissolve": "dissolve",
    "zoom": "zoom",
    "wipe": "wipe",
    "slide": "slide",
    "none": "none",
}

CAMERA_MOTION_MAP = {
    "push in": "zoom_in",
    "pull out": "zoom_out",
    "camera zoom": "zoom_in",
    "zoom": "zoom_in",
    "zoom in": "zoom_in",
    "zoom out": "zoom_out",
    "tracking shot": "pan_left",
    "tracking": "pan_left",
    "pan shot": "pan_left",
    "pan": "pan_left",
    "pan left": "pan_left",
    "pan right": "pan_right",
    "pan up": "pan_up",
    "pan down": "pan_down",
    "tilt": "tilt",
    "static": "static",
}

OVERLAY_POSITION_MAP = {
    "upper": "top",
      "middle": "center",
      "lower": "bottom",
      "top": "top",
      "center": "center",
      "bottom": "bottom",
}

ANIMATION_MAP = {
    "slide in": "slide",
    "slide": "slide",
    "fade in": "fade",
    "fade": "fade",
    "pop in": "pop",
    "pop": "pop",
    "type writer": "typewriter",
    "typewriter": "typewriter",
    "none": "none",
}

VISUAL_PRIORITY_MAP = {
    "highest": "critical",
    "important": "high",
    "normal": "medium",
    "minor": "low",
    "critical": "critical",
    "high": "high",
    "hi gh": "high",
    "medium": "medium",
    "low": "low",
}

VISUAL_SOURCE_STRATEGY_MAP = {
    "stock only": "stock_only",
    "stock_only": "stock_only",
    "ai preferred": "ai_preferred",
    "ai_preferred": "ai_preferred",
    "ai required": "ai_required",
    "ai_required": "ai_required",
    "ai": "ai_preferred",
    "stock": "stock_only",
    "diagram": "diagram",
    "map": "map",
    "hybrid": "hybrid",
    "archival": "archival",
}


def clean_str(val: Any) -> str:
    if not isinstance(val, str):
        return ""
    # trim, lowercase, normalize underscores/hyphens/spaces to single space
    s = val.strip().lower()
    s = re.sub(r'[-_]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


class SceneNormalizer:
    """Deterministic enum normalization layer for cinematic scene plan JSON objects."""

    @staticmethod
    def normalize(scene_json: dict, logger: logging.Logger | None = None) -> dict:
        # Avoid mutating the original structure
        data = json.loads(json.dumps(scene_json))
        telemetry = {
            "shot_type": 0,
            "transition": 0,
            "camera_motion": 0,
            "overlay": 0,
            "priority": 0,
            "visual_source_strategy": 0,
        }

        def process(node: Any) -> Any:
            if isinstance(node, list):
                return [process(item) for item in node]
            if isinstance(node, dict):
                # Safe Defaults: Inject defaults if missing
                if "transition" in node:
                    trans_node = node["transition"]
                    if isinstance(trans_node, dict):
                        if "transition_type" not in trans_node or trans_node["transition_type"] is None:
                            trans_node["transition_type"] = "none"
                    elif isinstance(trans_node, str):
                        node["transition"] = {"transition_type": trans_node}

                if "overlay" in node:
                    overlay_node = node["overlay"]
                    if isinstance(overlay_node, dict):
                        if "position" not in overlay_node or overlay_node["position"] is None:
                            overlay_node["position"] = "bottom"
                        if "animation" not in overlay_node or overlay_node["animation"] is None:
                            overlay_node["animation"] = "fade"
                        if "style" not in overlay_node or overlay_node["style"] is None:
                            overlay_node["style"] = "default"
                    elif isinstance(overlay_node, str):
                        node["overlay"] = {
                            "overlay_type": overlay_node,
                            "position": "bottom",
                            "animation": "fade",
                            "style": "default",
                        }

                # Normalize keys/values recursively
                for k, v in list(node.items()):
                    if isinstance(v, str):
                        cleaned = clean_str(v)
                        if k == "shot_type" and cleaned in SHOT_TYPE_MAP:
                            canonical = SHOT_TYPE_MAP[cleaned]
                            if v != canonical:
                                if logger:
                                    logger.info('Normalized shot_type:\noriginal = "%s"\ncanonical = "%s"', v, canonical)
                                node[k] = canonical
                                telemetry["shot_type"] += 1
                        elif k == "visual_priority" and cleaned in VISUAL_PRIORITY_MAP:
                            canonical = VISUAL_PRIORITY_MAP[cleaned]
                            if v != canonical:
                                if logger:
                                    logger.info('Normalized visual_priority:\noriginal = "%s"\ncanonical = "%s"', v, canonical)
                                node[k] = canonical
                                telemetry["priority"] += 1
                        elif k == "visual_source_strategy" and cleaned in VISUAL_SOURCE_STRATEGY_MAP:
                            canonical = VISUAL_SOURCE_STRATEGY_MAP[cleaned]
                            if v != canonical:
                                if logger:
                                    logger.info('Normalized visual_source_strategy:\noriginal = "%s"\ncanonical = "%s"', v, canonical)
                                node[k] = canonical
                                telemetry["visual_source_strategy"] += 1
                        elif k == "transition_type" and cleaned in TRANSITION_MAP:
                            canonical = TRANSITION_MAP[cleaned]
                            if v != canonical:
                                if logger:
                                    logger.info('Normalized transition:\noriginal = "%s"\ncanonical = "%s"', v, canonical)
                                node[k] = canonical
                                telemetry["transition"] += 1
                        elif k == "motion_type" and cleaned in CAMERA_MOTION_MAP:
                            canonical = CAMERA_MOTION_MAP[cleaned]
                            if v != canonical:
                                if logger:
                                    logger.info('Normalized camera_motion:\noriginal = "%s"\ncanonical = "%s"', v, canonical)
                                node[k] = canonical
                                telemetry["camera_motion"] += 1
                        elif k == "position" and cleaned in OVERLAY_POSITION_MAP:
                            canonical = OVERLAY_POSITION_MAP[cleaned]
                            if v != canonical:
                                if logger:
                                    logger.info('Normalized overlay_position:\noriginal = "%s"\ncanonical = "%s"', v, canonical)
                                node[k] = canonical
                                telemetry["overlay"] += 1
                        elif k == "animation" and cleaned in ANIMATION_MAP:
                            canonical = ANIMATION_MAP[cleaned]
                            if v != canonical:
                                if logger:
                                    logger.info('Normalized animation:\noriginal = "%s"\ncanonical = "%s"', v, canonical)
                                node[k] = canonical
                                telemetry["overlay"] += 1

                    node[k] = process(node[k])

                # Safe Defaults: Speed in camera instruction
                if "camera_motion" in node:
                    cam_node = node["camera_motion"]
                    if isinstance(cam_node, dict):
                        if "speed" not in cam_node or cam_node["speed"] is None:
                            cam_node["speed"] = "medium"

            return node

        process(data)

        # Log Telemetry Summary
        active_fields = {k: v for k, v in telemetry.items() if v > 0}
        if active_fields and logger:
            summary_lines = ["", "Scene Normalizer Summary", "------------------------"]
            for k, v in active_fields.items():
                summary_lines.append(f"{k} normalized: {v}")
            logger.info("\n".join(summary_lines))

        data["_normalization_fixes"] = sum(telemetry.values())
        return data
