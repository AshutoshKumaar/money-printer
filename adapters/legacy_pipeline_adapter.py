from __future__ import annotations

from pathlib import Path
from core.models import Script, Segment
from story.models import NarrativeScript
from scene.models import ScenePlanManifest
from visual.models import VisualAssetManifest


class LegacyPipelineAdapter:
    """Adapts new modular engine outputs to legacy dataclass contracts without mutating input objects."""

    @staticmethod
    def adapt_script(
        topic: str,
        narrative: NarrativeScript,
        scene_plan: ScenePlanManifest,
        visual_assets: VisualAssetManifest | None = None,
    ) -> Script:
        def keep_only_devanagari(text: str) -> str:
            cleaned = []
            for char in text:
                if '\u0900' <= char <= '\u097F' or char.isspace() or char.isdigit():
                    cleaned.append(char)
            import re
            res = "".join(cleaned)
            res = re.sub(r"\s+", " ", res).strip()
            return res

        legacy_segments = []
        for scene in scene_plan.scenes:
            # Find matching story segment by index (both 1-indexed)
            matched_segment = next(
                (seg for seg in narrative.segments if seg.index == scene.scene_index),
                None
            )
            if matched_segment is None:
                # Fallback to index-based lookup
                if 0 <= scene.scene_index - 1 < len(narrative.segments):
                    matched_segment = narrative.segments[scene.scene_index - 1]

            spoken_text = ""
            subtitle_text = ""
            if matched_segment:
                spoken_text = (matched_segment.spoken_hindi or "").strip()
                subtitle_text = (getattr(matched_segment, "subtitle_text", "") or spoken_text).strip()

            if not spoken_text:
                is_last_scene = (scene.scene_index >= len(narrative.segments))
                if is_last_scene:
                    if scene.scene_index == len(scene_plan.scenes):
                        spoken_text = (narrative.cta or "लाइक और सब्सक्राइब करें।").strip()
                        subtitle_text = "Like aur subscribe karein."
                    else:
                        spoken_text = ""
                        subtitle_text = ""
                else:
                    spoken_text = ""
                    subtitle_text = ""

            if not subtitle_text and spoken_text:
                subtitle_text = spoken_text

            # Check visual asset info if provided
            provider = ""
            confidence = 0.0
            asset_type = "image"
            if visual_assets:
                asset = next((a for a in visual_assets.assets if a.scene_index == scene.scene_index), None)
                if asset:
                    provider = asset.provider
                    confidence = asset.confidence
                    asset_type = asset.asset_type

            legacy_segments.append(
                Segment(
                    text=spoken_text,
                    subtitle=subtitle_text,
                    image_prompt=scene.ai_image_prompt,
                    search_query=scene.search_query or scene.visual_description,
                    visual_type=asset_type,
                    visual_category=scene.purpose,
                    visual_concept=scene.visual_description,
                    visual_provider=provider,
                    visual_confidence=confidence,
                )
            )

        # Retrieve SEO metadata from narrative script
        seo = getattr(narrative, "seo", {}) or {}
        title = seo.get("title") or (topic + " - Hindi Shorts")
        description = seo.get("description") or f"Automated Short video about {topic}. {narrative.context}"
        tags = seo.get("tags") or ["shorts", "hindi", "facts", topic]
        hashtags = seo.get("hashtags") or ["Shorts", "Hindi", "Facts"]

        # Construct legacy Script object
        return Script(
            title=title.strip(),
            description=description.strip(),
            tags=[str(t).strip() for t in tags if str(t).strip()],
            hashtags=[str(ht).strip().lstrip("#") for ht in hashtags if str(ht).strip()],
            segments=legacy_segments,
            topic=topic,
        )

    @staticmethod
    def extract_image_paths(
        scene_plan: ScenePlanManifest,
        visual_assets: VisualAssetManifest,
    ) -> list[Path]:
        image_paths = []
        for scene in scene_plan.scenes:
            asset = next((a for a in visual_assets.assets if a.scene_index == scene.scene_index), None)
            if asset and asset.file_path:
                image_paths.append(Path(asset.file_path))
        return image_paths
