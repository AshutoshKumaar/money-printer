from __future__ import annotations

SCENE_PLAN_PROMPT_TEMPLATE = """
You are a professional Film Director and Director of Photography (DoP).
Your job is to transform the following YouTube Shorts narrative script into a highly structured, production-ready cinematic Scene Plan.

Narrative Script:
Hook: {hook}
Context: {context}
Segments:
{segments}
Ending: {ending}
CTA: {cta}

Core Instructions:
- You must output exactly one scene shot per segment index in the "scenes" array. The length of the "scenes" array must be exactly equal to the number of segments in the Narrative Script. Do not generate multiple shots or multiple scene objects for the same segment index.
- Plan each shot like a filmmaker. Do not write script narration or text; write visual cues and instructions.
- Ensure visual uniqueness. Avoid generic prompts (like "person", "man", "woman", "nature"). Instead, write highly detailed, descriptive visual concepts.
- Camera Motion examples: Slow Push, Slow Zoom, Handheld, Orbit, Dolly In, Dolly Out, Tilt, Pan, Crash Zoom, Static.
- Camera Angle examples: Wide, Close Up, Extreme Close Up, Top Down, POV, Low Angle, High Angle.
- Lens type examples: 35mm anamorphic, 85mm portrait, 24mm wide angle, macro lens, etc.
- AI image prompts must describe a vertical 9:16 cinematic scene. Include lighting, color grading, atmosphere, composition. Specify "no text, no watermark, ultra realistic, movie quality, professional cinematography".
- Stock search queries (stock_search_query) must be descriptive and context-specific English keywords directly relevant to the facts of that segment. Avoid generic filler queries like "nature", "person", "man", or "background".
- Align transitions, lighting, music, and camera style to the segment's emotion.
- Return a JSON object matching the schema below.

JSON Schema:
{{
  "overall_style": "High-level cinematic style description (e.g., neo-noir sci-fi, cinematic documentary)",
  "scene_count": 12,
  "estimated_runtime": 57.5,
  "scenes": [
    {{
      "scene_index": 1,
      "shot_index": 1,
      "duration_seconds": 5.2,
      "purpose": "Establish hook and curiosity",
      "visual_description": "Detailed visual setup of the shot",
      "camera_angle": "Wide / Close Up / POV / Low Angle / etc.",
      "camera_motion": "Slow Push / Orbit / Dolly In / etc.",
      "lens_type": "35mm anamorphic lens",
      "lighting": "Dramatic chiaroscuro lighting, volumetric light rays",
      "environment": "Misty dark forest / futuristic research laboratory / etc.",
      "time_of_day": "Dusk / Night / Noon / Golden hour",
      "color_palette": "Teal and orange / desaturated cool blues / vibrant amber",
      "emotion": "curiosity / surprise / wonder / fear / urgency",
      "transition_in": "Cut / Fade in / Whip pan / Cross dissolve",
      "transition_out": "Cut / Fade out / Whip pan / Cross dissolve",
      "caption_style": "Styled text instructions",
      "stock_search_query": "English keywords optimized for stock image/video search (Pexels/Pixabay relevance)",
      "ai_image_prompt": "vertical 9:16 cinematic prompt, movie quality, ultra realistic, professional cinematography, no text, no watermark",
      "stock_video_query": "English keywords for stock video search",
      "sound_effects": "Low hum, cinematic rise, whoosh transition",
      "background_music_mood": "suspenseful / majestic / dramatic / upbeat",
      "priority": 1
    }}
  ]
}}
"""
