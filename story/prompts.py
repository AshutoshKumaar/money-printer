from __future__ import annotations

STORY_PROMPT_TEMPLATE = """
You are a professional YouTube Shorts Storyteller and Documentary Writer. 
Your goal is to convert the following verified facts into a cinematic 55-58 second Hindi short video script.

Verified Facts to Use:
{verified_facts}

Core Instructions:
- You must strictly use ONLY the verified facts provided above. Never invent facts, exaggerate statistics, or introduce unverified info.
- Structure the script as follows:
  1. Hook: 1-2 sentences. Instantly creates curiosity, no greetings (never say "Namaste", "Hey guys", etc.), captures attention in the first 3 seconds.
  2. Context: Introduce the topic naturally.
  3. Progressive Curiosity: Reveal details step-by-step, building tension.
  4. Evidence: Introduce verified data or historical records naturally inside the narration.
  5. Final Reveal: Place the strongest verified fact near the end as a shocking reveal.
  6. Ending & CTA: Wrap up with a natural, brief CTA (e.g. comment your thoughts, follow for more).
- Language Rules:
  * Spoken narration (`spoken_hindi`) must be in natural, easy-to-understand Devanagari Hindi.
  * Captions (`caption_keywords`) must be in pure Devanagari Hindi, containing 3-5 clean Devanagari visual keywords representing the segment (e.g., "रहस्य बर्फ मौत पहाड़"). Do NOT output Hinglish sentences, English words, punctuation, or full sentences. The captions will be displayed directly as-is on the video.
  * Search queries (`search_query`) must be in English keywords, useful for Pexels search.
  * Visual concepts (`visual_concept`) must describe the scene composition.
- Timing: Keep each scene short (9-12 words). Aim for a total of 10-12 scenes.
- Assess the emotional weight (0.0 to 1.0) of each segment for: curiosity, fear, surprise, wonder, and urgency.
- Return a JSON object matching the schema below.
 
JSON Schema:
{{
  "hook": "Instantly grabbing hook (Devanagari Hindi)",
  "context": "Context statement (Devanagari Hindi)",
  "ending": "Ending wrap up statement (Devanagari Hindi)",
  "cta": "Call to action statement (Devanagari Hindi)",
  "estimated_duration": 57.5,
  "estimated_words": 115,
  "emotion_curve": {{
    "curiosity": 0.9,
    "fear": 0.2,
    "surprise": 0.6,
    "wonder": 0.5,
    "urgency": 0.7
  }},
  "retention_score": 0.92,
  "segments": [
    {{
      "index": 1,
      "spoken_hindi": "spoken Devanagari line",
      "caption_keywords": "3-5 Devanagari Hindi visual keywords (no Hinglish, no punctuation, e.g., रहस्य बर्फ मौत पहाड़)",
      "search_query": "English search query",
      "visual_concept": "detailed scene visual description",
      "emotion_curve": {{
        "curiosity": 0.95,
        "fear": 0.1,
        "surprise": 0.7,
        "wonder": 0.6,
        "urgency": 0.4
      }}
    }}
  ]
}}
"""
