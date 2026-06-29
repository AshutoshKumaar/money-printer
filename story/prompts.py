from __future__ import annotations

STORY_PROMPT_TEMPLATE = """
You are a professional Narrative Designer. Your goal is to convert the following verified facts into a structured, highly engaging, and factual Short video script.

Verified Facts to Use (with Fact IDs):
{verified_facts}

Narration Language: {language}

Core Instructions:
1. You must strictly use ONLY the verified facts provided above. Never invent facts, exaggerate statistics, or introduce unverified info.
2. Structure the script into defined narrative phases:
   - Hook: Instantly grabbing hook statement.
   - Context: Introductory context of the topic.
   - Escalation: Increasing curiosity and details.
   - Climax: Peak explanation or reveal of the strongest fact.
   - Ending: Natural wrapping thought or mystery.
3. Every narration segment must introduce new information. No filler segments.
4. Curiosity loops: Each narration segment must resolve the previous segment's curiosity while introducing a new curiosity.
5. Pacing: Structure the narration segments so that the total estimated speaking duration is between 35 and 55 seconds.
6. Language: Spoken narration must be written in natural, conversational {language}. Keep sentences short and optimized for spoken delivery.
7. Link every narration segment to the relevant verified fact IDs that support it in `verified_fact_ids`. If a segment references uncertainty (e.g. disputed evidence), link it and maintain the uncertainty.

Respond ONLY with a JSON object matching this schema:
{{
  "language": "{language}",
  "hook": "Grabbing hook narration line",
  "context": "Context narration line",
  "escalation": "Escalation narrative line",
  "climax": "Climax narrative line",
  "ending": "Ending narrative line",
  "narration_segments": [
    {{
      "index": 1,
      "narration_text": "Narration text in {language}",
      "estimated_duration": 4.5,
      "target_start": 0.0,
      "target_end": 4.5,
      "emotion": "curiosity",
      "purpose": "Define the purpose/curiosity hook",
      "verified_fact_ids": ["fact_id_1"],
      "beat_type": "hook"
    }}
  ],
  "quality": {{
    "retention_score": 0.92,
    "pacing_score": 0.88,
    "curiosity_score": 0.95,
    "clarity_score": 0.90,
    "emotional_score": 0.85,
    "estimated_retention_curve": [1.0, 0.95, 0.92, 0.89, 0.87, 0.85]
  }}
}}

Constraints on Field Values:
- "emotion": Must be one of: "curiosity", "suspense", "wonder", "surprise", "fear", "urgency", "neutral".
- "beat_type": Must be one of: "hook", "question", "setup", "evidence", "twist", "reveal", "reflection", "cta".
"""
