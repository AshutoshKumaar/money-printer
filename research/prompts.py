from __future__ import annotations

RESEARCH_PROMPT_TEMPLATE = """
Conduct objective, factual research on the following topic: "{topic}".

Your goal is to extract, verify, and organize structured knowledge about the topic.
Do NOT write scripts, narration, voiceover text, storytelling, or clickbait.
Do NOT suggest image prompts, video descriptions, or YouTube tags.
Provide direct, highly information-dense, and objective content for all fields.

Return a JSON object conforming exactly to the following structure:

{{
  "topic_summary": "High-level summary of the topic (1-3 sentences).",
  "historical_context": {{
    "timeline": ["Date/Period: Event details", "Date/Period: Event details"],
    "overview": "Overview of historical context or origins."
  }},
  "scientific_context": {{
    "explanation": "Detailed scientific/technical explanation of how/why it works.",
    "concepts": ["Key scientific term/concept 1", "Key scientific term/concept 2"]
  }},
  "important_entities": {{
    "people": ["Key person name: Role/significance"],
    "places": ["Specific geographic locations"],
    "organizations": ["Relevant scientific/historical organizations"],
    "technologies": ["Relevant technologies, tools, or methods"],
    "events": ["Specific historical/scientific events"],
    "objects": ["Key physical objects or artifacts"]
  }},
  "verified_facts": [
    {{
      "fact": "Factual claim or statement.",
      "confidence_score": 0.95,
      "verification_status": "verified",
      "reasoning": "Reasoning, sources, or evidence supporting the claim.",
      "evidence_level": "strong",
      "source_type": "scientific consensus"
    }}
  ],
  "common_misconceptions": [
    {{
      "myth": "Common myth or misconception.",
      "verified_fact": "The actual verified reality."
    }}
  ],
  "unanswered_questions": [
    {{
      "question": "Scientific, historical, or general mystery/unanswered question.",
      "uncertainty_type": "scientific"
    }}
  ],
  "visual_opportunities": [
    {{
      "opportunity_type": "photograph",
      "description": "Factual description of a suitable visual (e.g. 'A map showing the boundaries of the Bermuda Triangle')."
    }}
  ],
  "story_opportunities": {{
    "strongest_hook": "A curiosity-inducing fact or question to start the video.",
    "biggest_surprise": "The most surprising revelation.",
    "emotional_moments": ["Moments of fear, awe, or urgency"],
    "best_climax": "The peak explanation or reveal of the video.",
    "strongest_ending": "Ending note, call to action, or lingering mystery."
  }},
  "seo_research": {{
    "primary_keywords": ["main keyword 1", "main keyword 2"],
    "secondary_keywords": ["secondary keyword 1", "secondary keyword 2"],
    "related_concepts": ["related concept 1", "related concept 2"]
  }},
  "research_confidence": {{
    "overall_score": 0.95,
    "potential_weak_areas": ["Areas lacking consensus or primary source access"],
    "recommended_verification_priority": "high"
  }}
}}

Constraints on Field Values:
- "verification_status": Must be either "verified", "partially_verified", or "unverified".
- "evidence_level": Must be either "strong", "moderate", or "weak".
- "source_type": Must be one of: "scientific consensus", "historical consensus", "government publication", "academic journal", "encyclopedia", "first-hand account", "requires verification", or "unknown".
- "opportunity_type" in visual_opportunities: Must be one of: "photograph", "map", "diagram", "historical artwork", "reconstruction", or "animation".
- "uncertainty_type" in unanswered_questions: Must be one of: "scientific", "historical", or "general".
"""
