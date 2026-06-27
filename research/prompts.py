from __future__ import annotations

RESEARCH_PROMPT_TEMPLATE = """
Conduct factual research on the following topic: "{topic}".

Instructions:
- Your output must be purely factual, objective, structured, and informative.
- Do NOT write scripts, narration, voiceover text, storytelling, or clickbait.
- Avoid opinions, exaggerations, emotions, and hyperbole.
- Provide direct, concise information for each section.
- If a section has no relevant facts (e.g. no controversies or myths for simple topics), return an empty list.
- Assess your own research confidence as a float value between 0.0 and 1.0 in `confidence_score` (1.0 being highly confident with verified primary sources, 0.0 being completely unverified).
- Return a JSON object matching the schema below.

JSON Schema:
{{
  "summary": "High-level objective summary of the topic (1-3 sentences).",
  "facts": ["Fact 1", "Fact 2", ...],
  "statistics": ["Stat 1 (with numbers/percentages)", "Stat 2", ...],
  "timeline": ["Event 1 (with date/year)", "Event 2", ...],
  "locations": ["Location 1", "Location 2", ...],
  "people": ["Key person 1", "Key person 2", ...],
  "scientific_explanations": ["Explanation of how/why something works", ...],
  "myths": ["Common misconception 1 vs the reality", ...],
  "controversies": ["Debate or controversy 1", ...],
  "sources": ["Expected references, organizations, or scientific databases", ...],
  "interesting_hooks": ["Surprising or curiosity-inducing fact that would make a good hook", ...],
  "warnings": ["Warning, safety hazard, contradiction, or limitation of research", ...],
  "confidence_score": 0.95
}}
"""
