from __future__ import annotations

VERIFICATION_PROMPT_TEMPLATE = """
You are a professional fact-checker. Your job is to verify, correct, and evaluate all factual statements compiled on the topic: "{topic}".

Research Context to Verify:
Summary: {summary}
Facts: {facts}
Statistics: {statistics}
Timeline: {timeline}
Myths: {myths}
Controversies: {controversies}

Instructions:
- Verify every statement, check for exaggerations, contradictions, outdated claims, or unsupported statistics.
- For each statement, determine its verification status: "verified", "partially_verified", "unverified", or "contradictory".
- For statements that are "verified" or "partially_verified", provide a corrected/cleaned version, an explanation, a confidence score (0.0 to 1.0), an importance score (0.0 to 1.0), and a category.
- If a statement is "unverified" or "contradictory", place it in the rejected list.
- If a statement required significant corrections, place it in the corrected list.
- Calculate an overall confidence score and a research quality score (0.0 to 1.0).
- List any general warnings (e.g., conflicting accounts, outdated data found).
- Do NOT generate script narration, storytelling, or clickbait.
- Return a JSON object matching the schema below.

JSON Schema:
{{
  "verified_facts": [
    {{
      "original_fact": "Original claim text",
      "status": "verified",
      "corrected_version": "Cleaned, precise claim text",
      "confidence": 0.98,
      "explanation": "Why it is verified, referencing established scientific/historical consensus",
      "importance_score": 0.9,
      "category": "science / history / statistics / etc."
    }}
  ],
  "rejected_facts": [
    {{
      "original_fact": "Original exaggerated or false claim text",
      "status": "contradictory",
      "corrected_version": "",
      "confidence": 0.1,
      "explanation": "Why this fact is rejected or contradictory",
      "importance_score": 0.0,
      "category": ""
    }}
  ],
  "corrected_facts": [
    {{
      "original_fact": "Original partially correct claim text",
      "status": "partially_verified",
      "corrected_version": "Corrected and precise version",
      "confidence": 0.75,
      "explanation": "What was wrong or missing from the original claim",
      "importance_score": 0.8,
      "category": "history"
    }}
  ],
  "warnings": ["Warning 1", "Warning 2"],
  "confidence_score": 0.95,
  "research_quality_score": 0.90
}}
"""
