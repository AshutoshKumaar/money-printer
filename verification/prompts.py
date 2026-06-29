from __future__ import annotations

VERIFICATION_PROMPT_TEMPLATE = """
You are a professional fact-checker. Your job is to verify, assess, and evaluate all factual statements compiled on the topic: "{topic}".

Research Context to Verify:
Summary: {summary}
Facts: {facts}
Misconceptions: {myths}
Timeline: {timeline}

Instructions:
1. Verify every statement individually. Do NOT rewrite the original fact. If clarification or corrections are needed, suggest them in `suggested_clarification` while keeping `original_fact` intact.
2. Detect duplicate facts: do NOT remove them. Simply flag them with `is_duplicate: true`.
3. Detect contradictions: if contradictory facts or conflicting evidence exist, preserve both viewpoints. Mark the status as "disputed", detail the disagreement in `contradictions_detected` and `reasoning`, and do NOT force a single conclusion unless the evidence clearly supports it.
4. For each fact, determine its verification status: "verified", "partially_verified", "disputed", "insufficient_evidence", or "unverified".
5. Evaluate `confidence_score` (0.0 to 1.0) and `importance_score` (0.0 to 1.0) for each statement.
6. Provide a logical fact-checking explanation in `reasoning`.
7. Categorize each record.
8. Assess the overall `verification_quality_score` (0.0 to 1.0) for the research.
9. List any warnings (e.g. conflicting accounts, critical omissions, or outdated data).

Respond ONLY with a JSON object matching this schema:
{{
  "verified_facts": [
    {{
      "fact": "Original claim text",
      "status": "verified",
      "original_fact": "Original claim text",
      "suggested_clarification": "Cleaned or corrected version if needed, else null",
      "confidence_score": 0.98,
      "reasoning": "Reasoning referencing consensus/sources",
      "evidence_level": "strong",
      "source_type": "scientific consensus",
      "contradictions_detected": [],
      "is_duplicate": false,
      "importance_score": 0.9,
      "category": "science"
    }}
  ],
  "verification_warnings": ["Warning 1", "Warning 2"],
  "verification_quality_score": 0.90
}}

Constraints on Field Values:
- "status": Must be one of: "verified", "partially_verified", "disputed", "insufficient_evidence", or "unverified".
- "evidence_level": Must be one of: "strong", "moderate", or "weak".
- "source_type": Must be one of: "scientific consensus", "historical consensus", "government publication", "academic journal", "encyclopedia", "first-hand account", "requires verification", or "unknown".
"""
