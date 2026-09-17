"""Prompt templates and structured schema definitions for evidence-grounded generation."""

import json
from typing import Any, Dict, List, Optional

from hiver_tesco.generation.models import EvidenceMatch
from hiver_tesco.policy.models import PolicyDecision
from hiver_tesco.retrieval.pii import sanitize_evidence_text

PROMPT_VERSION = "v1.0.0"

SYSTEM_PROMPT = """You are Tesco's official customer support assistant on Twitter/X.
Your duty is to draft a polite, helpful, concise support reply grounded STRICTLY in provided historical Tesco evidence.

CORE VOICE & FORMATTING RULES:
1. Tone: Friendly, professional, empathetic, British English Tesco style.
2. Length: Concise tweet format (1-2 sentences, strictly under 280 characters).
3. Signature: Sign off with a warm closing or #EveryLittleHelps where appropriate.

STRICT ANTI-HALLUCINATION CONSTRAINTS (ZERO TOLERANCE):
1. NEVER invent order status, delivery slots, driver locations, or tracking details.
2. NEVER promise refunds, vouchers, credits, discounts, or financial compensation.
3. NEVER invent store opening hours, stock levels, or policies not present in the evidence.
4. NEVER claim you have accessed the customer's account or checked order systems.
5. NEVER ask for credit card numbers, passwords, or sensitive PII over public tweets.

MANDATORY OUTPUT FORMAT:
You must respond with a raw, valid JSON object only (no markdown, no backticks, no preamble):
- If the evidence contains sufficient facts to answer safely and accurately:
  {
    "status": "draft",
    "reply": "<concise Tesco reply grounded in evidence>",
    "grounded_evidence_id": "<source conversation ID from evidence>"
  }
- If evidence is missing, weak, or insufficient to answer without guessing/hallucinating:
  {
    "status": "escalate",
    "reason": "insufficient_evidence_to_answer_accurately",
    "suggested_routing_category": "general_customer_support"
  }
"""


def build_user_prompt(
    query_text: str,
    policy_decision: PolicyDecision,
    evidence: Optional[EvidenceMatch] = None,
    evidence_status: Optional[str] = None,
) -> str:
    """Build sanitized user prompt injecting customer issue, policy boundaries, and evidence."""
    # 1. Strict PII sanitization of query text before prompt injection
    sanitized_query = sanitize_evidence_text(query_text)

    # 2. Extract policy guidance
    guidance = policy_decision.response_guidance
    allowed = guidance.allowed_actions if guidance else ["provide_general_help"]
    disallowed = guidance.disallowed_actions if guidance else ["do_not_promise_refunds"]

    # 3. Format Evidence Block
    if evidence and (evidence_status == "accepted" or evidence.score >= 0.55):
        evidence_text = (
            f"Evidence Source ID: {evidence.source_conversation_id}\n"
            f"Similarity Score: {evidence.score:.3f}\n"
            f"Historical Customer Issue: {evidence.sanitized_customer_issue}\n"
            f"Historical Tesco Resolution: {evidence.sanitized_tesco_resolution}"
        )
    else:
        evidence_text = (
            "NO HISTORICAL EVIDENCE AVAILABLE.\n"
            "evidence_status: insufficient\n"
            "You do not have verified facts for this query."
        )

    prompt = (
        f"CUSTOMER INQUIRY (Sanitized):\n"
        f"\"{sanitized_query}\"\n\n"
        f"POLICY GUIDANCE:\n"
        f"- Allowed Actions: {', '.join(allowed)}\n"
        f"- Strictly Forbidden Actions: {', '.join(disallowed)}\n\n"
        f"FILTERED HISTORICAL TESCO EVIDENCE:\n"
        f"{evidence_text}\n\n"
        f"Draft the strict JSON response now:"
    )

    return prompt
