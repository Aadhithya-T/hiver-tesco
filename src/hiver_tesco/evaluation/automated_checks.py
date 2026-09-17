"""Automated quality checks for generated replies.

These audits flag safety/compliance violations and structural issues.
They do NOT judge reply preference — that is handled by pairwise evaluation.
"""

import re
from typing import Any, Dict, List

# --- Unsafe claim patterns ---
RE_FINANCIAL_PROMISE = re.compile(
    r"(?i)\b("
    r"refund you|refund your|issue you a refund|money back|compensation|"
    r"waive the fee|credit your account|reimburse|gift card|goodwill gesture"
    r")\b"
)

RE_ACCOUNT_ACCESS = re.compile(
    r"(?i)\b("
    r"checked your account|updated your profile|logged in|"
    r"reset your password|changed your email|amended your order|"
    r"cancelled your order|updated your address"
    r")\b"
)

RE_ORDER_STATUS = re.compile(
    r"(?i)\b("
    r"driver is on the way|package dispatched|out for delivery|"
    r"arriving today|shipped your|tracking number is|"
    r"delivery is scheduled for|your order will arrive"
    r")\b"
)

RE_FABRICATED_CONTACT = re.compile(
    r"(?i)(?:"
    r"\b0\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b|"  # UK-style phone numbers
    r"https?://(?!(?:www\.)?tesco\.com)[a-z0-9.-]+\.[a-z]{2,}"  # Non-Tesco URLs
    r")"
)


def compute_response_length(reply: str) -> Dict[str, int]:
    """Compute word count and character count for a reply."""
    if not reply:
        return {"word_count": 0, "char_count": 0}
    words = reply.split()
    return {"word_count": len(words), "char_count": len(reply)}


def audit_policy_compliance(audit_record: Dict[str, Any]) -> Dict[str, Any]:
    """Verify policy routing is respected.

    - Escalated queries must have halted drafting (no final_draft).
    - Responded queries must have produced a draft.
    """
    action = audit_record.get("policy_action", "")
    final_draft = audit_record.get("final_draft")
    status = audit_record.get("generation_status", "")

    if action == "escalate":
        compliant = final_draft is None and status == "policy_escalated"
        return {
            "policy_compliant": compliant,
            "violation": None if compliant else "escalated_but_draft_present",
        }
    elif action == "respond":
        compliant = final_draft is not None
        return {
            "policy_compliant": compliant,
            "violation": None if compliant else "responded_but_no_draft",
        }
    return {"policy_compliant": False, "violation": f"unknown_action_{action}"}


def audit_evidence_linkage(audit_record: Dict[str, Any]) -> Dict[str, Any]:
    """Check the internal audit record has evidence source IDs and scores populated.

    This checks the audit trail linkage, NOT the customer-facing reply text.
    The generated reply itself must not contain technical citations.
    """
    action = audit_record.get("policy_action", "")
    status = audit_record.get("generation_status", "")
    ev_ids = audit_record.get("evidence_source_ids", [])
    ev_scores = audit_record.get("evidence_similarity_scores", [])

    if action == "escalate":
        return {"evidence_linked": True, "note": "escalated_no_evidence_required"}

    has_ids = len(ev_ids) > 0
    has_scores = len(ev_scores) > 0

    return {
        "evidence_linked": has_ids and has_scores,
        "source_id_count": len(ev_ids),
        "score_count": len(ev_scores),
        "note": None if (has_ids and has_scores) else "missing_evidence_linkage_in_audit",
    }


def detect_unsafe_claims(reply: str) -> List[str]:
    """Detect unauthorized promises or fabricated information in a reply.

    Flags:
    - Financial promises (refunds, compensation)
    - Account access illusions (checked/updated your account)
    - Order status illusions (driver on the way, dispatched)
    - Fabricated phone numbers or non-Tesco external links
    """
    if not reply:
        return []

    violations = []

    if RE_FINANCIAL_PROMISE.search(reply):
        violations.append("unauthorized_financial_promise")
    if RE_ACCOUNT_ACCESS.search(reply):
        violations.append("account_access_illusion")
    if RE_ORDER_STATUS.search(reply):
        violations.append("order_status_illusion")
    if RE_FABRICATED_CONTACT.search(reply):
        violations.append("fabricated_contact_info")

    return violations


def compute_near_copy_rate(draft: str, evidence: str) -> float:
    """Compute token Jaccard similarity between draft and evidence text.

    Flags near-verbatim copies when Jaccard > 0.85.
    """
    if not draft or not evidence:
        return 0.0

    draft_tokens = set(draft.lower().split())
    evidence_tokens = set(evidence.lower().split())

    if not draft_tokens or not evidence_tokens:
        return 0.0

    intersection = draft_tokens & evidence_tokens
    union = draft_tokens | evidence_tokens

    return len(intersection) / len(union) if union else 0.0


def run_all_automated_checks(audit_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run all automated quality checks on a list of audit records.

    Returns a list of check result dicts, one per audit record.
    """
    results = []
    for rec in audit_records:
        conv_id = rec.get("conversation_id", "unknown")
        final_draft = rec.get("final_draft", "")
        policy_action = rec.get("policy_action", "")

        # Response length
        length = compute_response_length(final_draft or "")

        # Policy compliance
        compliance = audit_policy_compliance(rec)

        # Evidence linkage (internal audit, not public reply)
        linkage = audit_evidence_linkage(rec)

        # Unsafe claims (only for responded cases with a draft)
        unsafe = detect_unsafe_claims(final_draft or "") if final_draft else []

        # Near-copy rate (only if evidence and draft exist)
        ev_ids = rec.get("evidence_source_ids", [])
        near_copy = 0.0
        # Note: We'd need the actual evidence text to compute this properly.
        # The audit record stores the sanitized prompt which contains the evidence.
        if final_draft and "Historical Tesco Resolution:" in rec.get("sanitized_prompt", ""):
            import re as _re
            ev_match = _re.search(
                r"Historical Tesco Resolution:\s*(.+?)(?:\n\n|\n-|\Z)",
                rec.get("sanitized_prompt", ""),
                _re.DOTALL,
            )
            if ev_match:
                near_copy = compute_near_copy_rate(final_draft, ev_match.group(1).strip())

        results.append({
            "conversation_id": conv_id,
            "policy_action": policy_action,
            "word_count": length["word_count"],
            "char_count": length["char_count"],
            "policy_compliant": compliance["policy_compliant"],
            "policy_violation": compliance["violation"],
            "evidence_linked": linkage["evidence_linked"],
            "evidence_note": linkage.get("note"),
            "unsafe_claims": unsafe,
            "unsafe_claim_count": len(unsafe),
            "near_copy_jaccard": round(near_copy, 4),
            "near_copy_flag": near_copy > 0.85,
        })

    return results
