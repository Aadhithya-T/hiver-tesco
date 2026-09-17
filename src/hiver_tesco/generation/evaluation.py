"""Evaluation and review table extraction for generated replies."""

from typing import Any, Dict, List
import pandas as pd

from hiver_tesco.generation.models import AuditRecord, GenerationStatus


def create_generation_review_dataframe(records: List[AuditRecord]) -> pd.DataFrame:
    """Extract a clean, side-by-side DataFrame comparing LLM drafts with template baseline."""
    rows = []
    for r in records:
        rows.append({
            "conversation_id": r.conversation_id,
            "policy_action": r.policy_action,
            "generation_status": r.generation_status,
            "evidence_source_id": r.evidence_source_ids[0] if r.evidence_source_ids else "none",
            "evidence_score": r.evidence_similarity_scores[0] if r.evidence_similarity_scores else None,
            "suggested_routing_category": r.suggested_routing_category or "none",
            "llm_draft_reply": r.final_draft or "[NO DRAFT - ESCALATED]",
            "template_baseline_reply": r.template_baseline_reply,
            "sentiment_conflict_detected": r.sentiment_conflict_detected,
            "sentiment_conflict_reason": r.sentiment_conflict_reason or "none",
            "cache_hit": r.cache_hit,
        })
    return pd.DataFrame(rows)


def summarize_generation_metrics(records: List[AuditRecord]) -> Dict[str, Any]:
    """Compute summary counts across policy actions and generation statuses."""
    total = len(records)
    policy_escalated = sum(1 for r in records if r.generation_status == GenerationStatus.POLICY_ESCALATED.value)
    drafts_generated = sum(1 for r in records if r.generation_status == GenerationStatus.GENERATED.value)
    model_refused = sum(1 for r in records if r.generation_status == GenerationStatus.MODEL_REFUSED_INSUFFICIENT_EVIDENCE.value)
    model_errors = sum(1 for r in records if r.generation_status == GenerationStatus.MODEL_ERROR.value)
    sentiment_conflicts = sum(1 for r in records if r.sentiment_conflict_detected)
    cache_hits = sum(1 for r in records if r.cache_hit)

    eligible = total - policy_escalated

    return {
        "total_queries": total,
        "policy_escalated_count": policy_escalated,
        "eligible_for_drafting_count": eligible,
        "drafts_generated_count": drafts_generated,
        "model_refusals_count": model_refused,
        "model_errors_count": model_errors,
        "sentiment_conflicts_detected_count": sentiment_conflicts,
        "cache_hits_count": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 4) if total > 0 else 0.0,
        "drafting_success_rate": round(drafts_generated / eligible, 4) if eligible > 0 else 0.0,
    }
