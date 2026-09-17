"""Separate evaluation of escalation routing decisions.

Escalated conversations are not pairwise-compared for reply quality
because no generated reply was produced. This module evaluates them
as routing/safety decisions against the golden labels.
"""

from typing import Any, Dict, List, Optional

import pandas as pd


def evaluate_escalation_routing(
    audit_records: List[Dict[str, Any]],
    golden_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate escalation routing accuracy against golden labels.

    Args:
        audit_records: List of audit record dicts from Phase 6 generation.
        golden_df: Golden set DataFrame with 'conversation_id' and 'escalation_needed' columns.

    Returns:
        Dict with confusion matrix, accuracy, precision, recall, F1, and per-rule breakdown.
    """
    golden_df = golden_df.copy()
    golden_df["conversation_id"] = golden_df["conversation_id"].astype(str)
    golden_map = dict(zip(golden_df["conversation_id"], golden_df["escalation_needed"]))

    tp = fp = tn = fn = 0
    per_rule: Dict[str, Dict[str, int]] = {}

    for rec in audit_records:
        conv_id = str(rec["conversation_id"])
        policy_action = rec["policy_action"]
        golden_label = golden_map.get(conv_id)

        if golden_label is None:
            continue

        predicted_escalate = policy_action == "escalate"
        actual_escalate = golden_label == "escalate"

        if predicted_escalate and actual_escalate:
            tp += 1
        elif predicted_escalate and not actual_escalate:
            fp += 1
        elif not predicted_escalate and actual_escalate:
            fn += 1
        else:
            tn += 1

        # Per-rule tracking
        rule_id = rec.get("policy_rule_id", "DEFAULT_SAFE_RESPOND")
        if rule_id not in per_rule:
            per_rule[rule_id] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}
        per_rule[rule_id]["total"] += 1
        if predicted_escalate and actual_escalate:
            per_rule[rule_id]["tp"] += 1
        elif predicted_escalate and not actual_escalate:
            per_rule[rule_id]["fp"] += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "total_evaluated": total,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_rule_breakdown": per_rule,
    }


def build_escalation_review_table(
    audit_records: List[Dict[str, Any]],
    golden_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a review table for escalation decisions.

    Returns DataFrame with one row per conversation showing the policy decision,
    golden label, and whether the routing was correct.
    """
    golden_df = golden_df.copy()
    golden_df["conversation_id"] = golden_df["conversation_id"].astype(str)
    golden_map = dict(zip(golden_df["conversation_id"], golden_df["escalation_needed"]))

    rows = []
    for rec in audit_records:
        conv_id = str(rec["conversation_id"])
        policy_action = rec["policy_action"]
        golden_label = golden_map.get(conv_id, "unknown")

        predicted_escalate = policy_action == "escalate"
        actual_escalate = golden_label == "escalate"
        correct = predicted_escalate == actual_escalate

        classification = "correct"
        if predicted_escalate and not actual_escalate:
            classification = "false_positive"
        elif not predicted_escalate and actual_escalate:
            classification = "false_negative"

        rows.append({
            "conversation_id": conv_id,
            "policy_action": policy_action,
            "policy_rule_id": rec.get("policy_rule_id", ""),
            "suggested_routing_category": rec.get("suggested_routing_category", ""),
            "golden_escalation": golden_label,
            "correct": correct,
            "classification": classification,
        })

    return pd.DataFrame(rows)
