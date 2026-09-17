"""Evaluation and trade-off analysis for deterministic policy layer."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from hiver_tesco.policy.engine import DeterministicPolicyEngine
from hiver_tesco.policy.models import PolicyAction, PolicyContext, PolicyDecision


@dataclass
class PolicyEvaluationMetrics:
    """Comprehensive escalation evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
    rule_coverage: Dict[str, int]
    category_coverage: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_policy_predictions(
    decisions: List[PolicyDecision],
    y_true: List[str],
) -> PolicyEvaluationMetrics:
    """Compute classification metrics for policy decisions against human golden labels."""
    tp = fp = tn = fn = 0
    rule_counts: Dict[str, int] = {}
    cat_counts: Dict[str, int] = {}

    for dec, true_label in zip(decisions, y_true):
        pred_label = "escalate" if dec.action == PolicyAction.ESCALATE else "do_not_escalate"

        if pred_label == "escalate":
            rule_counts[dec.rule_id] = rule_counts.get(dec.rule_id, 0) + 1
            cat_counts[dec.rule_category] = cat_counts.get(dec.rule_category, 0) + 1

        if true_label == "escalate":
            if pred_label == "escalate":
                tp += 1
            else:
                fn += 1
        else:  # do_not_escalate
            if pred_label == "escalate":
                fp += 1
            else:
                tn += 1

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return PolicyEvaluationMetrics(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        false_negative_rate=round(fnr, 4),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        total_samples=total,
        rule_coverage=rule_counts,
        category_coverage=cat_counts,
    )


def extract_policy_error_tables(
    records: List[Dict[str, Any]],
    decisions: List[PolicyDecision],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate False Positive and False Negative review tables with detailed diagnostics."""
    fp_rows = []
    fn_rows = []

    for rec, dec in zip(records, decisions):
        pred_label = "escalate" if dec.action == PolicyAction.ESCALATE else "do_not_escalate"
        true_label = rec["human_escalate"]

        diag_row = {
            "conversation_id": rec["conversation_id"],
            "query_text": rec["query_text"],
            "human_label": true_label,
            "human_rationale": rec.get("human_rationale", ""),
            "policy_action": dec.action.value,
            "policy_rule_id": dec.rule_id,
            "policy_category": dec.rule_category,
            "matched_triggers": ", ".join(dec.matched_triggers) if dec.matched_triggers else "none",
            "model_confidence": rec.get("model_confidence"),
            "retrieval_score": rec.get("retrieval_score"),
        }

        if true_label == "do_not_escalate" and pred_label == "escalate":
            diag_row["trade_off_impact"] = "unnecessary_escalation_operator_overhead"
            fp_rows.append(diag_row)
        elif true_label == "escalate" and pred_label == "do_not_escalate":
            diag_row["trade_off_impact"] = "missed_escalation_safety_customer_risk"
            fn_rows.append(diag_row)

    return pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)
