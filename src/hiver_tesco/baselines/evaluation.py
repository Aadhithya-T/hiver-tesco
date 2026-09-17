"""Evaluation metrics, confusion matrices, and error analysis routines."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def evaluate_intent_predictions(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics for intent prediction."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    per_class = {}
    for label in labels:
        if label in report:
            per_class[label] = {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1_score": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }


def evaluate_escalation_predictions(
    y_true: List[str],
    y_pred: List[str],
    pos_label: str = "escalate",
) -> Dict[str, Any]:
    """Calculate escalation metrics with explicit emphasis on false negatives."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))

    # Confusion matrix elements: [ [TN, FP], [FN, TP] ]
    labels = ["do_not_escalate", "escalate"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    total_positives = tp + fn
    fnr = round(float(fn / total_positives), 4) if total_positives > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_negative_rate": fnr,
        "confusion_matrix": {
            "labels": labels,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "matrix": cm.tolist(),
        },
    }


def extract_escalation_false_negatives(
    records: List[Dict[str, Any]],
    y_true: List[str],
    y_pred: List[str],
) -> List[Dict[str, Any]]:
    """Extract and analyze false-negative escalation errors (ground truth escalate, predicted do_not_escalate)."""
    false_negatives = []
    for idx, (r, true_val, pred_val) in enumerate(zip(records, y_true, y_pred)):
        if true_val == "escalate" and pred_val == "do_not_escalate":
            false_negatives.append({
                "conversation_id": r.get("conversation_id"),
                "primary_intent": r.get("primary_intent"),
                "escalation_rationale": r.get("escalation_rationale"),
                "query_text": r.get("query_text", ""),
                "evidence_notes": r.get("evidence_notes", ""),
                "ambiguity_notes": r.get("ambiguity_notes", ""),
                "analysis_note": (
                    f"Missed escalation for intent '{r.get('primary_intent')}' with rationale "
                    f"'{r.get('escalation_rationale')}'. Query lacked high-risk trigger keywords."
                ),
            })
    return false_negatives
