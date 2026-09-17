"""Retrieval evaluation engine computing Hit@k, Precision@k, and MRR.

Enforces strict vs relaxed relevance definitions and requires validated human
annotations before metrics are reported.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from hiver_tesco.retrieval.relevance_validator import validate_relevance_file


@dataclass
class MetricSummary:
    """Metrics for a retriever under a specific relevance definition."""

    hit_at_1: float
    hit_at_3: float
    precision_at_1: float
    precision_at_3: float
    mrr: float
    num_queries: int


@dataclass
class RetrievalEvaluationReport:
    """Full evaluation report contrasting lexical and semantic retrieval."""

    strict_metrics: Dict[str, MetricSummary]
    relaxed_metrics: Dict[str, MetricSummary]
    filter_stats: Dict[str, Any]
    calibrated_threshold: float
    sample_size_queries: int
    total_evaluated_candidates: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strict_metrics": {k: asdict(v) for k, v in self.strict_metrics.items()},
            "relaxed_metrics": {k: asdict(v) for k, v in self.relaxed_metrics.items()},
            "filter_stats": self.filter_stats,
            "calibrated_threshold": self.calibrated_threshold,
            "sample_size_queries": self.sample_size_queries,
            "total_evaluated_candidates": self.total_evaluated_candidates,
        }


def _is_positive(label: str, strict: bool) -> bool:
    clean = str(label).strip().lower()
    if strict:
        return clean == "relevant"
    return clean in ("relevant", "partially_relevant")


def compute_retrieval_metrics(
    df: pd.DataFrame,
    retriever_type: str,
    strict: bool,
    max_k: int = 3,
) -> MetricSummary:
    """Compute Hit@k, Precision@k, and MRR for a retriever."""
    sub = df[df["retriever_type"] == retriever_type]
    queries = sub["query_id"].unique()
    num_queries = len(queries)
    if num_queries == 0:
        return MetricSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    hits_1 = []
    hits_3 = []
    prec_1 = []
    prec_3 = []
    mrrs = []

    for qid in queries:
        q_rows = sub[sub["query_id"] == qid].sort_values("rank")
        # Check rank 1
        r1_rows = q_rows[q_rows["rank"] == 1]
        r1_pos = False
        if not r1_rows.empty:
            r1_pos = _is_positive(r1_rows.iloc[0]["human_relevance"], strict)

        hits_1.append(1.0 if r1_pos else 0.0)
        prec_1.append(1.0 if r1_pos else 0.0)

        # Check top-k (up to 3)
        top_k_rows = q_rows[q_rows["rank"] <= max_k]
        pos_count = 0
        first_pos_rank = None

        for _, r in top_k_rows.iterrows():
            pos = _is_positive(r["human_relevance"], strict)
            if pos:
                pos_count += 1
                if first_pos_rank is None:
                    first_pos_rank = int(r["rank"])

        hits_3.append(1.0 if pos_count > 0 else 0.0)
        prec_3.append(pos_count / max_k)
        mrrs.append((1.0 / first_pos_rank) if first_pos_rank is not None else 0.0)

    return MetricSummary(
        hit_at_1=float(sum(hits_1) / num_queries),
        hit_at_3=float(sum(hits_3) / num_queries),
        precision_at_1=float(sum(prec_1) / num_queries),
        precision_at_3=float(sum(prec_3) / num_queries),
        mrr=float(sum(mrrs) / num_queries),
        num_queries=num_queries,
    )


def compute_evidence_filter_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute rejection rates and reasons across all candidate retrievals."""
    total = len(df)
    if total == 0:
        return {"total": 0, "rejection_rate": 0.0}

    rejected_mask = df["filter_status"] == "rejected"
    num_rejected = int(rejected_mask.sum())
    num_accepted = total - num_rejected

    reasons: Dict[str, int] = {}
    for r_str in df.loc[rejected_mask, "rejection_reasons"].dropna():
        for item in str(r_str).split(";"):
            clean_item = item.strip()
            if clean_item:
                # Group by primary reason prefix
                prefix = clean_item.split(":")[0].strip()
                reasons[prefix] = reasons.get(prefix, 0) + 1

    return {
        "total_evaluated_candidates": total,
        "accepted_count": num_accepted,
        "rejected_count": num_rejected,
        "rejection_rate": float(num_rejected / total),
        "acceptance_rate": float(num_accepted / total),
        "rejection_reasons_breakdown": reasons,
    }


def calibrate_semantic_threshold(
    df: pd.DataFrame,
    strict: bool = False,
    threshold_range: Tuple[float, float, float] = (0.30, 0.75, 0.05),
) -> float:
    """Find the semantic threshold on Dev judgments that maximizes F1 of relevant retrieval."""
    sem_df = df[df["retriever_type"] == "semantic"].copy()
    if sem_df.empty:
        return 0.45

    min_t, max_t, step = threshold_range
    best_threshold = 0.45
    best_f1 = -1.0

    current = min_t
    while current <= max_t:
        t = round(current, 2)
        # Predicted accepted if score >= t
        pred_acc = sem_df["score"] >= t
        true_pos = sem_df["human_relevance"].apply(lambda l: _is_positive(l, strict))

        tp = int((pred_acc & true_pos).sum())
        fp = int((pred_acc & ~true_pos).sum())
        fn = int((~pred_acc & true_pos).sum())

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

        current += step

    return best_threshold


def evaluate_retrieval_from_file(
    annotated_csv_path: Path,
) -> RetrievalEvaluationReport:
    """Validate human relevance file and compute comprehensive evaluation metrics."""
    df = validate_relevance_file(annotated_csv_path)

    # Compute strict metrics
    strict_lexical = compute_retrieval_metrics(df, "lexical_bm25", strict=True)
    strict_semantic = compute_retrieval_metrics(df, "semantic", strict=True)

    # Compute relaxed metrics
    relaxed_lexical = compute_retrieval_metrics(df, "lexical_bm25", strict=False)
    relaxed_semantic = compute_retrieval_metrics(df, "semantic", strict=False)

    # Filter statistics
    filter_stats = compute_evidence_filter_stats(df)

    # Calibrate threshold on Dev
    calibrated_th = calibrate_semantic_threshold(df, strict=False)

    unique_queries = int(df["query_id"].nunique())

    return RetrievalEvaluationReport(
        strict_metrics={
            "lexical_bm25": strict_lexical,
            "semantic": strict_semantic,
        },
        relaxed_metrics={
            "lexical_bm25": relaxed_lexical,
            "semantic": relaxed_semantic,
        },
        filter_stats=filter_stats,
        calibrated_threshold=calibrated_th,
        sample_size_queries=unique_queries,
        total_evaluated_candidates=len(df),
    )
