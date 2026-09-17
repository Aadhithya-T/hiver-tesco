"""CLI to evaluate Development historical retrieval and evidence filtering.

Enforces human relevance validation, computes strict and relaxed Hit@k, Precision@k,
and MRR metrics, calculates evidence filter rejection rates, and exports the final
Phase 4 summary evaluation report.
"""

import json
from pathlib import Path
import sys

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from hiver_tesco.retrieval.evaluation import evaluate_retrieval_from_file
from hiver_tesco.retrieval.relevance_validator import RelevanceValidationError, validate_relevance_file


def main():
    repo_root = Path(__file__).resolve().parent.parent
    annotated_csv_path = repo_root / "data" / "annotation" / "retrieval_relevance_dev_sample.csv"
    output_metrics_json = repo_root / "outputs" / "retrieval" / "retrieval_metrics.json"
    output_report_md = repo_root / "outputs" / "retrieval" / "retrieval_evaluation_summary.md"

    print("=" * 60)
    print("PHASE 4: DEVELOPMENT RETRIEVAL EVALUATION")
    print("=" * 60)

    # 1. Validation Gate
    print(f"Validating human relevance annotation file: {annotated_csv_path}")
    try:
        df_annotated = validate_relevance_file(annotated_csv_path)
        print("Relevance annotation file passed validation successfully!")
    except RelevanceValidationError as e:
        print("\n" + "!" * 60)
        print("VALIDATION ERROR: Relevance annotation is incomplete or invalid!")
        print("Numerical retrieval metrics and threshold tuning are BLOCKED.")
        print("!" * 60)
        print(str(e))
        sys.exit(1)

    # 2. Compute Metrics & Calibrate Threshold
    print("\nComputing strict and relaxed retrieval metrics...")
    report = evaluate_retrieval_from_file(annotated_csv_path)

    # 3. Save JSON Metrics
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_json, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Saved metrics JSON to {output_metrics_json}")

    # 4. Print Summary Table
    strict_bm25 = report.strict_metrics["lexical_bm25"]
    strict_sem = report.strict_metrics["semantic"]
    relaxed_bm25 = report.relaxed_metrics["lexical_bm25"]
    relaxed_sem = report.relaxed_metrics["semantic"]

    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVALUATION RESULTS (N = {report.sample_size_queries} Dev Queries)")
    print("=" * 60)
    print("STRICT METRICS (Positive = 'relevant' only):")
    print(f"  BM25:     Hit@1={strict_bm25.hit_at_1:.3f} | Hit@3={strict_bm25.hit_at_3:.3f} | Prec@1={strict_bm25.precision_at_1:.3f} | Prec@3={strict_bm25.precision_at_3:.3f} | MRR={strict_bm25.mrr:.3f}")
    print(f"  Semantic: Hit@1={strict_sem.hit_at_1:.3f} | Hit@3={strict_sem.hit_at_3:.3f} | Prec@1={strict_sem.precision_at_1:.3f} | Prec@3={strict_sem.precision_at_3:.3f} | MRR={strict_sem.mrr:.3f}")
    print("-" * 60)
    print("RELAXED METRICS (Positive = 'relevant' OR 'partially_relevant'):")
    print(f"  BM25:     Hit@1={relaxed_bm25.hit_at_1:.3f} | Hit@3={relaxed_bm25.hit_at_3:.3f} | Prec@1={relaxed_bm25.precision_at_1:.3f} | Prec@3={relaxed_bm25.precision_at_3:.3f} | MRR={relaxed_bm25.mrr:.3f}")
    print(f"  Semantic: Hit@1={relaxed_sem.hit_at_1:.3f} | Hit@3={relaxed_sem.hit_at_3:.3f} | Prec@1={relaxed_sem.precision_at_1:.3f} | Prec@3={relaxed_sem.precision_at_3:.3f} | MRR={relaxed_sem.mrr:.3f}")
    print("-" * 60)
    print("EVIDENCE FILTER STATISTICS:")
    fstats = report.filter_stats
    print(f"  Total Evaluated: {fstats['total_evaluated_candidates']}")
    print(f"  Accepted:        {fstats['accepted_count']} ({fstats['acceptance_rate']*100:.1f}%)")
    print(f"  Rejected:        {fstats['rejected_count']} ({fstats['rejection_rate']*100:.1f}%)")
    print(f"  Calibrated Semantic Threshold: {report.calibrated_threshold:.2f}")
    print("=" * 60)

    # 5. Generate Markdown Report
    md_lines = [
        "# Phase 4 — Historical Resolution Retrieval Evaluation Report\n\n",
        "## Overview\n",
        f"- **Evaluation Sample**: {report.sample_size_queries} representative Development queries across 8 intents.\n",
        f"- **Corpus**: 15,850 historical Tesco resolutions (golden set strictly quarantined; per-query historical cutoffs applied).\n",
        "- **Ranking Field**: `sanitized_customer_issue` only (Tesco replies kept solely as evidence payload).\n",
        f"- **Calibrated Semantic Threshold**: `{report.calibrated_threshold:.2f}`\n\n",
        "## Performance Metrics\n\n",
        "### Strict Evaluation (Only 'relevant' counts as positive)\n",
        "| Retriever | Hit@1 | Hit@3 | Precision@1 | Precision@3 | MRR |\n",
        "|---|---|---|---|---|---|\n",
        f"| **Lexical (BM25)** | {strict_bm25.hit_at_1:.3f} | {strict_bm25.hit_at_3:.3f} | {strict_bm25.precision_at_1:.3f} | {strict_bm25.precision_at_3:.3f} | {strict_bm25.mrr:.3f} |\n",
        f"| **Semantic (Dense)** | {strict_sem.hit_at_1:.3f} | {strict_sem.hit_at_3:.3f} | {strict_sem.precision_at_1:.3f} | {strict_sem.precision_at_3:.3f} | {strict_sem.mrr:.3f} |\n\n",
        "### Relaxed Evaluation ('relevant' + 'partially_relevant' count as positive)\n",
        "| Retriever | Hit@1 | Hit@3 | Precision@1 | Precision@3 | MRR |\n",
        "|---|---|---|---|---|---|\n",
        f"| **Lexical (BM25)** | {relaxed_bm25.hit_at_1:.3f} | {relaxed_bm25.hit_at_3:.3f} | {relaxed_bm25.precision_at_1:.3f} | {relaxed_bm25.precision_at_3:.3f} | {relaxed_bm25.mrr:.3f} |\n",
        f"| **Semantic (Dense)** | {relaxed_sem.hit_at_1:.3f} | {relaxed_sem.hit_at_3:.3f} | {relaxed_sem.precision_at_1:.3f} | {relaxed_sem.precision_at_3:.3f} | {relaxed_sem.mrr:.3f} |\n\n",
        "## Evidence Filter Rejection Rates\n",
        f"- **Total Candidates Evaluated**: {fstats['total_evaluated_candidates']}\n",
        f"- **Accepted**: {fstats['accepted_count']} ({fstats['acceptance_rate']*100:.1f}%)\n",
        f"- **Rejected**: {fstats['rejected_count']} ({fstats['rejection_rate']*100:.1f}%)\n\n",
    ]

    with open(output_report_md, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"Exported evaluation markdown report to {output_report_md}")


if __name__ == "__main__":
    main()
