"""Relevance annotation validation for retrieval evaluation.

Enforces that human relevance annotations are complete, use valid labels,
and possess all required identifiers before metrics can be computed.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd

ALLOWED_RELEVANCE_LABELS: Set[str] = {
    "relevant",
    "partially_relevant",
    "irrelevant",
    "not_relevant",
}

REQUIRED_COLUMNS: List[str] = [
    "sample_id",
    "query_id",
    "query_started_at",
    "query_text",
    "retriever_type",
    "rank",
    "candidate_id",
    "score",
    "filter_status",
    "candidate_customer_issue",
    "candidate_tesco_resolution",
    "human_relevance",
    "relevance_notes",
]


class RelevanceValidationError(Exception):
    """Raised when the human relevance annotation file fails validation."""
    pass


def validate_relevance_file(csv_path: Path) -> pd.DataFrame:
    """Validate that the relevance annotation CSV is complete and valid.

    Args:
        csv_path: Path to the annotated CSV file.

    Returns:
        Validated pandas DataFrame.

    Raises:
        RelevanceValidationError: If the file is missing, columns are missing,
        rows contain blank/placeholder labels, or invalid enums are used.
    """
    if not csv_path.exists():
        raise RelevanceValidationError(f"Relevance annotation file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Column presence check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise RelevanceValidationError(
            f"Missing required columns in {csv_path}: {missing_cols}"
        )

    if len(df) == 0:
        raise RelevanceValidationError(f"Annotation file is empty: {csv_path}")

    errors: List[str] = []

    # 2. Check for missing or placeholder human_relevance values
    for idx, row in df.iterrows():
        raw_label = row["human_relevance"]
        if pd.isna(raw_label) or str(raw_label).strip() == "" or str(raw_label).strip().lower() in ("todo", "unannotated", "none"):
            errors.append(
                f"Row {idx + 1} (Query {row['query_id']}, Candidate {row['candidate_id']}): "
                f"human_relevance is blank or placeholder."
            )
            continue

        clean_label = str(raw_label).strip().lower()
        if clean_label not in ALLOWED_RELEVANCE_LABELS:
            errors.append(
                f"Row {idx + 1} (Query {row['query_id']}): Invalid label '{clean_label}'. "
                f"Allowed: {sorted(list(ALLOWED_RELEVANCE_LABELS))}"
            )

    if errors:
        summary = "\n".join(errors[:10])
        total = len(errors)
        raise RelevanceValidationError(
            f"Relevance validation failed with {total} error(s):\n{summary}"
            + (f"\n...and {total - 10} more." if total > 10 else "")
        )

    return df
