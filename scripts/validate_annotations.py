#!/usr/bin/env python3
"""Validation script to check completeness, schema validity, and PII safety of human annotations."""

import argparse
import csv
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Set, Tuple

VALID_PRIMARY_INTENTS: Set[str] = {
    "delivery_and_orders",
    "product_quality_and_safety",
    "stock_and_availability",
    "store_experience_and_staff",
    "pricing_promotions_and_vouchers",
    "clubcard_and_loyalty",
    "website_and_app_technical",
    "general_feedback_and_chitchat",
}

VALID_ESCALATION_VALUES: Set[str] = {
    "escalate",
    "do_not_escalate",
}

VALID_ESCALATION_RATIONALES: Set[str] = {
    "requires_pii_or_dm",
    "refund_or_compensation",
    "product_safety_investigation",
    "formal_complaint",
    "technical_support",
    "none",
}

# PII Detection Patterns for evidence_notes safeguard
UK_POSTCODE_REGEX = re.compile(r"\b[A-Z]{1,2}[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}\b", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?<!\w)(?:\+44|0)\s?(?:7\d{3}|\d{4}|\d{3})\s?\d{3}\s?\d{3,4}\b")
RAW_DIGITS_REGEX = re.compile(r"(?<!\[)(?<!#)\b\d{7,}\b(?!\])")


def check_pii_in_evidence_notes(text: str) -> List[str]:
    """Check for leaked PII in evidence_notes."""
    violations = []
    if UK_POSTCODE_REGEX.search(text):
        violations.append("Possible UK residential postcode detected")
    if EMAIL_REGEX.search(text):
        violations.append("Email address detected")
    if PHONE_REGEX.search(text):
        violations.append("Telephone/mobile number detected")
    if RAW_DIGITS_REGEX.search(text):
        violations.append("Raw 7+ digit number (possible order/account ID) detected; please redact")
    return violations


def load_annotation_records(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from either CSV or JSONL."""
    records = []
    if file_path.suffix.lower() == ".csv":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                records.append(dict(r))
    elif file_path.suffix.lower() in {".jsonl", ".json"}:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    anno = data.get("annotation", {})
                    records.append({
                        "conversation_id": data.get("conversation_id"),
                        "primary_intent": anno.get("primary_intent") or "",
                        "escalation_needed": anno.get("escalation_needed") or "",
                        "escalation_rationale": anno.get("escalation_rationale") or "",
                        "evidence_notes": anno.get("evidence_notes") or "",
                        "ambiguity_notes": anno.get("ambiguity_notes") or "",
                    })
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Expected .csv or .jsonl")
    return records


def validate_annotations(records: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Validate annotation records against schema, taxonomy labels, and PII safety rules."""
    errors: List[str] = []
    warnings: List[str] = []

    seen_conv_ids: Set[str] = set()
    total_records = len(records)
    labelled_records = 0

    intent_counts: Dict[str, int] = {k: 0 for k in VALID_PRIMARY_INTENTS}
    escalation_counts: Dict[str, int] = {"escalate": 0, "do_not_escalate": 0}
    rationale_counts: Dict[str, int] = {k: 0 for k in VALID_ESCALATION_RATIONALES}

    for idx, r in enumerate(records, 1):
        cid = str(r.get("conversation_id", "")).strip()
        if not cid:
            errors.append(f"Row {idx}: Missing conversation_id.")
            continue

        if cid in seen_conv_ids:
            errors.append(f"Row {idx}: Duplicate conversation_id '{cid}'.")
        seen_conv_ids.add(cid)

        intent = str(r.get("primary_intent", "")).strip().lower()
        escalate = str(r.get("escalation_needed", "")).strip().lower()
        rationale = str(r.get("escalation_rationale", "")).strip().lower()
        evidence = str(r.get("evidence_notes", "")).strip()

        # Check if row is annotated
        if not intent and not escalate:
            # Unlabelled row
            continue

        labelled_records += 1

        # 1. Primary Intent Validation
        if intent not in VALID_PRIMARY_INTENTS:
            errors.append(
                f"Row {idx} (conv {cid}): Invalid primary_intent '{intent}'. "
                f"Must be one of {sorted(list(VALID_PRIMARY_INTENTS))}."
            )
        else:
            intent_counts[intent] += 1

        # 2. Escalation Needed Validation
        if escalate not in VALID_ESCALATION_VALUES:
            errors.append(
                f"Row {idx} (conv {cid}): Invalid escalation_needed '{escalate}'. "
                f"Must be 'escalate' or 'do_not_escalate'."
            )
        else:
            escalation_counts[escalate] += 1

        # 3. Escalation Rationale Consistency
        if escalate == "escalate":
            if not rationale or rationale == "none":
                errors.append(
                    f"Row {idx} (conv {cid}): escalation_rationale is required when escalation_needed is 'escalate'."
                )
            elif rationale not in VALID_ESCALATION_RATIONALES:
                errors.append(
                    f"Row {idx} (conv {cid}): Invalid escalation_rationale '{rationale}'. "
                    f"Must be one of {sorted(list(VALID_ESCALATION_RATIONALES))}."
                )
            else:
                rationale_counts[rationale] += 1
        elif escalate == "do_not_escalate":
            if rationale and rationale != "none":
                warnings.append(
                    f"Row {idx} (conv {cid}): escalation_rationale specified '{rationale}' but escalation_needed is 'do_not_escalate'."
                )
            rationale_counts["none"] += 1

        # 4. PII Check in evidence_notes
        if evidence:
            pii_issues = check_pii_in_evidence_notes(evidence)
            for issue in pii_issues:
                errors.append(f"Row {idx} (conv {cid}): PII Violation in evidence_notes: {issue}.")

    summary = {
        "total_records": total_records,
        "labelled_records": labelled_records,
        "unlabelled_records": total_records - labelled_records,
        "completion_rate_pct": round((labelled_records / total_records * 100), 1) if total_records > 0 else 0.0,
        "intent_distribution": {k: v for k, v in intent_counts.items() if v > 0},
        "escalation_distribution": escalation_counts,
        "rationale_distribution": {k: v for k, v in rationale_counts.items() if v > 0},
        "warnings_count": len(warnings),
        "errors_count": len(errors),
    }

    is_valid = len(errors) == 0
    return is_valid, summary, errors + warnings


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("validate_annotations")

    parser = argparse.ArgumentParser(description="Validate human annotation files for Phase 2 golden set.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/annotation/golden_candidates_200.csv"),
        help="Path to CSV or JSONL annotation file (default: data/annotation/golden_candidates_200.csv)",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        logger.error(f"File not found: {args.input_path}")
        sys.exit(1)

    logger.info(f"Validating annotations in {args.input_path}...")
    records = load_annotation_records(args.input_path)

    is_valid, summary, issues = validate_annotations(records)

    print("\n" + "=" * 65)
    print("ANNOTATION VALIDATION REPORT")
    print("=" * 65)
    print(f"File Path           : {args.input_path}")
    print(f"Total Candidates    : {summary['total_records']}")
    print(f"Labelled Records    : {summary['labelled_records']} ({summary['completion_rate_pct']}%)")
    print(f"Unlabelled Records  : {summary['unlabelled_records']}")
    print("-" * 65)

    if summary["labelled_records"] > 0:
        print("Intent Distribution:")
        for intent, cnt in sorted(summary["intent_distribution"].items(), key=lambda x: -x[1]):
            print(f"  - {intent:<35}: {cnt:>3}")
        print("\nEscalation Distribution:")
        print(f"  - Escalate        : {summary['escalation_distribution']['escalate']}")
        print(f"  - Do Not Escalate : {summary['escalation_distribution']['do_not_escalate']}")
        print("\nEscalation Rationales:")
        for rat, cnt in sorted(summary["rationale_distribution"].items(), key=lambda x: -x[1]):
            print(f"  - {rat:<30}: {cnt:>3}")
        print("-" * 65)

    if issues:
        print(f"\nIssues Detected ({len(issues)} total):")
        for iss in issues[:15]:
            print(f"  [!] {iss}")
        if len(issues) > 15:
            print(f"  ... and {len(issues) - 15} more issues.")
        print("-" * 65)

    if is_valid:
        if summary["unlabelled_records"] == 0:
            print("[SUCCESS] Golden dataset is COMPLETE and 100% schema-valid!")
        else:
            print("[INFO] Template structure is schema-valid! (Awaiting manual human annotation).")
        sys.exit(0)
    else:
        print("[FAIL] Validation failed with errors. Please fix above issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
