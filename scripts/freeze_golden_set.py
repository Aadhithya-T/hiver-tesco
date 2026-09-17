#!/usr/bin/env python3
"""CLI utility to validate, hash, freeze, and stratify the human-labelled golden dataset."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List
from sklearn.model_selection import train_test_split

# Ensure src/ and root are on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from scripts.validate_annotations import load_annotation_records, validate_annotations


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s")


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def freeze_golden_dataset(
    input_path: Path,
    golden_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Validate, copy, hash, and partition the golden dataset into stratified 60/20/20 splits."""
    logger = logging.getLogger("freeze_golden_set")
    if not input_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {input_path}")

    # 1. Validate annotations
    logger.info(f"Validating annotations in {input_path}...")
    records = load_annotation_records(input_path)
    is_valid, summary, issues = validate_annotations(records)

    if not is_valid or summary["unlabelled_records"] > 0:
        raise ValueError(
            f"Cannot freeze incomplete or invalid annotations: {summary['unlabelled_records']} unlabelled, "
            f"{len(issues)} issues found."
        )

    logger.info("Validation passed! 100% complete and schema-valid.")

    # 2. Compute SHA256 checksum
    sha256_hash = compute_sha256(input_path)
    logger.info(f"Computed SHA-256: {sha256_hash}")

    golden_dir.mkdir(parents=True, exist_ok=True)
    frozen_csv_path = golden_dir / "golden_set_v1.0.csv"
    manifest_path = golden_dir / "golden_set_manifest.json"

    # 3. Copy frozen artifact
    shutil.copy2(input_path, frozen_csv_path)
    logger.info(f"Frozen dataset copied to {frozen_csv_path}")

    # 4. Generate deterministic stratified splits (60% train: 120, 20% dev: 40, 20% test: 40)
    intents = [r["primary_intent"] for r in records]

    # Split into 80% (train+dev: 160) and 20% (test: 40)
    train_dev_idx, test_idx, train_dev_y, test_y = train_test_split(
        list(range(len(records))),
        intents,
        test_size=0.20,
        random_state=seed,
        stratify=intents,
    )

    # Split train+dev into 60% train (120) and 20% dev (40)
    train_sub_idx, dev_sub_idx, train_y, dev_y = train_test_split(
        train_dev_idx,
        train_dev_y,
        test_size=0.25,  # 0.25 * 0.80 = 0.20
        random_state=seed,
        stratify=train_dev_y,
    )

    train_records = [records[i] for i in train_sub_idx]
    dev_records = [records[i] for i in dev_sub_idx]
    test_records = [records[i] for i in test_idx]

    train_ids = [r["conversation_id"] for r in train_records]
    dev_ids = [r["conversation_id"] for r in dev_records]
    test_ids = [r["conversation_id"] for r in test_records]

    splits_map = {}
    for cid in train_ids:
        splits_map[cid] = "train"
    for cid in dev_ids:
        splits_map[cid] = "dev"
    for cid in test_ids:
        splits_map[cid] = "test"

    splits_dir = golden_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "train_ids.json", "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)
    with open(splits_dir / "dev_ids.json", "w", encoding="utf-8") as f:
        json.dump(dev_ids, f, indent=2)
    with open(splits_dir / "test_ids.json", "w", encoding="utf-8") as f:
        json.dump(test_ids, f, indent=2)
    with open(splits_dir / "golden_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits_map, f, indent=2)

    # Compute per-split class support breakdown
    all_intents = sorted(list(summary["intent_distribution"].keys()))
    support_summary = {}
    for intent in all_intents:
        support_summary[intent] = {
            "total": sum(1 for r in records if r["primary_intent"] == intent),
            "train": sum(1 for r in train_records if r["primary_intent"] == intent),
            "dev": sum(1 for r in dev_records if r["primary_intent"] == intent),
            "test": sum(1 for r in test_records if r["primary_intent"] == intent),
        }

    with open(splits_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(support_summary, f, indent=2)

    # 5. Write manifest metadata
    manifest = {
        "artifact_name": "golden_set_v1.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "sha256": sha256_hash,
        "total_records": len(records),
        "taxonomy_version": "Draft Human Taxonomy v1.0",
        "annotator_metadata": {
            "number_of_reviewers": 1,
            "single_reviewer_disclosure": (
                "All 200 conversations were manually annotated by one reviewer. "
                "No inter-annotator agreement was measured."
            ),
            "adjudication_process": (
                "Single reviewer manual annotation adhering to Phase 2 guidelines. "
                "Ambiguities and multi-intent borderline cases noted in ambiguity_notes."
            ),
        },
        "intent_distribution": summary["intent_distribution"],
        "escalation_distribution": summary["escalation_distribution"],
        "rationale_distribution": summary["rationale_distribution"],
        "splits": {
            "split_ratio": "60/20/20",
            "seed": seed,
            "train_size": len(train_ids),
            "dev_size": len(dev_ids),
            "test_size": len(test_ids),
            "intent_support_per_split": support_summary,
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to {manifest_path}")
    logger.info(f"Splits written to {splits_dir}: Train={len(train_ids)}, Dev={len(dev_ids)}, Test={len(test_ids)}")

    return manifest


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Freeze golden dataset and create deterministic stratified splits.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/annotation/golden_candidates_200.csv"),
        help="Path to validated golden set CSV (default: data/annotation/golden_candidates_200.csv)",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("data/golden"),
        help="Destination directory for frozen artifacts (default: data/golden)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic stratified split (default: 42)",
    )

    args = parser.parse_args()
    try:
        manifest = freeze_golden_dataset(args.input_path, args.golden_dir, seed=args.seed)
        print("\n" + "=" * 65)
        print("GOLDEN SET SUCCESSFULLY FROZEN")
        print("=" * 65)
        print(f"Frozen File   : {args.golden_dir / 'golden_set_v1.0.csv'}")
        print(f"SHA-256       : {manifest['sha256']}")
        print(f"Total Records : {manifest['total_records']}")
        print(f"Splits        : Train={manifest['splits']['train_size']}, Dev={manifest['splits']['dev_size']}, Test={manifest['splits']['test_size']}")
        print("=" * 65)
    except Exception as e:
        sys.stderr.write(f"Error freezing golden dataset: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
