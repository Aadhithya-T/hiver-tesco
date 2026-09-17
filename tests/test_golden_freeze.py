"""Unit tests for Phase 3 golden set freezing, checksums, and stratified splits."""

import json
from pathlib import Path
import pytest
from scripts.freeze_golden_set import compute_sha256, freeze_golden_dataset


def test_freeze_golden_dataset(tmp_path: Path):
    # Use real annotation file as input
    input_csv = Path("data/annotation/golden_candidates_200.csv")
    assert input_csv.exists()

    golden_dir = tmp_path / "golden"
    manifest = freeze_golden_dataset(input_csv, golden_dir, seed=42)

    # 1. Verify frozen file exists and matches SHA-256
    frozen_csv = golden_dir / "golden_set_v1.0.csv"
    assert frozen_csv.exists()
    assert compute_sha256(frozen_csv) == manifest["sha256"]
    assert manifest["sha256"] == "1d6c44e24344c98a3446fa22faa534e33f91d71ad233013c3fc73c8a933f51a5"

    # 2. Verify single-reviewer disclosure in manifest
    assert manifest["annotator_metadata"]["number_of_reviewers"] == 1
    assert "one reviewer" in manifest["annotator_metadata"]["single_reviewer_disclosure"].lower()

    # 3. Verify splits: 120 train, 40 dev, 40 test
    splits = manifest["splits"]
    assert splits["train_size"] == 120
    assert splits["dev_size"] == 40
    assert splits["test_size"] == 40
    assert splits["train_size"] + splits["dev_size"] + splits["test_size"] == 200

    # 4. Verify mutual exclusivity and collective exhaustiveness
    with open(golden_dir / "splits" / "train_ids.json") as f:
        train_ids = set(json.load(f))
    with open(golden_dir / "splits" / "dev_ids.json") as f:
        dev_ids = set(json.load(f))
    with open(golden_dir / "splits" / "test_ids.json") as f:
        test_ids = set(json.load(f))

    assert len(train_ids) == 120
    assert len(dev_ids) == 40
    assert len(test_ids) == 40
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(test_ids)
    assert dev_ids.isdisjoint(test_ids)
    assert len(train_ids | dev_ids | test_ids) == 200

    # 5. Verify rare class preservation in all splits
    support = splits["intent_support_per_split"]
    assert "clubcard_and_loyalty" in support
    assert support["clubcard_and_loyalty"]["train"] >= 1
    assert support["clubcard_and_loyalty"]["dev"] >= 1
    assert support["clubcard_and_loyalty"]["test"] >= 1
    assert support["pricing_promotions_and_vouchers"]["train"] >= 1
    assert support["pricing_promotions_and_vouchers"]["dev"] >= 1
    assert support["pricing_promotions_and_vouchers"]["test"] >= 1
