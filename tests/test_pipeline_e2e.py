"""End-to-end integration tests and idempotency verification for the pipeline."""

import csv
import hashlib
from pathlib import Path
from hiver_tesco.config import PipelineConfig
from hiver_tesco.pipeline import run_extraction_pipeline


def create_synthetic_dataset_csv(file_path: Path):
    """Create a synthetic CSV with diverse scenarios."""
    rows = [
        # Conversation 1: Clean linear
        ["101", "cust1", "TRUE", "Tue Oct 31 10:00:00 +0000 2017", "@Tesco Where is my order?", "", "", "101"],
        ["102", "Tesco", "FALSE", "Tue Oct 31 10:05:00 +0000 2017", "@cust1 Hi, please check DM.", "", "101", "101"],
        ["103", "cust1", "TRUE", "Tue Oct 31 10:10:00 +0000 2017", "@Tesco Sent details.", "", "102", "101"],

        # Conversation 2: Branching
        ["201", "cust2", "TRUE", "Tue Oct 31 11:00:00 +0000 2017", "@Tesco Damaged item!", "", "", "201"],
        ["202", "Tesco", "FALSE", "Tue Oct 31 11:05:00 +0000 2017", "@cust2 Sorry! Part 1/2", "", "201", "201"],
        ["203", "Tesco", "FALSE", "Tue Oct 31 11:05:01 +0000 2017", "@cust2 Send barcode Part 2/2", "", "201", "201"],

        # Conversation 3: Cross-brand
        ["301", "cust3", "TRUE", "Tue Oct 31 12:00:00 +0000 2017", "@Tesco Better than @sainsburys!", "", "", "301"],
        ["302", "Tesco", "FALSE", "Tue Oct 31 12:05:00 +0000 2017", "@cust3 Glad to hear!", "", "301", "301"],

        # Conversation 4: Missing parent (400 is not in dataset)
        ["401", "Tesco", "FALSE", "Tue Oct 31 13:00:00 +0000 2017", "@cust4 We refunded you.", "", "400", "400"],
        ["402", "cust4", "TRUE", "Tue Oct 31 13:05:00 +0000 2017", "@Tesco Thanks!", "", "401", "400"],

        # Conversation 5: Non-Tesco conversation (AppleSupport) -> should be excluded by require_brand_participation!
        ["501", "cust5", "TRUE", "Tue Oct 31 14:00:00 +0000 2017", "@AppleSupport help with iPhone", "", "", "501"],
        ["502", "AppleSupport", "FALSE", "Tue Oct 31 14:05:00 +0000 2017", "@cust5 Please restart.", "", "501", "501"],
    ]

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tweet_id", "author_id", "inbound", "created_at",
            "text", "response_tweet_id", "in_response_to_tweet_id", "conversation_id"
        ])
        writer.writerows(rows)


def hash_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_pipeline_e2e_and_idempotence(tmp_path: Path):
    input_csv = tmp_path / "synthetic_tweets.csv"
    create_synthetic_dataset_csv(input_csv)

    out_dir_1 = tmp_path / "run_1"
    config_1 = PipelineConfig(
        input_path=input_csv,
        output_dir=out_dir_1,
        brand_id="Tesco",
        seed=42,
        sample_size=3,
    )

    convs_1, audit_1, schema_1 = run_extraction_pipeline(config_1)

    # 1. Verification of funnel and brand filtering
    assert schema_1.total_rows == 12
    assert audit_1.total_conversations_discovered == 5
    # Conv 5 (AppleSupport) has no Tesco participation -> excluded!
    assert audit_1.conversations_with_brand_participation == 4
    assert len(convs_1) == 4

    # 2. Output files exist
    assert config_1.get_conversations_path().exists()
    assert config_1.get_audit_json_path().exists()
    assert config_1.get_audit_csv_path().exists()
    assert (config_1.get_samples_dir() / "conversation_sample_3.md").exists()
    assert (config_1.get_samples_dir() / "conversation_sample_3.jsonl").exists()

    # 3. Quality flag assertions
    conv_by_id = {c.conversation_id: c for c in convs_1}
    assert conv_by_id["101"].quality_flags.is_clean is True
    assert conv_by_id["201"].quality_flags.is_branching is True
    assert conv_by_id["301"].quality_flags.has_cross_brand_mention is True
    assert conv_by_id["400"].quality_flags.has_missing_parent is True

    # 4. Idempotency test: Run a second time with same inputs to run_2
    out_dir_2 = tmp_path / "run_2"
    config_2 = PipelineConfig(
        input_path=input_csv,
        output_dir=out_dir_2,
        brand_id="Tesco",
        seed=42,
        sample_size=3,
    )
    run_extraction_pipeline(config_2)

    # Compare SHA256 hashes of conversations.jsonl and audit_report.json
    hash_jsonl_1 = hash_file(config_1.get_conversations_path())
    hash_jsonl_2 = hash_file(config_2.get_conversations_path())
    assert hash_jsonl_1 == hash_jsonl_2

    hash_audit_1 = hash_file(config_1.get_audit_json_path())
    hash_audit_2 = hash_file(config_2.get_audit_json_path())
    assert hash_audit_1 == hash_audit_2
