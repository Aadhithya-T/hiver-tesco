"""Unit tests for safe dataset loading and schema reporting."""

import csv
from pathlib import Path
import pytest
from hiver_tesco.loader import load_dataset, parse_inbound_flag, redact_text_for_sample


def test_parse_inbound_flag():
    assert parse_inbound_flag("TRUE") is True
    assert parse_inbound_flag("True") is True
    assert parse_inbound_flag(True) is True
    assert parse_inbound_flag("1") is True
    assert parse_inbound_flag("FALSE") is False
    assert parse_inbound_flag(False) is False
    assert parse_inbound_flag("0") is False


def test_redact_text_for_sample():
    raw = "Hey @115881 thanks for reaching out to @Tesco"
    redacted = redact_text_for_sample(raw)
    assert "@[CUSTOMER_ID]" in redacted
    assert "@Tesco" in redacted
    assert "@115881" not in redacted


def test_load_dataset_valid(tmp_path: Path):
    csv_file = tmp_path / "valid.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tweet_id", "author_id", "inbound", "created_at",
            "text", "response_tweet_id", "in_response_to_tweet_id", "conversation_id"
        ])
        writer.writerow([
            "1", "100", "TRUE", "Tue Oct 31 18:00:00 +0000 2017",
            "Need help @Tesco", "2", "", "1"
        ])
        writer.writerow([
            "2", "Tesco", "FALSE", "Tue Oct 31 18:05:00 +0000 2017",
            "@100 Hello how can we help?", "", "1", "1"
        ])

    tweets, report, all_ids = load_dataset(csv_file)
    assert len(tweets) == 2
    assert report.total_rows == 2
    assert report.unique_tweet_ids == 2
    assert report.duplicate_tweet_ids == 0
    assert all_ids == {"1", "2"}
    assert report.null_counts["in_response_to_tweet_id"] == 1


def test_load_dataset_missing_column(tmp_path: Path):
    csv_file = tmp_path / "missing_col.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tweet_id", "author_id", "inbound"])  # Missing created_at, text, etc.
        writer.writerow(["1", "100", "TRUE"])

    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(csv_file)


def test_load_dataset_invalid_date(tmp_path: Path):
    csv_file = tmp_path / "bad_date.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tweet_id", "author_id", "inbound", "created_at",
            "text", "response_tweet_id", "in_response_to_tweet_id", "conversation_id"
        ])
        writer.writerow([
            "1", "100", "TRUE", "invalid-date-format",
            "Text", "", "", "1"
        ])

    with pytest.raises(ValueError, match="Invalid date format"):
        load_dataset(csv_file)
