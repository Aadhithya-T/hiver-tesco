#!/usr/bin/env python3
"""CLI utility to inspect dataset schema, nulls, cardinality, and sample rows."""

import argparse
from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hiver_tesco.loader import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Inspect TWCS/Tesco dataset schema and data quality."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("tesco_tweets.csv"),
        help="Path to the input CSV file (default: tesco_tweets.csv)",
    )
    args = parser.parse_args()

    try:
        _, schema_report, _ = load_dataset(args.input_path)
        schema_report.print_summary(sys.stdout)
    except Exception as e:
        sys.stderr.write(f"Error inspecting dataset: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
