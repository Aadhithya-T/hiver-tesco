#!/usr/bin/env python3
"""Main entry point to run the Phase 1 Tesco conversation extraction pipeline."""

import argparse
import logging
from pathlib import Path
import sys
import time

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hiver_tesco.config import DEFAULT_BRAND_ID, PipelineConfig
from hiver_tesco.pipeline import run_extraction_pipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    setup_logging()
    logger = logging.getLogger("run_pipeline")

    parser = argparse.ArgumentParser(
        description="Extract, reconstruct, and audit Tesco customer support conversations."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("tesco_tweets.csv"),
        help="Path to the raw Tesco tweets CSV (default: tesco_tweets.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save generated artifacts (default: outputs)",
    )
    parser.add_argument(
        "--brand-id",
        type=str,
        default=DEFAULT_BRAND_ID,
        help=f"Account identifier for brand (default: {DEFAULT_BRAND_ID})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and ordering (default: 42)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of conversations to sample for human taxonomy review (default: 200)",
    )

    args = parser.parse_args()

    config = PipelineConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        brand_id=args.brand_id,
        seed=args.seed,
        sample_size=args.sample_size,
    )

    start_time = time.time()
    logger.info(f"Starting Phase 1 extraction pipeline with input={config.input_path}, brand={config.brand_id}")

    try:
        convs, audit, schema = run_extraction_pipeline(config)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed:.2f}s!")
    logger.info(f"Reconstructed conversations: {len(convs):,}")
    logger.info(f"Clean conversations: {audit.clean_conversations:,} ({audit.clean_conversations/len(convs)*100:.1f}%)")
    logger.info(f"Flagged conversations: {audit.flagged_conversations:,} ({audit.flagged_conversations/len(convs)*100:.1f}%)")
    logger.info(f"Output files:")
    logger.info(f"  - Dataset JSONL: {config.get_conversations_path()}")
    logger.info(f"  - Audit JSON:    {config.get_audit_json_path()}")
    logger.info(f"  - Audit CSV:     {config.get_audit_csv_path()}")
    if config.sample_size > 0:
        logger.info(f"  - Human Samples: {config.get_samples_dir()}")


if __name__ == "__main__":
    main()
