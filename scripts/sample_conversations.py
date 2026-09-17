#!/usr/bin/env python3
"""CLI utility to sample conversations from conversations.jsonl for human taxonomy review."""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hiver_tesco.models import (
    Direction,
    QualityFlags,
    ReconstructedConversation,
    ThreadMessage,
)
from hiver_tesco.sampler import (
    export_sample_jsonl,
    export_sample_markdown,
    sample_conversations_stratified,
)


def load_conversations_from_jsonl(file_path: Path) -> list:
    """Load reconstructed conversations from JSONL file."""
    conversations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            q_data = data["quality_flags"]
            flags = QualityFlags(
                has_missing_parent=q_data.get("has_missing_parent", False),
                has_cycle=q_data.get("has_cycle", False),
                has_duplicate_tweet_id=q_data.get("has_duplicate_tweet_id", False),
                has_cross_brand_mention=q_data.get("has_cross_brand_mention", False),
                is_outbound_root=q_data.get("is_outbound_root", False),
                is_multi_customer=q_data.get("is_multi_customer", False),
                is_branching=q_data.get("is_branching", False),
                has_empty_text=q_data.get("has_empty_text", False),
                is_ambiguous_ownership=q_data.get("is_ambiguous_ownership", False),
                missing_parent_ids=q_data.get("missing_parent_ids", []),
                cross_brand_handles=q_data.get("cross_brand_handles", []),
            )
            messages = [
                ThreadMessage(
                    tweet_id=m["tweet_id"],
                    parent_tweet_id=m["parent_tweet_id"],
                    child_tweet_ids=m["child_tweet_ids"],
                    author_id=m["author_id"],
                    direction=Direction(m["direction"]),
                    created_at=m["created_at"],
                    text=m["text"],
                    turn_index=m["turn_index"],
                    depth=m["depth"],
                    is_root=m["is_root"],
                )
                for m in data["messages"]
            ]
            conv = ReconstructedConversation(
                conversation_id=data["conversation_id"],
                source_root_tweet_id=data["source_root_tweet_id"],
                message_count=data["message_count"],
                inbound_count=data["inbound_count"],
                outbound_count=data["outbound_count"],
                customer_author_ids=data["customer_author_ids"],
                started_at=data["started_at"],
                ended_at=data["ended_at"],
                duration_seconds=data["duration_seconds"],
                quality_flags=flags,
                messages=messages,
                provenance=data.get("provenance", {}),
            )
            conversations.append(conv)
    return conversations


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("sample_conversations")

    parser = argparse.ArgumentParser(
        description="Sample reconstructed conversations for human review and taxonomy discovery."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("outputs/conversations.jsonl"),
        help="Path to reconstructed conversations.jsonl (default: outputs/conversations.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/samples"),
        help="Directory to save human review samples (default: outputs/samples)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of conversations to sample (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )

    args = parser.parse_args()

    if not args.input_jsonl.exists():
        logger.error(f"Input file does not exist: {args.input_jsonl}")
        logger.error("Run scripts/run_pipeline.py first to generate conversations.jsonl.")
        sys.exit(1)

    logger.info(f"Loading conversations from {args.input_jsonl}...")
    conversations = load_conversations_from_jsonl(args.input_jsonl)
    logger.info(f"Loaded {len(conversations):,} conversations.")

    sampled = sample_conversations_stratified(
        conversations=conversations,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / f"conversation_sample_{args.sample_size}.md"
    jsonl_path = args.output_dir / f"conversation_sample_{args.sample_size}.jsonl"

    export_sample_markdown(sampled, md_path, seed=args.seed)
    export_sample_jsonl(sampled, jsonl_path)

    logger.info(f"Successfully exported {len(sampled)} samples:")
    logger.info(f"  - Markdown Review Booklet: {md_path}")
    logger.info(f"  - JSONL Sample:            {jsonl_path}")


if __name__ == "__main__":
    main()
