#!/usr/bin/env python3
"""Script to select a deterministic, stratified 150-250 candidate conversation set for Phase 2 golden annotation."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hiver_tesco.config import PipelineConfig


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s")


def format_dialogue_for_csv(messages: List[Dict[str, Any]]) -> str:
    """Format full turn-by-turn conversation messages into a clean multi-line text block for spreadsheet viewing."""
    lines = []
    for msg in messages:
        direction = "CUSTOMER" if msg.get("direction") == "inbound" else "TESCO"
        turn = msg.get("turn_index", 0)
        author = msg.get("author_id", "")
        ts = msg.get("created_at", "")[:19].replace("T", " ")
        parent = f" (in reply to {msg.get('parent_tweet_id')})" if msg.get("parent_tweet_id") else " (root)"
        header = f"[Turn {turn} | {direction} | @{author} | {ts}]{parent}:"
        text = msg.get("text", "").strip()
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def select_stratified_candidates(
    conversations: List[Dict[str, Any]],
    sample_size: int = 200,
    seed: int = 100,
) -> List[Dict[str, Any]]:
    """Deterministically select a stratified candidate set with strictly disjoint special and length tiers.

    Guarantees:
    - Exactly sample_size items returned (if dataset allows).
    - Special slice (cross-brand/broadcast) is strictly disjoint from length tiers.
    - Deterministic order and reproducible selection via seed.
    """
    rng = random.Random(seed)

    # Sort conversations by conversation_id first for initial stability
    sorted_convs = sorted(conversations, key=lambda c: c["conversation_id"])

    # 1. Partition into strictly disjoint pools
    pool_special: List[Dict[str, Any]] = []
    pool_short: List[Dict[str, Any]] = []
    pool_medium: List[Dict[str, Any]] = []
    pool_long: List[Dict[str, Any]] = []

    for c in sorted_convs:
        flags = c.get("quality_flags", {})
        # Special slice condition: cross-brand mention OR outbound root
        if flags.get("has_cross_brand_mention") or flags.get("is_outbound_root"):
            pool_special.append(c)
        elif c.get("message_count", 0) <= 3:
            pool_short.append(c)
        elif 4 <= c.get("message_count", 0) <= 6:
            pool_medium.append(c)
        else:
            pool_long.append(c)

    # 2. Target allocations (summing to sample_size)
    n_special = min(len(pool_special), max(5, int(sample_size * 0.05)))   # 10 for size 200 (5%)
    n_short = min(len(pool_short), int(sample_size * 0.35))                # 70 for size 200 (35%)
    n_medium = min(len(pool_medium), int(sample_size * 0.40))             # 80 for size 200 (40%)
    n_long = min(len(pool_long), sample_size - (n_special + n_short + n_medium)) # 40 for size 200 (20%)

    # 3. Sample from each disjoint pool
    sampled: List[Dict[str, Any]] = []
    sampled.extend(rng.sample(pool_special, n_special) if pool_special else [])
    sampled.extend(rng.sample(pool_short, n_short) if pool_short else [])
    sampled.extend(rng.sample(pool_medium, n_medium) if pool_medium else [])
    sampled.extend(rng.sample(pool_long, n_long) if pool_long else [])

    # Handle any shortfall
    if len(sampled) < sample_size:
        selected_ids = {c["conversation_id"] for c in sampled}
        remaining = [c for c in sorted_convs if c["conversation_id"] not in selected_ids]
        shortfall = min(len(remaining), sample_size - len(sampled))
        sampled.extend(rng.sample(remaining, shortfall))

    # Sort final candidates deterministically by length then conversation_id
    sampled.sort(key=lambda c: (c.get("message_count", 0), c["conversation_id"]))
    return sampled


def write_csv_template(candidates: List[Dict[str, Any]], output_path: Path) -> None:
    """Write human-editable CSV template with empty annotation fields."""
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "candidate_idx",
        "conversation_id",
        "message_count",
        "duration_minutes",
        "flags",
        "dialogue_text",
        # Human Annotation Fields (Empty)
        "primary_intent",
        "escalation_needed",
        "escalation_rationale",
        "evidence_notes",
        "ambiguity_notes",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for idx, conv in enumerate(candidates, 1):
            flags = conv.get("quality_flags", {})
            flag_items = []
            if flags.get("is_clean", True):
                flag_items.append("Clean")
            if flags.get("has_cross_brand_mention"):
                flag_items.append(f"Cross-Brand ({','.join(flags.get('cross_brand_handles', []))})")
            if flags.get("is_outbound_root"):
                flag_items.append("Outbound-Root")
            if flags.get("is_branching"):
                flag_items.append("Branching")
            if flags.get("is_multi_customer"):
                flag_items.append("Multi-Customer")
            if flags.get("has_missing_parent"):
                flag_items.append("Missing-Parent")

            flag_str = "; ".join(flag_items) if flag_items else "Clean"
            duration_mins = round(conv.get("duration_seconds", 0) / 60.0, 1)

            writer.writerow({
                "candidate_idx": idx,
                "conversation_id": conv["conversation_id"],
                "message_count": conv.get("message_count", len(conv.get("messages", []))),
                "duration_minutes": duration_mins,
                "flags": flag_str,
                "dialogue_text": format_dialogue_for_csv(conv.get("messages", [])),
                "primary_intent": "",
                "escalation_needed": "",
                "escalation_rationale": "",
                "evidence_notes": "",
                "ambiguity_notes": "",
            })


def write_jsonl_template(candidates: List[Dict[str, Any]], output_path: Path) -> None:
    """Write JSONL template with full conversation records and empty annotation objects."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, conv in enumerate(candidates, 1):
            record = dict(conv)
            record["candidate_idx"] = idx
            record["annotation"] = {
                "primary_intent": None,
                "escalation_needed": None,
                "escalation_rationale": None,
                "evidence_notes": None,
                "ambiguity_notes": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    setup_logging()
    logger = logging.getLogger("create_golden_candidate_set")

    parser = argparse.ArgumentParser(description="Create deterministic 150-250 candidate set for golden annotation.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("outputs/conversations.jsonl"),
        help="Path to Phase 1 reconstructed conversations.jsonl (default: outputs/conversations.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotation"),
        help="Directory to save golden candidate templates (default: data/annotation)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Number of candidates to sample (default: 200, range: 150-250)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Deterministic random seed (default: 100, distinct from Phase 1 seed 42 to preserve held-out sets)",
    )

    args = parser.parse_args()

    if not args.input_jsonl.exists():
        logger.error(f"Input file not found: {args.input_jsonl}")
        logger.error("Please run scripts/run_pipeline.py first.")
        sys.exit(1)

    logger.info(f"Loading conversations from {args.input_jsonl}...")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        conversations = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(conversations):,} conversations.")

    logger.info(f"Selecting {args.size} stratified candidates (seed={args.seed})...")
    candidates = select_stratified_candidates(conversations, sample_size=args.size, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"golden_candidates_{args.size}.csv"
    jsonl_path = args.output_dir / f"golden_candidates_{args.size}.jsonl"
    manifest_path = args.output_dir / "golden_candidate_manifest.json"

    logger.info(f"Writing CSV template to {csv_path}...")
    write_csv_template(candidates, csv_path)

    logger.info(f"Writing JSONL template to {jsonl_path}...")
    write_jsonl_template(candidates, jsonl_path)

    # Count breakdown
    special_count = sum(
        1 for c in candidates
        if c.get("quality_flags", {}).get("has_cross_brand_mention")
        or c.get("quality_flags", {}).get("is_outbound_root")
    )
    short_count = sum(
        1 for c in candidates
        if not (c.get("quality_flags", {}).get("has_cross_brand_mention") or c.get("quality_flags", {}).get("is_outbound_root"))
        and c.get("message_count", 0) <= 3
    )
    med_count = sum(
        1 for c in candidates
        if not (c.get("quality_flags", {}).get("has_cross_brand_mention") or c.get("quality_flags", {}).get("is_outbound_root"))
        and 4 <= c.get("message_count", 0) <= 6
    )
    long_count = sum(
        1 for c in candidates
        if not (c.get("quality_flags", {}).get("has_cross_brand_mention") or c.get("quality_flags", {}).get("is_outbound_root"))
        and c.get("message_count", 0) >= 7
    )

    manifest = {
        "candidate_set_name": f"golden_candidates_{args.size}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(candidates),
        "source_dataset_total_conversations": len(conversations),
        "seed": args.seed,
        "sampling_stratification": {
            "special_slice_cross_brand_or_broadcast": special_count,
            "short_tier_2_to_3_msgs": short_count,
            "medium_tier_4_to_6_msgs": med_count,
            "long_tier_7_plus_msgs": long_count,
        },
        "taxonomy_version": "Draft Human Taxonomy v1.0",
        "guidelines_document": "docs/annotation_guidelines.md",
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Successfully generated {len(candidates)} candidates:")
    logger.info(f"  - Special slice (cross-brand/broadcast): {special_count}")
    logger.info(f"  - Short tier (2-3 msgs):                {short_count}")
    logger.info(f"  - Medium tier (4-6 msgs):               {med_count}")
    logger.info(f"  - Long tier (7+ msgs):                  {long_count}")
    logger.info(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
