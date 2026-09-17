"""Deterministic diversity-oriented sampling for human taxonomy discovery."""

from pathlib import Path
import json
import logging
import random
from typing import Dict, List, Optional
from hiver_tesco.models import ReconstructedConversation

logger = logging.getLogger(__name__)


def sample_conversations_stratified(
    conversations: List[ReconstructedConversation],
    sample_size: int = 200,
    seed: int = 42,
) -> List[ReconstructedConversation]:
    """Sample conversations deterministically with diversity across length and thread types.

    Stratification tiers:
    - Tier 1: Short (2-3 messages) - simple queries, immediate resolutions (~35%)
    - Tier 2: Medium (4-6 messages) - multi-turn troubleshooting, DM routing (~40%)
    - Tier 3: Long (7+ messages) - complex disputes, delivery failures, extended interactions (~20%)
    - Special Slice: Cross-brand or outbound root threads (~5%)
    """
    if len(conversations) <= sample_size:
        logger.info(f"Total conversations ({len(conversations)}) <= sample_size ({sample_size}). Returning all.")
        return list(conversations)

    rng = random.Random(seed)

    # Sort deterministically by conversation_id first
    stable_convs = sorted(conversations, key=lambda c: c.conversation_id)

    tier_short: List[ReconstructedConversation] = []
    tier_medium: List[ReconstructedConversation] = []
    tier_long: List[ReconstructedConversation] = []
    tier_special: List[ReconstructedConversation] = []

    for c in stable_convs:
        # Check special interest flags first
        if c.quality_flags.has_cross_brand_mention or c.quality_flags.is_outbound_root:
            tier_special.append(c)
        elif c.message_count <= 3:
            tier_short.append(c)
        elif 4 <= c.message_count <= 6:
            tier_medium.append(c)
        else:
            tier_long.append(c)

    # Target counts
    n_special = min(len(tier_special), max(5, int(sample_size * 0.05)))
    n_short = min(len(tier_short), int(sample_size * 0.35))
    n_medium = min(len(tier_medium), int(sample_size * 0.40))
    n_long = min(len(tier_long), sample_size - (n_special + n_short + n_medium))

    sampled: List[ReconstructedConversation] = []
    sampled.extend(rng.sample(tier_special, n_special) if tier_special else [])
    sampled.extend(rng.sample(tier_short, n_short) if tier_short else [])
    sampled.extend(rng.sample(tier_medium, n_medium) if tier_medium else [])
    sampled.extend(rng.sample(tier_long, n_long) if tier_long else [])

    # If any shortfall remains, fill from remaining conversations
    if len(sampled) < sample_size:
        sampled_ids = {c.conversation_id for c in sampled}
        remaining = [c for c in stable_convs if c.conversation_id not in sampled_ids]
        shortfall = min(len(remaining), sample_size - len(sampled))
        sampled.extend(rng.sample(remaining, shortfall))

    # Sort final sample deterministically by conversation_id
    sampled.sort(key=lambda c: (c.message_count, c.conversation_id))
    return sampled


def export_sample_markdown(
    conversations: List[ReconstructedConversation],
    output_path: Path,
    seed: int = 42,
) -> None:
    """Export a human-readable Markdown review booklet for manual taxonomy discovery."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Tesco Customer Support Conversation Sample (Human Review Booklet)\n\n")
        f.write(f"- **Total Sampled Conversations**: {len(conversations)}\n")
        f.write(f"- **Sampling Strategy**: Stratified by length (Short: 2-3, Medium: 4-6, Long: 7+) with special slices (cross-brand/broadcast)\n")
        f.write(f"- **Reproducibility Seed**: {seed}\n")
        f.write("- **Purpose**: Phase 1 inspection for inductive taxonomy and intent discovery.\n\n")
        f.write("---\n\n")

        for idx, conv in enumerate(conversations, 1):
            flags = conv.quality_flags
            f.write(f"## [{idx}/{len(conversations)}] Conversation ID: `{conv.conversation_id}`\n\n")
            f.write(f"- **Messages**: {conv.message_count} (Inbound: {conv.inbound_count}, Outbound: {conv.outbound_count})\n")
            f.write(f"- **Timeline**: {conv.started_at} -> {conv.ended_at} (Duration: {int(conv.duration_seconds)}s)\n")
            f.write(f"- **Customer Authors**: `{', '.join(conv.customer_author_ids)}`\n")

            flag_items = []
            if flags.is_clean:
                flag_items.append("Clean")
            if flags.has_missing_parent:
                flag_items.append("Missing Parent")
            if flags.has_cross_brand_mention:
                flag_items.append(f"Cross-Brand ({', '.join(flags.cross_brand_handles)})")
            if flags.is_outbound_root:
                flag_items.append("Outbound Root (Broadcast/Proactive)")
            if flags.is_multi_customer:
                flag_items.append("Multi-Customer")
            if flags.is_branching:
                flag_items.append("Branching Replies")
            f.write(f"- **Flags**: {', '.join(flag_items)}\n\n")

            f.write("### Thread Dialogue:\n\n")
            for msg in conv.messages:
                direction_label = "[CUSTOMER (Inbound)]" if msg.direction.value == "inbound" else "[TESCO (Outbound)]"
                reply_to = f" | Reply to: `{msg.parent_tweet_id}`" if msg.parent_tweet_id else " | Root Tweet"
                f.write(f"> **{direction_label}** -- Author: `{msg.author_id}` -- `{msg.created_at}`{reply_to}\n>\n")
                # Quote each line of text
                for line in msg.text.splitlines():
                    f.write(f"> {line}\n")
                f.write("\n")

            f.write("#### Human Reviewer Annotations:\n")
            f.write("- [ ] **Customer Primary Intent**: `____________________________________`\n")
            f.write("- [ ] **Secondary Topic / Entity**: `____________________________________`\n")
            f.write("- [ ] **Resolution Status**: `[ ] Resolved In-Thread  [ ] Escalated to DM  [ ] Unresolved  [ ] Off-topic`\n")
            f.write("- [ ] **Notes**: `____________________________________`\n\n")
            f.write("---\n\n")

    logger.info(f"Exported human review Markdown booklet to {output_path}")


def export_sample_jsonl(
    conversations: List[ReconstructedConversation],
    output_path: Path,
) -> None:
    """Export the sampled conversations to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"Exported sampled conversations JSONL to {output_path}")
