"""Pipeline orchestrator for conversation extraction, reconstruction, and auditing."""

from collections import defaultdict
from pathlib import Path
import json
import logging
from typing import Dict, List, Set, Tuple

from hiver_tesco.audit import AuditReport
from hiver_tesco.config import PipelineConfig
from hiver_tesco.graph import reconstruct_conversation
from hiver_tesco.loader import load_dataset, SchemaReport
from hiver_tesco.models import RawTweet, ReconstructedConversation
from hiver_tesco.sampler import (
    export_sample_jsonl,
    export_sample_markdown,
    sample_conversations_stratified,
)

logger = logging.getLogger(__name__)


def run_extraction_pipeline(
    config: PipelineConfig,
) -> Tuple[List[ReconstructedConversation], AuditReport, SchemaReport]:
    """Execute the end-to-end extraction, reconstruction, and auditing pipeline."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate raw dataset
    raw_tweets, schema_report, all_tweet_ids = load_dataset(config.input_path)

    audit = AuditReport(
        total_input_rows=schema_report.total_rows,
        unique_tweets=schema_report.unique_tweet_ids,
        duplicate_tweet_ids=schema_report.duplicate_tweet_ids,
        brand_id=config.brand_id,
        seed=config.seed,
    )

    # 2. Group tweets by conversation_id
    grouped_tweets: Dict[str, List[RawTweet]] = defaultdict(list)
    for tweet in raw_tweets:
        grouped_tweets[tweet.conversation_id].append(tweet)

    audit.total_conversations_discovered = len(grouped_tweets)

    # 3. Filter and reconstruct conversations
    brand_lower = config.brand_id.strip().lower()
    reconstructed_conversations: List[ReconstructedConversation] = []

    for conv_id, tweets in grouped_tweets.items():
        # Check brand participation
        has_brand = any(t.author_id.strip().lower() == brand_lower for t in tweets)
        if config.require_brand_participation and not has_brand:
            continue

        audit.conversations_with_brand_participation += 1

        # Check for duplicate tweet IDs inside this single thread
        thread_tids = [t.tweet_id for t in tweets]
        thread_has_dup = len(thread_tids) != len(set(thread_tids))

        reconstructed = reconstruct_conversation(
            conversation_id=conv_id,
            tweets=tweets,
            known_global_tweet_ids=all_tweet_ids,
            brand_id=config.brand_id,
            thread_has_duplicate_id=thread_has_dup,
        )

        reconstructed_conversations.append(reconstructed)
        audit.record_conversation(reconstructed)

    # 4. Sort conversations deterministically by (started_at, conversation_id)
    reconstructed_conversations.sort(key=lambda c: (c.started_at, c.conversation_id))

    # 5. Export conversations.jsonl
    convs_output_path = config.get_conversations_path()
    logger.info(f"Writing {len(reconstructed_conversations):,} conversations to {convs_output_path}...")
    with open(convs_output_path, "w", encoding="utf-8") as f:
        for conv in reconstructed_conversations:
            f.write(json.dumps(conv.to_dict(), ensure_ascii=False) + "\n")

    # 6. Export audit reports
    audit_json_path = config.get_audit_json_path()
    audit_csv_path = config.get_audit_csv_path()
    logger.info(f"Writing audit reports to {audit_json_path} and {audit_csv_path}...")
    audit.write_json(audit_json_path)
    audit.write_csv(audit_csv_path)

    # 7. Human taxonomy sampling (if requested)
    if config.sample_size > 0:
        samples_dir = config.get_samples_dir()
        sampled = sample_conversations_stratified(
            conversations=reconstructed_conversations,
            sample_size=config.sample_size,
            seed=config.seed,
        )

        if config.export_sample_markdown:
            md_path = samples_dir / f"conversation_sample_{config.sample_size}.md"
            export_sample_markdown(sampled, md_path, seed=config.seed)

        if config.export_sample_jsonl:
            jsonl_path = samples_dir / f"conversation_sample_{config.sample_size}.jsonl"
            export_sample_jsonl(sampled, jsonl_path)

    return reconstructed_conversations, audit, schema_report
