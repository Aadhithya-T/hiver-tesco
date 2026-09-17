"""Unit tests for human taxonomy discovery sampler."""

from datetime import datetime, timezone
from pathlib import Path
from hiver_tesco.graph import reconstruct_conversation
from hiver_tesco.models import ReconstructedConversation
from hiver_tesco.sampler import (
    export_sample_jsonl,
    export_sample_markdown,
    sample_conversations_stratified,
)
from conftest import make_raw_tweet


def create_dummy_conversation(cid: str, length: int, cross_brand: bool = False) -> ReconstructedConversation:
    t0 = datetime(2017, 10, 31, 12, 0, 0, tzinfo=timezone.utc)
    tweets = []
    text_root = "@Tesco need help @sainsburys" if cross_brand else "@Tesco need help"
    tweets.append(make_raw_tweet(f"{cid}_0", "cust", True, t0, text_root, "", cid))

    for i in range(1, length):
        dt = datetime(2017, 10, 31, 12, i, 0, tzinfo=timezone.utc)
        author = "Tesco" if i % 2 == 1 else "cust"
        inbound = (author != "Tesco")
        tweets.append(make_raw_tweet(f"{cid}_{i}", author, inbound, dt, f"Reply {i}", f"{cid}_{i-1}", cid))

    known_ids = {t.tweet_id for t in tweets}
    return reconstruct_conversation(cid, tweets, known_ids, brand_id="Tesco")


def test_sampler_reproducibility():
    convs = [create_dummy_conversation(f"c_{i}", length=(i % 8) + 2) for i in range(50)]

    sample_1 = sample_conversations_stratified(convs, sample_size=10, seed=42)
    sample_2 = sample_conversations_stratified(convs, sample_size=10, seed=42)
    sample_diff_seed = sample_conversations_stratified(convs, sample_size=10, seed=999)

    assert [c.conversation_id for c in sample_1] == [c.conversation_id for c in sample_2]
    # Different seed should likely produce a different selection
    assert [c.conversation_id for c in sample_1] != [c.conversation_id for c in sample_diff_seed]


def test_sampler_exports(tmp_path: Path):
    convs = [
        create_dummy_conversation("c_short", length=2),
        create_dummy_conversation("c_med", length=4),
        create_dummy_conversation("c_long", length=8, cross_brand=True),
    ]

    md_path = tmp_path / "sample.md"
    jsonl_path = tmp_path / "sample.jsonl"

    export_sample_markdown(convs, md_path, seed=42)
    export_sample_jsonl(convs, jsonl_path)

    assert md_path.exists()
    assert jsonl_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "Tesco Customer Support Conversation Sample" in md_content
    assert "c_long" in md_content
    assert "Human Reviewer Annotations" in md_content

    jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 3
