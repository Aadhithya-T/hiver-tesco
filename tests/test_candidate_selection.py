"""Unit tests for Phase 2 golden candidate selection and disjoint stratification."""

import pytest
from scripts.create_golden_candidate_set import (
    format_dialogue_for_csv,
    select_stratified_candidates,
)


@pytest.fixture
def mock_conversations():
    """Create a mock population of 500 conversations with diverse flags and lengths."""
    convs = []
    for i in range(500):
        # assign lengths: 40% length 2-3, 40% length 4-6, 20% length 7+
        if i % 5 in (0, 1):
            length = (i % 2) + 2  # 2 or 3
        elif i % 5 in (2, 3):
            length = (i % 3) + 4  # 4, 5, or 6
        else:
            length = (i % 6) + 7  # 7 to 12

        # every 15th conv is cross-brand or outbound root
        is_special = (i % 15 == 0)
        flags = {
            "is_clean": not is_special,
            "has_cross_brand_mention": is_special and (i % 30 == 0),
            "is_outbound_root": is_special and (i % 30 != 0),
            "is_branching": (length > 4),
        }
        convs.append({
            "conversation_id": f"conv_{i:04d}",
            "message_count": length,
            "quality_flags": flags,
            "messages": [
                {
                    "tweet_id": f"t_{i}_{j}",
                    "parent_tweet_id": None if j == 0 else f"t_{i}_{j-1}",
                    "direction": "inbound" if j % 2 == 0 else "outbound",
                    "author_id": f"cust_{i}" if j % 2 == 0 else "Tesco",
                    "turn_index": j,
                    "created_at": f"2017-10-31T12:{j:02d}:00+00:00",
                    "text": f"Tweet text {j}",
                }
                for j in range(length)
            ],
            "duration_seconds": length * 60,
        })
    return convs


def test_select_stratified_candidates_exact_count(mock_conversations):
    candidates = select_stratified_candidates(mock_conversations, sample_size=200, seed=100)
    assert len(candidates) == 200

    # Ensure all conversation IDs are unique
    conv_ids = [c["conversation_id"] for c in candidates]
    assert len(conv_ids) == len(set(conv_ids))


def test_select_stratified_candidates_disjoint_tiers(mock_conversations):
    candidates = select_stratified_candidates(mock_conversations, sample_size=200, seed=100)

    special = [
        c for c in candidates
        if c["quality_flags"].get("has_cross_brand_mention") or c["quality_flags"].get("is_outbound_root")
    ]
    short = [
        c for c in candidates
        if not (c["quality_flags"].get("has_cross_brand_mention") or c["quality_flags"].get("is_outbound_root"))
        and c["message_count"] <= 3
    ]
    med = [
        c for c in candidates
        if not (c["quality_flags"].get("has_cross_brand_mention") or c["quality_flags"].get("is_outbound_root"))
        and 4 <= c["message_count"] <= 6
    ]
    long = [
        c for c in candidates
        if not (c["quality_flags"].get("has_cross_brand_mention") or c["quality_flags"].get("is_outbound_root"))
        and c["message_count"] >= 7
    ]

    # Check that sums match exactly 200 with zero overlap
    total_disjoint = len(special) + len(short) + len(med) + len(long)
    assert total_disjoint == 200
    assert len(special) == 10  # 5% of 200
    assert len(short) == 70    # 35% of 200
    assert len(med) == 80      # 40% of 200
    assert len(long) == 40     # 20% of 200


def test_candidate_selection_determinism(mock_conversations):
    run_1 = select_stratified_candidates(mock_conversations, sample_size=50, seed=100)
    run_2 = select_stratified_candidates(mock_conversations, sample_size=50, seed=100)
    run_diff_seed = select_stratified_candidates(mock_conversations, sample_size=50, seed=999)

    assert [c["conversation_id"] for c in run_1] == [c["conversation_id"] for c in run_2]
    assert [c["conversation_id"] for c in run_1] != [c["conversation_id"] for c in run_diff_seed]


def test_format_dialogue_for_csv():
    messages = [
        {"turn_index": 0, "direction": "inbound", "author_id": "cust1", "created_at": "2017-10-31T12:00:00+00:00", "parent_tweet_id": None, "text": "Need help with delivery"},
        {"turn_index": 1, "direction": "outbound", "author_id": "Tesco", "created_at": "2017-10-31T12:05:00+00:00", "parent_tweet_id": "t1", "text": "Hi! Please DM us."},
    ]
    dialogue = format_dialogue_for_csv(messages)
    assert "[Turn 0 | CUSTOMER | @cust1 | 2017-10-31 12:00:00] (root):" in dialogue
    assert "Need help with delivery" in dialogue
    assert "[Turn 1 | TESCO | @Tesco | 2017-10-31 12:05:00] (in reply to t1):" in dialogue
