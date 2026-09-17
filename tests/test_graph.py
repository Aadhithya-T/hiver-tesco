"""Unit tests for thread graph reconstruction and ordering."""

from hiver_tesco.graph import reconstruct_conversation
from hiver_tesco.models import Direction


def test_reconstruct_linear_thread(linear_thread):
    known_ids = {t.tweet_id for t in linear_thread}
    conv = reconstruct_conversation(
        conversation_id="conv_linear",
        tweets=linear_thread,
        known_global_tweet_ids=known_ids,
        brand_id="Tesco",
    )

    assert conv.conversation_id == "conv_linear"
    assert conv.message_count == 3
    assert conv.inbound_count == 2
    assert conv.outbound_count == 1
    assert conv.source_root_tweet_id == "101"

    # Message order & turn indices
    assert [m.tweet_id for m in conv.messages] == ["101", "102", "103"]
    assert [m.turn_index for m in conv.messages] == [0, 1, 2]
    assert [m.depth for m in conv.messages] == [0, 1, 2]
    assert [m.direction for m in conv.messages] == [
        Direction.INBOUND,
        Direction.OUTBOUND,
        Direction.INBOUND,
    ]
    assert conv.messages[0].is_root is True
    assert conv.messages[1].is_root is False


def test_reconstruct_branching_thread(branching_thread):
    known_ids = {t.tweet_id for t in branching_thread}
    conv = reconstruct_conversation(
        conversation_id="conv_branch",
        tweets=branching_thread,
        known_global_tweet_ids=known_ids,
        brand_id="Tesco",
    )

    assert conv.message_count == 3
    assert conv.quality_flags.is_branching is True

    # Root has 2 children: 202 and 203
    root_msg = conv.messages[0]
    assert root_msg.tweet_id == "201"
    assert set(root_msg.child_tweet_ids) == {"202", "203"}

    # Depths: 201 is depth 0, 202 and 203 are both depth 1
    msg_dict = {m.tweet_id: m for m in conv.messages}
    assert msg_dict["201"].depth == 0
    assert msg_dict["202"].depth == 1
    assert msg_dict["203"].depth == 1


def test_reconstruct_missing_parent_thread(missing_parent_thread):
    known_ids = {t.tweet_id for t in missing_parent_thread}
    conv = reconstruct_conversation(
        conversation_id="301",
        tweets=missing_parent_thread,
        known_global_tweet_ids=known_ids,
        brand_id="Tesco",
    )

    assert conv.quality_flags.has_missing_parent is True
    assert "301" in conv.quality_flags.missing_parent_ids
    # Because 301 is missing, source_root_tweet_id is None
    assert conv.source_root_tweet_id is None
    # Canonical root chosen as earliest available tweet (302)
    assert conv.messages[0].tweet_id == "302"
    assert conv.messages[0].is_root is True


def test_reconstruct_cycle_thread(cycle_thread):
    known_ids = {t.tweet_id for t in cycle_thread}
    conv = reconstruct_conversation(
        conversation_id="conv_cycle",
        tweets=cycle_thread,
        known_global_tweet_ids=known_ids,
        brand_id="Tesco",
    )

    assert conv.quality_flags.has_cycle is True
    # Still produces all 3 messages deterministically without infinite loop
    assert conv.message_count == 3
    assert len(conv.messages) == 3
