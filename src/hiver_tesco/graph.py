"""Thread graph reconstruction, root detection, cycle breaking, and message ordering."""

from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from hiver_tesco.direction import classify_direction
from hiver_tesco.models import (
    Direction,
    QualityFlags,
    RawTweet,
    ReconstructedConversation,
    ThreadMessage,
)
from hiver_tesco.anomalies import evaluate_quality_flags


def build_children_map(tweets: List[RawTweet]) -> Dict[str, List[str]]:
    """Build a mapping from parent_id to list of child tweet_ids within the thread."""
    children_map: Dict[str, List[str]] = {t.tweet_id: [] for t in tweets}
    for t in tweets:
        parent_id = t.in_response_to_tweet_id
        if parent_id and parent_id in children_map:
            children_map[parent_id].append(t.tweet_id)
    # Deterministically sort children by created_at then tweet_id
    tweet_dict = {t.tweet_id: t for t in tweets}
    for p_id in children_map:
        children_map[p_id].sort(
            key=lambda tid: (tweet_dict[tid].created_at, tweet_dict[tid].tweet_id)
        )
    return children_map


def compute_depths(
    root_ids: List[str],
    children_map: Dict[str, List[str]],
    all_tweet_ids: Set[str],
) -> Dict[str, int]:
    """Compute tree depth for each tweet in the conversation starting from roots."""
    depths: Dict[str, int] = {}
    queue: List[Tuple[str, int]] = [(r, 0) for r in root_ids]
    visited: Set[str] = set()

    while queue:
        node, d = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        depths[node] = d
        for child in children_map.get(node, []):
            if child not in visited:
                queue.append((child, d + 1))

    # Any unreachable nodes (e.g. disconnected components) get fallback depth
    for tid in all_tweet_ids:
        if tid not in depths:
            depths[tid] = 0

    return depths


def reconstruct_conversation(
    conversation_id: str,
    tweets: List[RawTweet],
    known_global_tweet_ids: Set[str],
    brand_id: str = "Tesco",
    thread_has_duplicate_id: bool = False,
    provenance_metadata: Optional[Dict[str, str]] = None,
) -> ReconstructedConversation:
    """Reconstruct a single conversation thread from a list of constituent tweets.

    Deterministic ordering:
    - Messages are ordered chronologically by (created_at, tweet_id).
    - Adjacency links (parent_tweet_id, child_tweet_ids) and tree depths are computed.
    - Quality flags are assessed and attached.
    """
    tweet_dict = {t.tweet_id: t for t in tweets}
    all_local_ids = set(tweet_dict.keys())
    children_map = build_children_map(tweets)

    # Identify candidate roots: tweets with no parent or whose parent is not in thread
    candidate_roots = [
        t for t in tweets
        if not t.in_response_to_tweet_id or t.in_response_to_tweet_id not in all_local_ids
    ]

    # Sort candidate roots chronologically to determine canonical thread root
    candidate_roots.sort(key=lambda t: (t.created_at, t.tweet_id))

    if candidate_roots:
        canonical_root = candidate_roots[0]
        root_ids = [t.tweet_id for t in candidate_roots]
    else:
        # Cycle present without any natural root: break cycle using earliest tweet
        sorted_tweets = sorted(tweets, key=lambda t: (t.created_at, t.tweet_id))
        canonical_root = sorted_tweets[0]
        root_ids = [canonical_root.tweet_id]

    depths = compute_depths(root_ids, children_map, all_local_ids)

    # Evaluate quality and anomaly flags
    quality_flags = evaluate_quality_flags(
        tweets=tweets,
        known_tweet_ids=known_global_tweet_ids,
        root_tweet=canonical_root,
        brand_id=brand_id,
        thread_has_duplicate_id=thread_has_duplicate_id,
    )

    # Sort all messages chronologically with tweet_id tie-breaker for 100% determinism
    ordered_raw_tweets = sorted(tweets, key=lambda t: (t.created_at, t.tweet_id))

    messages: List[ThreadMessage] = []
    inbound_count = 0
    outbound_count = 0
    customer_authors: Set[str] = set()

    for idx, t in enumerate(ordered_raw_tweets):
        direction, is_ambiguous = classify_direction(
            author_id=t.author_id,
            inbound_flag=t.inbound,
            brand_id=brand_id,
        )

        if is_ambiguous:
            quality_flags.is_ambiguous_ownership = True

        if direction == Direction.INBOUND:
            inbound_count += 1
            customer_authors.add(t.author_id)
        elif direction == Direction.OUTBOUND:
            outbound_count += 1

        is_root = (t.tweet_id == canonical_root.tweet_id)

        msg = ThreadMessage(
            tweet_id=t.tweet_id,
            parent_tweet_id=t.in_response_to_tweet_id if t.in_response_to_tweet_id else None,
            child_tweet_ids=children_map.get(t.tweet_id, []),
            author_id=t.author_id,
            direction=direction,
            created_at=t.created_at.isoformat(),
            text=t.text,
            turn_index=idx,
            depth=depths.get(t.tweet_id, 0),
            is_root=is_root,
        )
        messages.append(msg)

    # Conversation timeline boundaries
    started_at = ordered_raw_tweets[0].created_at.isoformat()
    ended_at = ordered_raw_tweets[-1].created_at.isoformat()
    duration = (ordered_raw_tweets[-1].created_at - ordered_raw_tweets[0].created_at).total_seconds()

    # Source root ID (None if missing from dataset)
    source_root_id = canonical_root.tweet_id if not canonical_root.in_response_to_tweet_id else None

    # Provenance
    prov = {
        "pipeline_phase": "Phase 1 - Extraction & Reconstruction",
        "sorting_rule": "chronological_by_created_at_then_tweet_id",
        "brand_id": brand_id,
    }
    if provenance_metadata:
        prov.update(provenance_metadata)

    return ReconstructedConversation(
        conversation_id=conversation_id,
        source_root_tweet_id=source_root_id,
        message_count=len(messages),
        inbound_count=inbound_count,
        outbound_count=outbound_count,
        customer_author_ids=sorted(list(customer_authors)),
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration,
        quality_flags=quality_flags,
        messages=messages,
        provenance=prov,
    )
