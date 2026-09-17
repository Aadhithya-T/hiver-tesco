"""Anomaly and quality issue detection for Twitter customer support threads."""

from typing import Dict, List, Set, Tuple
from hiver_tesco.config import COMPETITOR_REGEX
from hiver_tesco.models import QualityFlags, RawTweet


def find_cross_brand_mentions(text: str) -> List[str]:
    """Extract competitor supermarket and brand handles mentioned in text."""
    matches = COMPETITOR_REGEX.findall(text)
    # Deduplicate while preserving lowercase consistency
    seen: Set[str] = set()
    result: List[str] = []
    for m in matches:
        low = m.lower()
        if low not in seen:
            seen.add(low)
            result.append(low)
    return result


def detect_cycles_in_thread(
    tweets: List[RawTweet],
) -> Tuple[bool, List[List[str]]]:
    """Detect any directed reply cycles in a thread using DFS recursion stack.

    Returns:
        (has_cycle: bool, list_of_cycle_paths: List[List[str]])
    """
    parent_map: Dict[str, str] = {
        t.tweet_id: t.in_response_to_tweet_id
        for t in tweets
        if t.in_response_to_tweet_id
    }

    visited: Set[str] = set()
    recursion_stack: Set[str] = set()
    cycles: List[List[str]] = []

    def dfs(node: str, path: List[str]):
        visited.add(node)
        recursion_stack.add(node)
        path.append(node)

        parent = parent_map.get(node)
        if parent:
            if parent in recursion_stack:
                # Cycle found!
                cycle_start_idx = path.index(parent)
                cycles.append(path[cycle_start_idx:] + [parent])
            elif parent not in visited and parent in parent_map:
                dfs(parent, path)

        recursion_stack.remove(node)
        path.pop()

    for t in tweets:
        if t.tweet_id not in visited:
            dfs(t.tweet_id, [])

    return len(cycles) > 0, cycles


def evaluate_quality_flags(
    tweets: List[RawTweet],
    known_tweet_ids: Set[str],
    root_tweet: RawTweet,
    brand_id: str = "Tesco",
    thread_has_duplicate_id: bool = False,
) -> QualityFlags:
    """Evaluate all data-quality and structural anomaly flags for a thread."""
    flags = QualityFlags()

    # 1. Duplicate IDs
    flags.has_duplicate_tweet_id = thread_has_duplicate_id

    # 2. Empty text
    flags.has_empty_text = any(not (t.text and t.text.strip()) for t in tweets)

    # 3. Missing parents / orphaned replies
    missing_ids: List[str] = []
    for t in tweets:
        if t.in_response_to_tweet_id and t.in_response_to_tweet_id not in known_tweet_ids:
            missing_ids.append(t.in_response_to_tweet_id)
    if missing_ids:
        flags.has_missing_parent = True
        flags.missing_parent_ids = sorted(list(set(missing_ids)))

    # 4. Cycles
    has_cycle, _ = detect_cycles_in_thread(tweets)
    flags.has_cycle = has_cycle

    # 5. Cross-brand mentions
    all_competitor_handles: Set[str] = set()
    for t in tweets:
        handles = find_cross_brand_mentions(t.text)
        all_competitor_handles.update(handles)
    if all_competitor_handles:
        flags.has_cross_brand_mention = True
        flags.cross_brand_handles = sorted(list(all_competitor_handles))

    # 6. Outbound root (marketing, broadcast, or proactive Tesco tweet)
    if root_tweet and root_tweet.author_id.strip().lower() == brand_id.strip().lower():
        flags.is_outbound_root = True

    # 7. Multi-customer thread
    customers = {
        t.author_id
        for t in tweets
        if t.author_id.strip().lower() != brand_id.strip().lower()
    }
    flags.is_multi_customer = len(customers) > 1

    # 8. Branching (more than one child for any parent)
    child_counts: Dict[str, int] = {}
    for t in tweets:
        p = t.in_response_to_tweet_id
        if p:
            child_counts[p] = child_counts.get(p, 0) + 1
    flags.is_branching = any(count > 1 for count in child_counts.values())

    return flags
