"""Unit tests for anomaly detection logic."""

from datetime import datetime, timezone
from hiver_tesco.anomalies import (
    detect_cycles_in_thread,
    evaluate_quality_flags,
    find_cross_brand_mentions,
)
from conftest import make_raw_tweet


def test_cross_brand_mentions_found():
    text = "Why is milk cheaper at @sainsburys and @asda than @Tesco?"
    matches = find_cross_brand_mentions(text)
    assert "@sainsburys" in matches
    assert "@asda" in matches
    assert "@tesco" not in matches  # Tesco is our target brand


def test_cross_brand_mentions_none():
    text = "Thanks @Tesco for helping me in store today."
    matches = find_cross_brand_mentions(text)
    assert len(matches) == 0


def test_detect_cycles():
    t0 = datetime(2017, 10, 31, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2017, 10, 31, 12, 1, 0, tzinfo=timezone.utc)
    t2 = datetime(2017, 10, 31, 12, 2, 0, tzinfo=timezone.utc)

    # A -> B -> A
    cycle_tweets = [
        make_raw_tweet("1", "Tesco", False, t0, "Hello", "2"),
        make_raw_tweet("2", "cust", True, t1, "Hi", "1"),
    ]
    has_cycle, paths = detect_cycles_in_thread(cycle_tweets)
    assert has_cycle is True
    assert len(paths) > 0

    # Clean linear A -> B -> C
    linear_tweets = [
        make_raw_tweet("1", "cust", True, t0, "Hello", ""),
        make_raw_tweet("2", "Tesco", False, t1, "Hi", "1"),
        make_raw_tweet("3", "cust", True, t2, "Thanks", "2"),
    ]
    has_cycle_lin, _ = detect_cycles_in_thread(linear_tweets)
    assert has_cycle_lin is False


def test_evaluate_quality_flags_clean(linear_thread):
    known_ids = {t.tweet_id for t in linear_thread}
    flags = evaluate_quality_flags(
        tweets=linear_thread,
        known_tweet_ids=known_ids,
        root_tweet=linear_thread[0],
        brand_id="Tesco",
    )
    assert flags.is_clean is True
    assert flags.has_missing_parent is False
    assert flags.has_cross_brand_mention is False
    assert flags.is_outbound_root is False


def test_evaluate_quality_flags_missing_parent(missing_parent_thread):
    # Only thread tweets are in known_ids; parent 301 is missing!
    known_ids = {t.tweet_id for t in missing_parent_thread}
    flags = evaluate_quality_flags(
        tweets=missing_parent_thread,
        known_tweet_ids=known_ids,
        root_tweet=missing_parent_thread[0],
        brand_id="Tesco",
    )
    assert flags.has_missing_parent is True
    assert "301" in flags.missing_parent_ids
    assert flags.is_clean is False


def test_evaluate_quality_flags_cross_brand(cross_brand_thread):
    known_ids = {t.tweet_id for t in cross_brand_thread}
    flags = evaluate_quality_flags(
        tweets=cross_brand_thread,
        known_tweet_ids=known_ids,
        root_tweet=cross_brand_thread[0],
        brand_id="Tesco",
    )
    assert flags.has_cross_brand_mention is True
    assert "@sainsburys" in flags.cross_brand_handles
    assert flags.is_clean is False
