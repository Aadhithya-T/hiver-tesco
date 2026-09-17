"""Shared pytest fixtures and synthetic data generators."""

from datetime import datetime, timezone
from pathlib import Path
import pytest
from hiver_tesco.models import RawTweet


@pytest.fixture
def base_time():
    return datetime(2017, 10, 31, 12, 0, 0, tzinfo=timezone.utc)


def make_raw_tweet(
    tweet_id: str,
    author_id: str,
    inbound: bool,
    created_at: datetime,
    text: str,
    in_response_to_tweet_id: str = "",
    conversation_id: str = "conv_1",
    response_tweet_id: str = "",
) -> RawTweet:
    return RawTweet(
        tweet_id=tweet_id,
        author_id=author_id,
        inbound=inbound,
        created_at_raw=created_at.strftime("%a %b %d %H:%M:%S +0000 %Y"),
        created_at=created_at,
        text=text,
        response_tweet_id=response_tweet_id or None,
        in_response_to_tweet_id=in_response_to_tweet_id or None,
        conversation_id=conversation_id,
    )


@pytest.fixture
def linear_thread(base_time):
    """Clean linear thread: Customer -> Tesco -> Customer."""
    t0 = base_time
    t1 = datetime(2017, 10, 31, 12, 5, 0, tzinfo=timezone.utc)
    t2 = datetime(2017, 10, 31, 12, 10, 0, tzinfo=timezone.utc)

    return [
        make_raw_tweet("101", "cust1", True, t0, "@Tesco My order #123 is delayed!", "", "conv_linear"),
        make_raw_tweet("102", "Tesco", False, t1, "@cust1 Hi, DM us your postcode.", "101", "conv_linear"),
        make_raw_tweet("103", "cust1", True, t2, "@Tesco Sent via DM.", "102", "conv_linear"),
    ]


@pytest.fixture
def branching_thread(base_time):
    """Branching thread: Customer root gets two Tesco replies (1/2 and 2/2)."""
    t0 = base_time
    t1 = datetime(2017, 10, 31, 12, 5, 0, tzinfo=timezone.utc)
    t2 = datetime(2017, 10, 31, 12, 5, 1, tzinfo=timezone.utc)

    return [
        make_raw_tweet("201", "cust1", True, t0, "@Tesco I need help with refund.", "", "conv_branch"),
        make_raw_tweet("202", "Tesco", False, t1, "@cust1 Hi! We're checking this. 1/2", "201", "conv_branch"),
        make_raw_tweet("203", "Tesco", False, t2, "@cust1 Could you confirm your store? 2/2", "201", "conv_branch"),
    ]


@pytest.fixture
def missing_parent_thread(base_time):
    """Thread where parent tweet is not in the dataset."""
    t1 = datetime(2017, 10, 31, 12, 5, 0, tzinfo=timezone.utc)
    t2 = datetime(2017, 10, 31, 12, 10, 0, tzinfo=timezone.utc)

    return [
        make_raw_tweet("302", "Tesco", False, t1, "@cust1 We looked into your missing item.", "301", "301"),
        make_raw_tweet("303", "cust1", True, t2, "@Tesco Thank you for sorting this.", "302", "301"),
    ]


@pytest.fixture
def cycle_thread(base_time):
    """Thread with a directed reply cycle: A -> B -> C -> A."""
    t0 = base_time
    t1 = datetime(2017, 10, 31, 12, 5, 0, tzinfo=timezone.utc)
    t2 = datetime(2017, 10, 31, 12, 10, 0, tzinfo=timezone.utc)

    return [
        make_raw_tweet("401", "cust1", True, t0, "Tweet 1", "403", "conv_cycle"),
        make_raw_tweet("402", "Tesco", False, t1, "Tweet 2", "401", "conv_cycle"),
        make_raw_tweet("403", "cust1", True, t2, "Tweet 3", "402", "conv_cycle"),
    ]


@pytest.fixture
def cross_brand_thread(base_time):
    """Thread mentioning competitor supermarket (@sainsburys)."""
    t0 = base_time
    t1 = datetime(2017, 10, 31, 12, 5, 0, tzinfo=timezone.utc)

    return [
        make_raw_tweet("501", "cust1", True, t0, "@Tesco do you price match @sainsburys on milk?", "", "conv_cross"),
        make_raw_tweet("502", "Tesco", False, t1, "@cust1 Hi there, please check our brand guarantee page.", "501", "conv_cross"),
    ]
