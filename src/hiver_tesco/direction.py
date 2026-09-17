"""Explicit direction and ownership labeling rules derived from the dataset."""

from typing import Tuple
from hiver_tesco.models import Direction


def classify_direction(
    author_id: str,
    inbound_flag: bool,
    brand_id: str = "Tesco",
) -> Tuple[Direction, bool]:
    """Classify a message as INBOUND (customer -> brand) or OUTBOUND (brand -> customer).

    Rules derived from the verified TWCS Tesco subset:
    1. Outbound messages: authored by `brand_id` (case-insensitive) AND inbound_flag is False.
    2. Inbound messages: authored by a customer (author_id != brand_id) AND inbound_flag is True.
    3. Ambiguous ownership:
       - Author matches brand_id, but inbound_flag is True.
       - Author does not match brand_id, but inbound_flag is False.

    Returns:
        Tuple of (Direction, is_ambiguous: bool)
    """
    author_clean = author_id.strip().lower()
    brand_clean = brand_id.strip().lower()
    is_brand_author = (author_clean == brand_clean)

    if is_brand_author and not inbound_flag:
        return Direction.OUTBOUND, False
    elif not is_brand_author and inbound_flag:
        return Direction.INBOUND, False
    elif is_brand_author and inbound_flag:
        # Conflict: Brand author claimed to be inbound
        return Direction.OUTBOUND, True
    else:
        # Conflict: Non-brand author claimed to be outbound
        return Direction.INBOUND, True
