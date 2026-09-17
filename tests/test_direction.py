"""Unit tests for direction and role classification."""

from hiver_tesco.direction import classify_direction
from hiver_tesco.models import Direction


def test_clean_outbound():
    direction, is_ambiguous = classify_direction("Tesco", False, brand_id="Tesco")
    assert direction == Direction.OUTBOUND
    assert is_ambiguous is False


def test_clean_outbound_case_insensitive():
    direction, is_ambiguous = classify_direction("tesco", False, brand_id="Tesco")
    assert direction == Direction.OUTBOUND
    assert is_ambiguous is False


def test_clean_inbound():
    direction, is_ambiguous = classify_direction("115881", True, brand_id="Tesco")
    assert direction == Direction.INBOUND
    assert is_ambiguous is False


def test_ambiguous_brand_inbound():
    # Brand claimed to be inbound
    direction, is_ambiguous = classify_direction("Tesco", True, brand_id="Tesco")
    assert is_ambiguous is True
    assert direction == Direction.OUTBOUND


def test_ambiguous_customer_outbound():
    # Customer claimed to be outbound
    direction, is_ambiguous = classify_direction("115881", False, brand_id="Tesco")
    assert is_ambiguous is True
    assert direction == Direction.INBOUND
