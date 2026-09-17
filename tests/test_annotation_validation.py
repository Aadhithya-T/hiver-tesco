"""Unit tests for Phase 2 annotation schema, labels, and PII validation."""

from scripts.validate_annotations import (
    check_pii_in_evidence_notes,
    validate_annotations,
)


def test_pii_safeguards():
    # Postcode
    assert len(check_pii_in_evidence_notes("Customer postcode is SW1A 1AA")) > 0
    assert len(check_pii_in_evidence_notes("postcode EC1A 1BB")) > 0

    # Email
    assert len(check_pii_in_evidence_notes("Contact me at john.doe@example.com")) > 0

    # Phone
    assert len(check_pii_in_evidence_notes("Call me on 07123456789")) > 0
    assert len(check_pii_in_evidence_notes("Phone is +44 7123 456789")) > 0

    # Raw 7+ digit number
    assert len(check_pii_in_evidence_notes("Order 12345678")) > 0

    # Safe generic annotations
    assert len(check_pii_in_evidence_notes("[Order number provided]")) == 0
    assert len(check_pii_in_evidence_notes("[Postcode provided in DM]")) == 0
    assert len(check_pii_in_evidence_notes("Store: Hammersmith Superstore")) == 0
    assert len(check_pii_in_evidence_notes("Product: Romano Chicken Pizza")) == 0
    assert len(check_pii_in_evidence_notes("Expiry date: 2017-11-01")) == 0


def test_validate_annotations_empty_template():
    # Empty template with 2 unlabelled rows should pass schema check
    records = [
        {"conversation_id": "1", "primary_intent": "", "escalation_needed": "", "escalation_rationale": "", "evidence_notes": "", "ambiguity_notes": ""},
        {"conversation_id": "2", "primary_intent": "", "escalation_needed": "", "escalation_rationale": "", "evidence_notes": "", "ambiguity_notes": ""},
    ]
    is_valid, summary, issues = validate_annotations(records)
    assert is_valid is True
    assert summary["labelled_records"] == 0
    assert summary["unlabelled_records"] == 2
    assert len(issues) == 0


def test_validate_annotations_valid_complete():
    records = [
        {
            "conversation_id": "1",
            "primary_intent": "delivery_and_orders",
            "escalation_needed": "escalate",
            "escalation_rationale": "requires_pii_or_dm",
            "evidence_notes": "[Order number provided in thread]",
            "ambiguity_notes": "",
        },
        {
            "conversation_id": "2",
            "primary_intent": "store_experience_and_staff",
            "escalation_needed": "do_not_escalate",
            "escalation_rationale": "none",
            "evidence_notes": "Store: Sheffield Superstore",
            "ambiguity_notes": "Praise for staff colleague",
        },
    ]
    is_valid, summary, issues = validate_annotations(records)
    assert is_valid is True
    assert summary["labelled_records"] == 2
    assert summary["unlabelled_records"] == 0
    assert summary["intent_distribution"]["delivery_and_orders"] == 1
    assert summary["intent_distribution"]["store_experience_and_staff"] == 1
    assert summary["escalation_distribution"]["escalate"] == 1
    assert summary["escalation_distribution"]["do_not_escalate"] == 1


def test_validate_annotations_invalid_intent():
    records = [
        {
            "conversation_id": "1",
            "primary_intent": "made_up_intent",
            "escalation_needed": "do_not_escalate",
            "escalation_rationale": "none",
            "evidence_notes": "",
            "ambiguity_notes": "",
        }
    ]
    is_valid, summary, issues = validate_annotations(records)
    assert is_valid is False
    assert any("Invalid primary_intent" in iss for iss in issues)


def test_validate_annotations_missing_escalation_rationale():
    records = [
        {
            "conversation_id": "1",
            "primary_intent": "delivery_and_orders",
            "escalation_needed": "escalate",
            "escalation_rationale": "",  # Missing rationale when escalate!
            "evidence_notes": "",
            "ambiguity_notes": "",
        }
    ]
    is_valid, summary, issues = validate_annotations(records)
    assert is_valid is False
    assert any("escalation_rationale is required" in iss for iss in issues)


def test_validate_annotations_pii_detected():
    records = [
        {
            "conversation_id": "1",
            "primary_intent": "delivery_and_orders",
            "escalation_needed": "escalate",
            "escalation_rationale": "requires_pii_or_dm",
            "evidence_notes": "Customer lives at SW1A 1AA, call 07123456789",  # Leaked PII!
            "ambiguity_notes": "",
        }
    ]
    is_valid, summary, issues = validate_annotations(records)
    assert is_valid is False
    assert any("PII Violation" in iss for iss in issues)


def test_validate_annotations_duplicate_conv_id():
    records = [
        {"conversation_id": "1", "primary_intent": "delivery_and_orders", "escalation_needed": "do_not_escalate", "escalation_rationale": "none", "evidence_notes": "", "ambiguity_notes": ""},
        {"conversation_id": "1", "primary_intent": "stock_and_availability", "escalation_needed": "do_not_escalate", "escalation_rationale": "none", "evidence_notes": "", "ambiguity_notes": ""},
    ]
    is_valid, summary, issues = validate_annotations(records)
    assert is_valid is False
    assert any("Duplicate conversation_id" in iss for iss in issues)
