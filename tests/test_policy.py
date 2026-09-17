"""Unit tests for Phase 5 deterministic escalation and response policy."""

import pytest
import pandas as pd

from hiver_tesco.policy.models import (
    EscalationCategory,
    PolicyAction,
    PolicyContext,
    PolicyDecision,
)
from hiver_tesco.policy.engine import DeterministicPolicyEngine
from hiver_tesco.policy.evaluation import (
    evaluate_policy_predictions,
    extract_policy_error_tables,
)


@pytest.fixture
def engine():
    return DeterministicPolicyEngine(retrieval_threshold=0.55, confidence_threshold=0.18)


def test_harassment_abuse_rule(engine):
    ctx = PolicyContext(
        query_text="I will report you to the police for this assault you bastards",
        model_confidence=0.85,
        retrieval_score=0.75,
        retrieval_accepted=True,
    )
    decision = engine.evaluate(ctx)
    assert decision.action == PolicyAction.ESCALATE
    assert decision.rule_id == "RULE_01_HARASSMENT_ABUSE"
    assert decision.priority == 1
    assert decision.rule_category == EscalationCategory.SAFETY_AND_HARASSMENT.value
    assert decision.suggested_routing_category == "safety_and_vulnerable_customer"


def test_food_safety_health_rule(engine):
    ctx = PolicyContext(
        query_text="Found glass and a dead beetle in my salad bag, I was sick and vomiting",
        model_confidence=0.90,
        retrieval_score=0.80,
        retrieval_accepted=True,
    )
    decision = engine.evaluate(ctx)
    assert decision.action == PolicyAction.ESCALATE
    assert decision.rule_id == "RULE_02_FOOD_SAFETY_HEALTH"
    assert decision.priority == 1
    assert decision.rule_category == EscalationCategory.PRODUCT_SAFETY_AND_HEALTH.value
    assert decision.suggested_routing_category == "product_safety_investigation"


def test_payment_refund_rule(engine):
    ctx = PolicyContext(
        query_text="You double charged me £15.50 on my card and I want a refund",
        model_confidence=0.70,
        retrieval_score=0.65,
        retrieval_accepted=True,
    )
    decision = engine.evaluate(ctx)
    assert decision.action == PolicyAction.ESCALATE
    assert decision.rule_id == "RULE_03_PAYMENT_FINANCIAL_LOSS"
    assert decision.priority == 2
    assert decision.rule_category == EscalationCategory.FINANCIAL_AND_REFUNDS.value
    assert decision.suggested_routing_category == "billing_and_refunds"


def test_account_security_pii_rule(engine):
    ctx = PolicyContext(
        query_text="My clubcard account was hacked and I cannot log in, please reset password",
        model_confidence=0.65,
        retrieval_score=0.70,
        retrieval_accepted=True,
    )
    decision = engine.evaluate(ctx)
    assert decision.action == PolicyAction.ESCALATE
    assert decision.rule_id == "RULE_04_ACCOUNT_SECURITY_PII"
    assert decision.priority == 2
    assert decision.rule_category == EscalationCategory.ACCOUNT_AND_SECURITY.value


def test_unresolved_repetition_rule(engine):
    # Regex repetition
    ctx1 = PolicyContext(
        query_text="Called 5 times and still waiting for delivery, nobody answered",
        model_confidence=0.50,
        retrieval_score=0.60,
        retrieval_accepted=True,
    )
    decision1 = engine.evaluate(ctx1)
    assert decision1.action == PolicyAction.ESCALATE
    assert decision1.rule_id == "RULE_05_UNRESOLVED_REPETITION"
    assert decision1.priority == 3

    # Turn count repetition
    ctx2 = PolicyContext(
        query_text="Any update on my query?",
        model_confidence=0.50,
        retrieval_score=0.60,
        retrieval_accepted=True,
        inbound_turn_count=2,
    )
    decision2 = engine.evaluate(ctx2)
    assert decision2.action == PolicyAction.ESCALATE
    assert decision2.rule_id == "RULE_05_UNRESOLVED_REPETITION"


def test_legal_regulatory_rule(engine):
    ctx = PolicyContext(
        query_text="If this is not sorted I am taking you to the Ombudsman and Trading Standards",
        model_confidence=0.60,
        retrieval_score=0.60,
        retrieval_accepted=True,
    )
    decision = engine.evaluate(ctx)
    assert decision.action == PolicyAction.ESCALATE
    assert decision.rule_id == "RULE_06_LEGAL_REGULATORY"
    assert decision.priority == 3
    assert decision.suggested_routing_category == "legal_and_regulatory_affairs"


def test_poor_retrieval_evidence_rule(engine):
    # Low score
    ctx_low = PolicyContext(
        query_text="Can I get some advice on this strange item?",
        model_confidence=0.80,
        retrieval_score=0.42,  # Below 0.55
        retrieval_accepted=True,
    )
    decision_low = engine.evaluate(ctx_low)
    assert decision_low.action == PolicyAction.ESCALATE
    assert decision_low.rule_id == "RULE_07_POOR_OR_NO_EVIDENCE"
    assert decision_low.priority == 4

    # Rejected by filter
    ctx_rej = PolicyContext(
        query_text="Can I get some advice on this item?",
        model_confidence=0.80,
        retrieval_score=0.75,
        retrieval_accepted=False,
    )
    decision_rej = engine.evaluate(ctx_rej)
    assert decision_rej.action == PolicyAction.ESCALATE
    assert decision_rej.rule_id == "RULE_07_POOR_OR_NO_EVIDENCE"


def test_low_model_confidence_rule(engine):
    # Case 1: Low confidence AND ungrounded retrieval -> must escalate
    ctx_ungrounded = PolicyContext(
        query_text="Item inquiry here",
        model_confidence=0.14,  # Below 0.20
        retrieval_score=0.58,   # Below 0.65 safety threshold
        retrieval_accepted=True,
    )
    decision = engine.evaluate(ctx_ungrounded)
    assert decision.action == PolicyAction.ESCALATE
    assert decision.rule_id == "RULE_08_LOW_MODEL_CONFIDENCE"
    assert decision.priority == 4

    # Case 2: Low confidence BUT strong retrieval -> safe respond
    ctx_grounded = PolicyContext(
        query_text="Item inquiry here",
        model_confidence=0.14,
        retrieval_score=0.75,  # Strong retrieval grounds the inquiry
        retrieval_accepted=True,
    )
    decision_grounded = engine.evaluate(ctx_grounded)
    assert decision_grounded.action == PolicyAction.RESPOND


def test_staff_incident_rule(engine):
    ctx = PolicyContext(
        query_text="The cashier in store was rude and appalling service",
        model_confidence=0.60,
        retrieval_score=0.65,
        retrieval_accepted=True,
    )
    decision = engine.evaluate(ctx)
    assert decision.action == PolicyAction.ESCALATE
    assert decision.rule_id == "RULE_09_STAFF_INCIDENT"
    assert decision.priority == 5
    assert decision.suggested_routing_category == "store_manager_escalations"


def test_priority_override_hierarchy(engine):
    # Query with health hazard (P1) AND poor retrieval (P4): P1 MUST win
    ctx = PolicyContext(
        query_text="Food had glass and caused sickness",
        model_confidence=0.10,  # Low conf (P4)
        retrieval_score=0.30,   # Low retr (P4)
        retrieval_accepted=False,
    )
    decision = engine.evaluate(ctx)
    assert decision.priority == 1
    assert decision.rule_id == "RULE_02_FOOD_SAFETY_HEALTH"


def test_safe_respond_decision(engine):
    ctx = PolicyContext(
        query_text="Do you have gluten free bread in stock at the flitwick store?",
        model_confidence=0.45,
        predicted_intent="stock_and_availability",
        retrieval_score=0.72,
        retrieval_accepted=True,
        inbound_turn_count=1,
    )
    decision = engine.evaluate(ctx)
    assert decision.action == PolicyAction.RESPOND
    assert decision.rule_id == "DEFAULT_SAFE_RESPOND"
    assert decision.response_guidance is not None
    assert "do_not_promise_refunds_or_compensation" in decision.response_guidance.disallowed_actions
    assert "provide_store_opening_hours_or_stock_status" in decision.response_guidance.allowed_actions


def test_policy_metrics_and_error_tables():
    decisions = [
        PolicyDecision(PolicyAction.ESCALATE, "reason1", "R1", "cat1", 1, []),
        PolicyDecision(PolicyAction.ESCALATE, "reason2", "R2", "cat2", 2, []),
        PolicyDecision(PolicyAction.RESPOND, "reason3", "DEFAULT", "cat3", 99, []),
        PolicyDecision(PolicyAction.RESPOND, "reason4", "DEFAULT", "cat3", 99, []),
    ]
    y_true = ["escalate", "do_not_escalate", "escalate", "do_not_escalate"]
    records = [
        {"conversation_id": "1", "query_text": "q1", "human_escalate": "escalate"},
        {"conversation_id": "2", "query_text": "q2", "human_escalate": "do_not_escalate"},
        {"conversation_id": "3", "query_text": "q3", "human_escalate": "escalate"},
        {"conversation_id": "4", "query_text": "q4", "human_escalate": "do_not_escalate"},
    ]

    metrics = evaluate_policy_predictions(decisions, y_true)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 1
    assert metrics.accuracy == 0.50

    fp_df, fn_df = extract_policy_error_tables(records, decisions)
    assert len(fp_df) == 1
    assert fp_df.iloc[0]["conversation_id"] == "2"
    assert len(fn_df) == 1
    assert fn_df.iloc[0]["conversation_id"] == "3"
