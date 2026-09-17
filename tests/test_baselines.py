"""Unit tests for Phase 3 simple baselines, leakage-safe query extraction, and metrics."""

import pytest
from hiver_tesco.baselines.escalation_baselines import (
    AlwaysDoNotEscalateBaseline,
    MajorityClassEscalateBaseline,
    RiskKeywordEscalationBaseline,
)
from hiver_tesco.baselines.evaluation import (
    evaluate_escalation_predictions,
    evaluate_intent_predictions,
    extract_escalation_false_negatives,
)
from hiver_tesco.baselines.intent_baselines import (
    MajorityClassIntentBaseline,
    TfidfLogRegIntentBaseline,
)
from hiver_tesco.baselines.query_extractor import extract_leakage_safe_query
from hiver_tesco.baselines.reply_templates import IntentTemplateReplyBaseline


def test_extract_leakage_safe_query():
    # Customer initiates, Tesco replies later
    messages = [
        {"turn_index": 0, "direction": "inbound", "author_id": "cust1", "text": "My milk was sour and rotten!"},
        {"turn_index": 1, "direction": "outbound", "author_id": "Tesco", "text": "Hi, please DM us for a refund."},
        {"turn_index": 2, "direction": "inbound", "author_id": "cust1", "text": "Sent DM."},
    ]
    query = extract_leakage_safe_query(messages)
    assert query == "My milk was sour and rotten!"
    assert "DM us for a refund" not in query
    assert "Sent DM" not in query


def test_extract_leakage_safe_query_broadcast_root():
    # Tesco promotional broadcast first, customer replies, then Tesco agent answers
    messages = [
        {"turn_index": 0, "direction": "outbound", "author_id": "Tesco", "text": "Try our Christmas pudding!"},
        {"turn_index": 1, "direction": "inbound", "author_id": "cust2", "text": "Why is it out of stock in Leeds?"},
        {"turn_index": 2, "direction": "outbound", "author_id": "Tesco", "text": "Hi, we will check Leeds store."},
    ]
    query = extract_leakage_safe_query(messages)
    assert query == "Why is it out of stock in Leeds?"
    assert "Christmas pudding" not in query
    assert "Leeds store" not in query


def test_majority_class_intent_baseline():
    train_texts = ["query 1", "query 2", "query 3"]
    train_labels = ["stock_and_availability", "delivery_and_orders", "delivery_and_orders"]

    model = MajorityClassIntentBaseline().fit(train_texts, train_labels)
    assert model.majority_class == "delivery_and_orders"

    preds = model.predict(["any new query", "another query"])
    assert preds == ["delivery_and_orders", "delivery_and_orders"]


def test_tfidf_logreg_intent_baseline():
    train_texts = [
        "Where is my delivery order",
        "Delivery driver arrived late and food cancelled",
        "The milk was rotten and smelled mouldy",
        "Found plastic insect in my food",
    ]
    train_labels = [
        "delivery_and_orders",
        "delivery_and_orders",
        "product_quality_and_safety",
        "product_quality_and_safety",
    ]

    model = TfidfLogRegIntentBaseline().fit(train_texts, train_labels)
    preds = model.predict(["Where is my delivery?", "The food was rotten and mouldy"])

    assert len(preds) == 2
    assert preds[0] == "delivery_and_orders"
    assert preds[1] == "product_quality_and_safety"


def test_escalation_baselines():
    train_texts = ["text 1", "text 2", "text 3"]
    train_labels = ["escalate", "escalate", "do_not_escalate"]

    # AlwaysDoNotEscalate
    always_no = AlwaysDoNotEscalateBaseline().fit(train_texts, train_labels)
    assert always_no.predict(["urgent refund please"]) == ["do_not_escalate"]

    # MajorityClassEscalate
    maj_esc = MajorityClassEscalateBaseline().fit(train_texts, train_labels)
    assert maj_esc.majority_label == "escalate"
    assert maj_esc.predict(["general compliment"]) == ["escalate"]

    # RiskKeywordEscalationBaseline
    risk_model = RiskKeywordEscalationBaseline().fit(train_texts, train_labels)

    # Positive risk triggers
    p1, triggers1 = risk_model.predict_single("I need a refund, you charged twice!")
    assert p1 == "escalate"
    assert "financial_loss_or_refund" in triggers1

    p2, triggers2 = risk_model.predict_single("Disgusting, there is mould and glass in my soup!")
    assert p2 == "escalate"
    assert "food_safety_or_foreign_body" in triggers2

    # Negative non-risk triggers
    p3, triggers3 = risk_model.predict_single("Colleague at till was very helpful today, thank you!")
    assert p3 == "do_not_escalate"
    assert len(triggers3) == 0


def test_reply_templates_conditioned_on_prediction():
    template_model = IntentTemplateReplyBaseline()

    # Reply must be conditioned on predicted values
    reply_esc = template_model.generate_reply("product_quality_and_safety", "escalate")
    assert "DM with your full name" in reply_esc
    assert "barcode/batch code" in reply_esc

    reply_no_esc = template_model.generate_reply("product_quality_and_safety", "do_not_escalate")
    assert "DM" not in reply_no_esc
    assert "feedback regarding our product range" in reply_no_esc

    # Oracle response explicitly tagged
    oracle = template_model.generate_oracle_reply("product_quality_and_safety", "escalate")
    assert oracle["oracle_diagnostic_only"] is True


def test_evaluation_metrics():
    y_true = ["delivery_and_orders", "product_quality_and_safety", "store_experience_and_staff"]
    y_pred = ["delivery_and_orders", "product_quality_and_safety", "delivery_and_orders"]

    intent_metrics = evaluate_intent_predictions(y_true, y_pred)
    assert intent_metrics["accuracy"] == pytest.approx(2 / 3, rel=1e-2)
    assert "per_class" in intent_metrics
    assert "confusion_matrix" in intent_metrics

    # Escalation metrics
    e_true = ["escalate", "do_not_escalate", "escalate"]
    e_pred = ["escalate", "do_not_escalate", "do_not_escalate"]  # 1 FN
    esc_metrics = evaluate_escalation_predictions(e_true, e_pred)

    assert esc_metrics["confusion_matrix"]["tp"] == 1
    assert esc_metrics["confusion_matrix"]["tn"] == 1
    assert esc_metrics["confusion_matrix"]["fn"] == 1
    assert esc_metrics["confusion_matrix"]["fp"] == 0
    assert esc_metrics["false_negative_rate"] == 0.5
