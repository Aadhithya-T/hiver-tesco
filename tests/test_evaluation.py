"""Hermetic tests for Phase 7 evaluation modules.

Zero live API calls. All tests use MockPairwiseJudge and synthetic data.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hiver_tesco.evaluation.automated_checks import (
    audit_evidence_linkage,
    audit_policy_compliance,
    compute_near_copy_rate,
    compute_response_length,
    detect_unsafe_claims,
    run_all_automated_checks,
)
from hiver_tesco.evaluation.disagreement_analyzer import (
    build_disagreement_summary,
    classify_disagreements,
)
from hiver_tesco.evaluation.escalation_evaluation import (
    build_escalation_review_table,
    evaluate_escalation_routing,
)
from hiver_tesco.evaluation.human_comparison import (
    _wilson_score_ci,
    compute_judge_human_agreement,
    compute_preference_stats,
    generate_blinded_annotation_csv,
)
from hiver_tesco.evaluation.pairwise_judge import (
    BlindedPairwiseJudge,
    MockPairwiseJudge,
    MitigatedResult,
    compute_position_bias_metrics,
    parse_judge_output,
    summarize_judge_preferences,
)
from hiver_tesco.evaluation.sample_selection import (
    select_evaluation_subset,
    validate_evaluation_subset,
)


# --- Fixtures ---

def _make_golden_df():
    """Create a minimal golden DataFrame for testing."""
    intents = [
        "product_quality_and_safety", "general_feedback_and_chitchat",
        "store_experience_and_staff", "delivery_and_orders",
        "stock_and_availability", "pricing_promotions_and_vouchers",
        "website_and_app_technical", "clubcard_and_loyalty",
    ]
    rows = []
    for i in range(80):
        intent = intents[i % 8]
        esc = "escalate" if i % 3 == 0 else "do_not_escalate"
        rows.append({
            "conversation_id": str(1000 + i),
            "message_count": 3 + (i % 10),
            "duration_minutes": 10,
            "flags": "",
            "dialogue_text": f"Test dialogue {i}",
            "primary_intent": intent,
            "escalation_needed": esc,
            "escalation_rationale": "test" if esc == "escalate" else "none",
            "evidence_notes": "",
            "ambiguity_notes": "borderline" if i % 7 == 0 else "",
        })
    return pd.DataFrame(rows)


def _make_audit_records(conversation_ids, respond_ids=None):
    """Create synthetic audit records."""
    if respond_ids is None:
        respond_ids = set()

    records = []
    for cid in conversation_ids:
        if cid in respond_ids:
            records.append({
                "conversation_id": cid,
                "policy_action": "respond",
                "policy_rule_id": "DEFAULT_SAFE_RESPOND",
                "policy_reason": "safe_inquiry",
                "suggested_routing_category": None,
                "evidence_source_ids": ["ev_123"],
                "evidence_similarity_scores": [0.72],
                "generation_status": "generated",
                "final_draft": f"Hi, thanks for getting in touch about conv {cid}.",
                "template_baseline_reply": "Hi there, thanks for reaching out to Tesco!",
                "sanitized_prompt": f'CUSTOMER INQUIRY (Sanitized): "Test query for {cid}" POLICY GUIDANCE: respond',
                "cache_hit": False,
                "sentiment_conflict_detected": False,
                "sentiment_conflict_reason": None,
            })
        else:
            records.append({
                "conversation_id": cid,
                "policy_action": "escalate",
                "policy_rule_id": "RULE_FINANCIAL_001",
                "policy_reason": "refund_request",
                "suggested_routing_category": "refunds_team",
                "evidence_source_ids": [],
                "evidence_similarity_scores": [],
                "generation_status": "policy_escalated",
                "final_draft": None,
                "template_baseline_reply": "Please DM us with details.",
                "sanitized_prompt": "SKIPPED_POLICY_ESCALATION",
                "cache_hit": False,
                "sentiment_conflict_detected": False,
                "sentiment_conflict_reason": None,
            })
    return records


# === Automated Checks Tests ===


def test_response_length():
    assert compute_response_length("")["word_count"] == 0
    assert compute_response_length("Hello world test")["word_count"] == 3
    assert compute_response_length("Hello world test")["char_count"] == 16


def test_policy_compliance_escalated():
    rec = {"policy_action": "escalate", "final_draft": None, "generation_status": "policy_escalated"}
    result = audit_policy_compliance(rec)
    assert result["policy_compliant"] is True

    rec_bad = {"policy_action": "escalate", "final_draft": "oops", "generation_status": "generated"}
    result_bad = audit_policy_compliance(rec_bad)
    assert result_bad["policy_compliant"] is False
    assert result_bad["violation"] == "escalated_but_draft_present"


def test_policy_compliance_responded():
    rec = {"policy_action": "respond", "final_draft": "Hi there!", "generation_status": "generated"}
    assert audit_policy_compliance(rec)["policy_compliant"] is True

    rec_bad = {"policy_action": "respond", "final_draft": None, "generation_status": "model_error"}
    assert audit_policy_compliance(rec_bad)["policy_compliant"] is False


def test_evidence_linkage_in_audit_not_reply():
    """Evidence linkage checks audit record, not customer-facing reply text."""
    rec_good = {
        "policy_action": "respond",
        "evidence_source_ids": ["12345"],
        "evidence_similarity_scores": [0.72],
        "final_draft": "Hi, thanks for contacting us.",  # No citations in reply
    }
    result = audit_evidence_linkage(rec_good)
    assert result["evidence_linked"] is True

    rec_missing = {
        "policy_action": "respond",
        "evidence_source_ids": [],
        "evidence_similarity_scores": [],
    }
    result_bad = audit_evidence_linkage(rec_missing)
    assert result_bad["evidence_linked"] is False


def test_unsafe_claims_detection():
    clean = "Hi, thanks for your feedback. We appreciate you letting us know."
    assert detect_unsafe_claims(clean) == []

    financial = "We will refund you for this item."
    assert "unauthorized_financial_promise" in detect_unsafe_claims(financial)

    account = "I've checked your account and updated your profile."
    claims = detect_unsafe_claims(account)
    assert "account_access_illusion" in claims

    order = "Your driver is on the way!"
    assert "order_status_illusion" in detect_unsafe_claims(order)


def test_near_copy_rate():
    draft = "Hi there thanks for getting in touch"
    evidence = "Hi there thanks for getting in touch with Tesco"
    rate = compute_near_copy_rate(draft, evidence)
    assert 0.7 < rate < 1.0  # High overlap but not identical

    assert compute_near_copy_rate("", "something") == 0.0
    assert compute_near_copy_rate("completely different words here", "nothing similar at all whatsoever") < 0.3


def test_run_all_automated_checks():
    records = _make_audit_records(["100", "101"], respond_ids={"100"})
    results = run_all_automated_checks(records)
    assert len(results) == 2
    assert results[0]["policy_compliant"] is True
    assert results[1]["policy_compliant"] is True


# === Sample Selection Tests ===


def test_stratified_sample_produces_correct_count():
    golden_df = _make_golden_df()
    dev_ids = [str(1000 + i) for i in range(40)]
    test_ids = [str(1000 + i) for i in range(40, 80)]

    full_60, dev_20, test_40 = select_evaluation_subset(golden_df, dev_ids, test_ids)

    assert len(full_60) == 60
    assert len(dev_20) == 20
    assert len(test_40) == 40


def test_stratified_sample_covers_all_intents():
    golden_df = _make_golden_df()
    dev_ids = [str(1000 + i) for i in range(40)]
    test_ids = [str(1000 + i) for i in range(40, 80)]

    full_60, _, _ = select_evaluation_subset(golden_df, dev_ids, test_ids)
    validation = validate_evaluation_subset(full_60)

    assert validation["all_8_intents_covered"] is True
    assert validation["both_escalation_labels"] is True


def test_stratified_sample_is_deterministic():
    golden_df = _make_golden_df()
    dev_ids = [str(1000 + i) for i in range(40)]
    test_ids = [str(1000 + i) for i in range(40, 80)]

    full_1, _, _ = select_evaluation_subset(golden_df, dev_ids, test_ids)
    full_2, _, _ = select_evaluation_subset(golden_df, dev_ids, test_ids)

    assert list(full_1["conversation_id"]) == list(full_2["conversation_id"])


# === Blinded Annotation Tests ===


def test_blinded_csv_has_no_intent_or_split_columns():
    golden_df = _make_golden_df()
    dev_ids = [str(1000 + i) for i in range(40)]
    test_ids = [str(1000 + i) for i in range(40, 80)]

    full_60, _, _ = select_evaluation_subset(golden_df, dev_ids, test_ids)
    respond_ids = {str(1000 + i) for i in range(0, 80, 2)}
    audit_records = _make_audit_records(
        [str(r) for r in full_60["conversation_id"]], respond_ids=respond_ids
    )

    blinded_df, mapping = generate_blinded_annotation_csv(full_60, audit_records)

    # Blinded CSV must NOT have these columns
    assert "primary_intent" not in blinded_df.columns
    assert "escalation_needed" not in blinded_df.columns
    assert "split" not in blinded_df.columns
    assert "which_is_generated" not in blinded_df.columns

    # Must have these columns
    assert "pair_id" in blinded_df.columns
    assert "conversation_id" in blinded_df.columns
    assert "candidate_a" in blinded_df.columns
    assert "candidate_b" in blinded_df.columns
    assert "human_preference" in blinded_df.columns

    # Mapping has metadata
    for pair_id, m in mapping.items():
        assert "primary_intent" in m
        assert "escalation_needed" in m
        assert "split" in m
        assert "which_is_generated" in m


def test_blinded_csv_only_includes_respond_cases():
    golden_df = _make_golden_df()
    dev_ids = [str(1000 + i) for i in range(40)]
    test_ids = [str(1000 + i) for i in range(40, 80)]

    full_60, _, _ = select_evaluation_subset(golden_df, dev_ids, test_ids)
    respond_ids = {str(1000 + i) for i in range(0, 80, 3)}  # Only 1/3 respond
    audit_records = _make_audit_records(
        [str(r) for r in full_60["conversation_id"]], respond_ids=respond_ids
    )

    blinded_df, mapping = generate_blinded_annotation_csv(full_60, audit_records)

    # Only respond cases should appear
    assert len(blinded_df) < 60
    assert len(blinded_df) == len(mapping)

    for pair_id, m in mapping.items():
        conv_id = m["conversation_id"]
        rec = next(r for r in audit_records if str(r["conversation_id"]) == conv_id)
        assert rec["policy_action"] == "respond"


# === Pairwise Judge Tests ===


def test_judge_output_parsing():
    valid = '{"preference": "A", "confidence": 4, "rationale": "Better tone"}'
    result = parse_judge_output(valid)
    assert result["preference"] == "A"
    assert result["confidence"] == 4

    valid_tie = '{"preference": "Tie", "confidence": 2, "rationale": "Equal"}'
    result_tie = parse_judge_output(valid_tie)
    assert result_tie["preference"] == "Tie"

    with pytest.raises(ValueError):
        parse_judge_output("not json at all")

    with pytest.raises(ValueError):
        parse_judge_output('{"preference": "C", "confidence": 3, "rationale": "invalid"}')


def test_blinded_swap_mechanics():
    """Test forward/reversed ordering and consensus logic."""
    mock_judge = MockPairwiseJudge()
    blinded = BlindedPairwiseJudge(mock_judge)

    # Shorter generated reply should win both passes
    result = blinded.evaluate_pair(
        conversation_id="test_1",
        query="Need help",
        generated_reply="Short reply.",
        template_reply="This is a longer template reply with more words for comparison.",
    )

    assert result.forward_judgment.position_a_source == "generated"
    assert result.forward_judgment.position_b_source == "template"
    assert result.reversed_judgment.position_a_source == "template"
    assert result.reversed_judgment.position_b_source == "generated"

    # Mock judge always prefers shorter → generated wins both passes
    assert result.forward_preference == "generated"
    assert result.reversed_preference == "generated"
    assert result.mitigated_winner == "generated"
    assert result.is_consistent is True


def test_inconsistent_swap_produces_tie():
    """When forward and reversed disagree, result is tie_inconsistent."""

    class PositionBiasedJudge(MockPairwiseJudge):
        """Always picks Position A regardless of content."""
        def judge(self, query, candidate_a, candidate_b):
            return {"preference": "A", "confidence": 3, "rationale": "Position A is better"}

    biased = BlindedPairwiseJudge(PositionBiasedJudge())
    result = biased.evaluate_pair(
        "test_bias", "Query", "Generated text", "Template text"
    )

    # Forward: A=generated → "generated"; Reversed: A=template → "template"
    assert result.forward_preference == "generated"
    assert result.reversed_preference == "template"
    assert result.mitigated_winner == "tie_inconsistent"
    assert result.is_consistent is False


def test_position_bias_metrics():
    """Test position-bias calculation."""
    results = [
        MitigatedResult(
            conversation_id="c1",
            forward_preference="generated",
            reversed_preference="generated",
            mitigated_winner="generated",
            is_consistent=True,
        ),
        MitigatedResult(
            conversation_id="c2",
            forward_preference="template",
            reversed_preference="template",
            mitigated_winner="template",
            is_consistent=True,
        ),
    ]
    # Need to set judgments for position-A counting
    from hiver_tesco.evaluation.pairwise_judge import PairwiseJudgment
    results[0].forward_judgment = PairwiseJudgment("c1", "forward", "generated", "template", "A", 4, "")
    results[0].reversed_judgment = PairwiseJudgment("c1", "reversed", "template", "generated", "B", 4, "")
    results[1].forward_judgment = PairwiseJudgment("c2", "forward", "generated", "template", "B", 3, "")
    results[1].reversed_judgment = PairwiseJudgment("c2", "reversed", "template", "generated", "A", 3, "")

    metrics = compute_position_bias_metrics(results)
    assert metrics["total_pairs"] == 2
    assert metrics["total_trials"] == 4
    assert metrics["position_a_wins"] == 2  # c1 forward A + c2 reversed A
    assert metrics["position_a_win_rate"] == 0.5
    assert metrics["swap_consistency_rate"] == 1.0
    assert metrics["flip_rate"] == 0.0  # No case picked A in both forward and reversed


# === Wilson CI Tests ===


def test_wilson_ci_known_values():
    """Test Wilson score CI against known statistical values."""
    # 50% success rate with n=100
    lower, upper = _wilson_score_ci(50, 100)
    assert 0.39 < lower < 0.42
    assert 0.58 < upper < 0.61

    # Edge: 0 successes
    lower_0, upper_0 = _wilson_score_ci(0, 20)
    assert lower_0 == 0.0
    assert 0.0 < upper_0 < 0.2

    # Edge: all successes
    lower_all, upper_all = _wilson_score_ci(20, 20)
    assert 0.8 < lower_all < 1.0
    assert upper_all == 1.0

    # Edge: empty
    assert _wilson_score_ci(0, 0) == (0.0, 0.0)


# === Preference Stats Tests ===


def test_preference_stats_separate_splits():
    """Verify Dev and Test are reported separately, never pooled."""
    human_df = pd.DataFrame([
        {"pair_id": "pair_100", "human_preference": "A", "human_rationale": "Better"},
        {"pair_id": "pair_101", "human_preference": "B", "human_rationale": "Clearer"},
        {"pair_id": "pair_200", "human_preference": "A", "human_rationale": "More helpful"},
    ])

    mapping = {
        "pair_100": {"conversation_id": "100", "split": "dev", "primary_intent": "delivery_and_orders",
                     "escalation_needed": "do_not_escalate", "which_is_generated": "A"},
        "pair_101": {"conversation_id": "101", "split": "dev", "primary_intent": "product_quality_and_safety",
                     "escalation_needed": "escalate", "which_is_generated": "A"},
        "pair_200": {"conversation_id": "200", "split": "test", "primary_intent": "delivery_and_orders",
                     "escalation_needed": "do_not_escalate", "which_is_generated": "B"},
    }

    stats = compute_preference_stats(human_df, mapping)

    # Dev and Test must be separate
    assert "dev" in stats
    assert "test" in stats
    assert stats["dev"]["total_pairs"] == 2
    assert stats["test"]["total_pairs"] == 1

    # Dev: pair_100 human=A, gen=A → gen wins; pair_101 human=B, gen=A → template wins
    assert stats["dev"]["generated_wins"] == 1
    assert stats["dev"]["template_wins"] == 1

    # Test: pair_200 human=A, gen=B → template wins
    assert stats["test"]["template_wins"] == 1


# === Agreement Tests (No Kappa) ===


def test_judge_human_agreement_raw_rate():
    human_df = pd.DataFrame([
        {"pair_id": "pair_1", "human_preference": "A", "human_rationale": ""},
        {"pair_id": "pair_2", "human_preference": "A", "human_rationale": ""},
    ])

    mapping = {
        "pair_1": {"conversation_id": "c1", "split": "test", "primary_intent": "x",
                   "escalation_needed": "do_not_escalate", "which_is_generated": "A"},
        "pair_2": {"conversation_id": "c2", "split": "test", "primary_intent": "y",
                   "escalation_needed": "do_not_escalate", "which_is_generated": "B"},
    }

    # Judge: c1→generated (agrees with human=A=gen), c2→generated (human=A≠gen=B → disagrees)
    judge_results = [
        {"conversation_id": "c1", "mitigated_winner": "generated"},
        {"conversation_id": "c2", "mitigated_winner": "generated"},
    ]

    agreement = compute_judge_human_agreement(human_df, judge_results, mapping)
    assert agreement["total_compared"] == 2
    assert agreement["agreements"] == 1
    assert agreement["disagreements_count"] == 1
    assert agreement["raw_agreement_rate"] == 0.5
    assert "single_reviewer_disclosure" in agreement


# === Escalation Evaluation Tests ===


def test_escalation_routing_metrics():
    audit_records = [
        {"conversation_id": "1", "policy_action": "escalate", "policy_rule_id": "RULE_FIN_001"},
        {"conversation_id": "2", "policy_action": "escalate", "policy_rule_id": "RULE_FIN_001"},
        {"conversation_id": "3", "policy_action": "respond", "policy_rule_id": "DEFAULT_SAFE"},
        {"conversation_id": "4", "policy_action": "respond", "policy_rule_id": "DEFAULT_SAFE"},
    ]

    golden_df = pd.DataFrame([
        {"conversation_id": "1", "escalation_needed": "escalate"},
        {"conversation_id": "2", "escalation_needed": "do_not_escalate"},  # FP
        {"conversation_id": "3", "escalation_needed": "escalate"},         # FN
        {"conversation_id": "4", "escalation_needed": "do_not_escalate"},
    ])

    result = evaluate_escalation_routing(audit_records, golden_df)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_negatives"] == 1
    assert result["accuracy"] == 0.5


def test_escalation_review_table():
    audit_records = [
        {"conversation_id": "1", "policy_action": "escalate", "policy_rule_id": "R1",
         "suggested_routing_category": "refunds"},
    ]
    golden_df = pd.DataFrame([{"conversation_id": "1", "escalation_needed": "escalate"}])

    table = build_escalation_review_table(audit_records, golden_df)
    assert len(table) == 1
    assert bool(table.iloc[0]["correct"]) is True
    assert table.iloc[0]["classification"] == "correct"


# === Disagreement Analysis Tests ===


def test_disagreement_classification():
    human_df = pd.DataFrame([
        {"pair_id": "pair_1", "human_preference": "A", "human_rationale": "More relevant"},
    ])

    mapping = {
        "pair_1": {"conversation_id": "c1", "split": "test", "primary_intent": "delivery_and_orders",
                   "escalation_needed": "do_not_escalate", "which_is_generated": "A"},
    }

    # Judge disagrees: prefers template
    judge_results = [
        {"conversation_id": "c1", "mitigated_winner": "template", "is_consistent": True},
    ]

    audit_records = [
        {"conversation_id": "c1", "final_draft": "Short reply.", "template_baseline_reply": "Template."},
    ]

    disagreements = classify_disagreements(human_df, judge_results, mapping, audit_records)
    assert len(disagreements) == 1
    assert disagreements.iloc[0]["human_preference"] == "generated"
    assert disagreements.iloc[0]["judge_preference"] == "template"


def test_disagreement_summary_foregrounds_human():
    df = pd.DataFrame([{
        "conversation_id": "c1", "pair_id": "p1", "split": "test",
        "primary_intent": "delivery_and_orders",
        "human_preference": "generated", "judge_preference": "template",
        "human_rationale": "More relevant", "judge_consistent": True,
        "failure_mode": "judge_prefers_boilerplate",
        "failure_description": "Judge prefers safe boilerplate",
    }])

    summary = build_disagreement_summary(df)
    assert "Human preference is the primary ground truth" in summary
    assert "Single-Reviewer Disclosure" in summary
    # Disclosure correctly mentions that Kappa is NOT reported (negative assertion)
    assert "No inter-annotator agreement metrics" in summary


# === Finalize Stage Test ===


def test_finalize_makes_no_api_calls():
    """The finalize stage only reads saved files and computes stats.
    This test verifies the computation pipeline works with no providers.
    """
    human_df = pd.DataFrame([
        {"pair_id": "pair_1", "human_preference": "A", "human_rationale": "Good"},
    ])
    mapping = {
        "pair_1": {"conversation_id": "c1", "split": "test", "primary_intent": "x",
                   "escalation_needed": "do_not_escalate", "which_is_generated": "A"},
    }
    judge_results = [{"conversation_id": "c1", "mitigated_winner": "generated", "is_consistent": True}]
    audit_records = [{"conversation_id": "c1", "final_draft": "Reply.", "template_baseline_reply": "Template."}]

    # All computations work without any API calls
    stats = compute_preference_stats(human_df, mapping)
    agreement = compute_judge_human_agreement(human_df, judge_results, mapping)
    disagreements = classify_disagreements(human_df, judge_results, mapping, audit_records)
    summary = build_disagreement_summary(disagreements)

    assert stats["test"]["total_pairs"] == 1
    assert agreement["raw_agreement_rate"] == 1.0
    assert len(disagreements) == 0
    assert "No disagreements" in summary
