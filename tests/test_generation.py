"""Unit tests for evidence-grounded LLM reply generation."""

import json
from pathlib import Path
import pytest

from hiver_tesco.generation.cache import ResponseCache
from hiver_tesco.generation.evaluation import (
    create_generation_review_dataframe,
    summarize_generation_metrics,
)
from hiver_tesco.generation.generator import GroundedReplyGenerator
from hiver_tesco.generation.models import (
    AuditRecord,
    EvidenceMatch,
    GenerationConfig,
    GenerationStatus,
    LLMMessage,
    LLMResponse,
)
from hiver_tesco.generation.prompts import build_user_prompt
from hiver_tesco.generation.providers import MockLLMProvider, OpenAICompatibleProvider
from hiver_tesco.policy.models import (
    EscalationCategory,
    PolicyAction,
    PolicyContext,
    PolicyDecision,
    ResponseGuidance,
)


@pytest.fixture
def temp_cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def temp_audit_file(tmp_path):
    return tmp_path / "audit.jsonl"


@pytest.fixture
def sample_guidance():
    return ResponseGuidance(
        allowed_actions=["provide_general_help", "explain_policy"],
        disallowed_actions=["do_not_promise_refunds", "do_not_ask_pii"],
        permitted_topic="stock_and_availability",
    )


@pytest.fixture
def sample_evidence():
    return EvidenceMatch(
        source_conversation_id="998877",
        score=0.78,
        sanitized_customer_issue="Do you have soy milk in stock at Cambridge?",
        sanitized_tesco_resolution="Hi, soy milk is stocked in all larger Extra stores. Please check with customer services.",
        metadata={},
    )


def test_policy_escalation_bypasses_generation(temp_cache_dir, temp_audit_file):
    """Verify that PolicyAction.ESCALATE halts generation immediately with zero provider calls."""
    mock_provider = MockLLMProvider()
    cache = ResponseCache(cache_dir=temp_cache_dir)
    generator = GroundedReplyGenerator(
        provider=mock_provider,
        cache=cache,
        audit_log_path=temp_audit_file,
    )

    policy_dec = PolicyDecision(
        action=PolicyAction.ESCALATE,
        primary_reason="food_safety_risk",
        rule_id="RULE_02_FOOD_SAFETY_HEALTH",
        rule_category=EscalationCategory.PRODUCT_SAFETY_AND_HEALTH.value,
        priority=1,
        matched_triggers=["mold"],
        suggested_routing_category="product_safety_investigation",
    )

    audit = generator.generate(
        conversation_id="conv_esc_1",
        query_text="My bread has mold on it",
        policy_decision=policy_dec,
        predicted_intent="product_quality_and_safety",
        evidence=None,
        evidence_accepted=False,
    )

    assert audit.policy_action == "escalate"
    assert audit.generation_status == GenerationStatus.POLICY_ESCALATED.value
    assert audit.final_draft is None
    assert audit.sanitized_prompt == "SKIPPED_POLICY_ESCALATION"
    assert audit.suggested_routing_category == "product_safety_investigation"
    assert "DM" in audit.template_baseline_reply  # Escalate template used


def test_pii_scrubbing_before_prompt_and_audit(temp_cache_dir, temp_audit_file, sample_guidance):
    """Verify that customer query is sanitized for PII before prompt, cache, or audit."""
    cache = ResponseCache(cache_dir=temp_cache_dir)
    generator = GroundedReplyGenerator(
        provider=MockLLMProvider(),
        cache=cache,
        audit_log_path=temp_audit_file,
    )

    policy_dec = PolicyDecision(
        action=PolicyAction.RESPOND,
        primary_reason="safe_inquiry",
        rule_id="DEFAULT_SAFE_RESPOND",
        rule_category=EscalationCategory.DEFAULT_AUTOMATED.value,
        priority=99,
        matched_triggers=[],
        response_guidance=sample_guidance,
    )

    raw_query = "My order 98765432 to SW1A 1AA is missing. Email john.doe@example.com or call 07123456789 @myhandle"
    audit = generator.generate(
        conversation_id="conv_pii_1",
        query_text=raw_query,
        policy_decision=policy_dec,
        predicted_intent="delivery_and_orders",
        evidence=None,
        evidence_accepted=False,
    )

    # Prompt stored in audit MUST be sanitized
    assert "98765432" not in audit.sanitized_prompt
    assert "SW1A 1AA" not in audit.sanitized_prompt
    assert "john.doe@example.com" not in audit.sanitized_prompt
    assert "07123456789" not in audit.sanitized_prompt
    assert "@myhandle" not in audit.sanitized_prompt

    assert "[ORDER_REF]" in audit.sanitized_prompt
    assert "[POSTCODE]" in audit.sanitized_prompt
    assert "[EMAIL]" in audit.sanitized_prompt
    assert "[PHONE]" in audit.sanitized_prompt
    assert "[CUSTOMER]" in audit.sanitized_prompt


def test_grounded_reply_generation_with_evidence(temp_cache_dir, temp_audit_file, sample_guidance, sample_evidence):
    """Verify grounded reply generation when accepted evidence is provided."""
    cache = ResponseCache(cache_dir=temp_cache_dir)
    generator = GroundedReplyGenerator(
        provider=MockLLMProvider(),
        cache=cache,
        audit_log_path=temp_audit_file,
    )

    policy_dec = PolicyDecision(
        action=PolicyAction.RESPOND,
        primary_reason="safe_inquiry",
        rule_id="DEFAULT_SAFE_RESPOND",
        rule_category=EscalationCategory.DEFAULT_AUTOMATED.value,
        priority=99,
        matched_triggers=[],
        response_guidance=sample_guidance,
    )

    audit = generator.generate(
        conversation_id="conv_grounded_1",
        query_text="Is soy milk stocked at Cambridge store?",
        policy_decision=policy_dec,
        predicted_intent="stock_and_availability",
        evidence=sample_evidence,
        evidence_accepted=True,
    )

    assert audit.policy_action == "respond"
    assert audit.generation_status == GenerationStatus.GENERATED.value
    assert audit.final_draft is not None
    assert "soy milk" in audit.final_draft.lower()
    assert audit.evidence_source_ids == ["998877"]
    assert audit.template_baseline_reply is not None


def test_structured_model_refusal_on_insufficient_evidence(temp_cache_dir, temp_audit_file, sample_guidance):
    """Verify that model returns strict JSON refusal when evidence is missing."""
    cache = ResponseCache(cache_dir=temp_cache_dir)
    generator = GroundedReplyGenerator(
        provider=MockLLMProvider(),
        cache=cache,
        audit_log_path=temp_audit_file,
    )

    policy_dec = PolicyDecision(
        action=PolicyAction.RESPOND,
        primary_reason="safe_inquiry",
        rule_id="DEFAULT_SAFE_RESPOND",
        rule_category=EscalationCategory.DEFAULT_AUTOMATED.value,
        priority=99,
        matched_triggers=[],
        response_guidance=sample_guidance,
    )

    # Missing evidence triggers mock provider's insufficient evidence refusal
    audit = generator.generate(
        conversation_id="conv_refuse_1",
        query_text="Do you sell rare dragonfruit from Tahiti?",
        policy_decision=policy_dec,
        predicted_intent="stock_and_availability",
        evidence=None,
        evidence_accepted=False,
    )

    assert audit.generation_status == GenerationStatus.MODEL_REFUSED_INSUFFICIENT_EVIDENCE.value
    assert audit.final_draft is None
    assert audit.policy_reason == "insufficient_evidence_to_answer_accurately"
    assert audit.suggested_routing_category == "general_customer_support"


def test_unparseable_output_fails_safe_to_escalation(temp_cache_dir, temp_audit_file, sample_guidance):
    """Verify that unparseable free text or invalid JSON triggers a safe escalation."""
    bad_output = "Here is a free text response that forgot to use JSON!"
    provider = MockLLMProvider(canned_responses={"bad query": bad_output})

    generator = GroundedReplyGenerator(
        provider=provider,
        cache=ResponseCache(cache_dir=temp_cache_dir),
        audit_log_path=temp_audit_file,
    )

    policy_dec = PolicyDecision(
        action=PolicyAction.RESPOND,
        primary_reason="safe_inquiry",
        rule_id="DEFAULT_SAFE_RESPOND",
        rule_category=EscalationCategory.DEFAULT_AUTOMATED.value,
        priority=99,
        matched_triggers=[],
        response_guidance=sample_guidance,
    )

    audit = generator.generate(
        conversation_id="conv_bad_json",
        query_text="bad query here",
        policy_decision=policy_dec,
        predicted_intent="general_feedback_and_chitchat",
        evidence=None,
        evidence_accepted=False,
    )

    # Must fail safe without throwing exception or leaking raw unformatted text
    assert audit.generation_status in (
        GenerationStatus.MODEL_ERROR.value,
        GenerationStatus.MODEL_REFUSED_INSUFFICIENT_EVIDENCE.value,
    )
    assert audit.final_draft is None
    assert audit.suggested_routing_category == "general_customer_support"


def test_response_cache_hit_and_miss(temp_cache_dir, temp_audit_file, sample_guidance, sample_evidence):
    """Verify that repeated requests hit the cache and avoid re-invoking the provider."""
    cache = ResponseCache(cache_dir=temp_cache_dir)
    generator = GroundedReplyGenerator(
        provider=MockLLMProvider(),
        cache=cache,
        audit_log_path=temp_audit_file,
    )

    policy_dec = PolicyDecision(
        action=PolicyAction.RESPOND,
        primary_reason="safe_inquiry",
        rule_id="DEFAULT_SAFE_RESPOND",
        rule_category=EscalationCategory.DEFAULT_AUTOMATED.value,
        priority=99,
        matched_triggers=[],
        response_guidance=sample_guidance,
    )

    # Call 1: Cache miss
    audit1 = generator.generate(
        conversation_id="conv_cache_test",
        query_text="Are you open on Sunday?",
        policy_decision=policy_dec,
        predicted_intent="store_experience_and_staff",
        evidence=sample_evidence,
        evidence_accepted=True,
    )
    assert audit1.cache_hit is False

    # Call 2: Cache hit
    audit2 = generator.generate(
        conversation_id="conv_cache_test",
        query_text="Are you open on Sunday?",
        policy_decision=policy_dec,
        predicted_intent="store_experience_and_staff",
        evidence=sample_evidence,
        evidence_accepted=True,
    )
    assert audit2.cache_hit is True
    assert audit2.final_draft == audit1.final_draft


def test_audit_record_serialization(temp_cache_dir, temp_audit_file, sample_guidance):
    """Verify audit record structure and file writing."""
    generator = GroundedReplyGenerator(
        provider=MockLLMProvider(),
        cache=ResponseCache(cache_dir=temp_cache_dir),
        audit_log_path=temp_audit_file,
    )

    policy_dec = PolicyDecision(
        action=PolicyAction.RESPOND,
        primary_reason="safe_inquiry",
        rule_id="DEFAULT_SAFE_RESPOND",
        rule_category=EscalationCategory.DEFAULT_AUTOMATED.value,
        priority=99,
        matched_triggers=[],
        response_guidance=sample_guidance,
    )

    generator.generate(
        conversation_id="conv_audit_1",
        query_text="General tweet hello",
        policy_decision=policy_dec,
        predicted_intent="general_feedback_and_chitchat",
        evidence=None,
        evidence_accepted=False,
    )

    assert temp_audit_file.exists()
    lines = temp_audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["conversation_id"] == "conv_audit_1"
    assert "sanitized_prompt" in record
    assert "timestamp_utc" in record
    assert "model_configuration" in record


def test_openai_provider_missing_key_error():
    """Verify that OpenAICompatibleProvider raises ValueError when API key is missing."""
    import os
    orig = os.environ.get("NON_EXISTENT_KEY")
    try:
        os.environ.pop("NON_EXISTENT_KEY", None)
        cfg = GenerationConfig(provider="openai", api_key_env_var="NON_EXISTENT_KEY")
        with pytest.raises(ValueError, match="Required environment variable"):
            OpenAICompatibleProvider(cfg)
    finally:
        if orig:
            os.environ["NON_EXISTENT_KEY"] = orig


def test_review_dataframe_and_metrics_summary():
    """Verify DataFrame conversion and summary calculation."""
    records = [
        AuditRecord(
            conversation_id="1",
            timestamp_utc="2026-09-17T12:00:00Z",
            policy_action="escalate",
            policy_rule_id="R1",
            policy_reason="food_safety",
            suggested_routing_category="product_safety",
            evidence_source_ids=[],
            evidence_similarity_scores=[],
            prompt_version="v1.0.0",
            model_configuration={},
            sanitized_prompt="",
            sanitized_raw_output="",
            final_draft=None,
            generation_status="policy_escalated",
            template_baseline_reply="template reply 1",
            cache_hit=False,
        ),
        AuditRecord(
            conversation_id="2",
            timestamp_utc="2026-09-17T12:00:00Z",
            policy_action="respond",
            policy_rule_id="DEFAULT",
            policy_reason="ok",
            suggested_routing_category=None,
            evidence_source_ids=["100"],
            evidence_similarity_scores=[0.82],
            prompt_version="v1.0.0",
            model_configuration={},
            sanitized_prompt="p",
            sanitized_raw_output="o",
            final_draft="Grounded draft 2",
            generation_status="generated",
            template_baseline_reply="template reply 2",
            cache_hit=True,
        ),
    ]

    df = create_generation_review_dataframe(records)
    assert len(df) == 2
    assert "llm_draft_reply" in df.columns
    assert "template_baseline_reply" in df.columns

    summary = summarize_generation_metrics(records)
    assert summary["total_queries"] == 2
    assert summary["policy_escalated_count"] == 1
    assert summary["drafts_generated_count"] == 1
    assert summary["cache_hits_count"] == 1
    assert summary["drafting_success_rate"] == 1.0
