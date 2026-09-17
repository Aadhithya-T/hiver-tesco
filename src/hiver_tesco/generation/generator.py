"""Orchestrator for evidence-grounded reply generation and audit logging."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from hiver_tesco.baselines.reply_templates import IntentTemplateReplyBaseline
from hiver_tesco.generation.cache import ResponseCache
from hiver_tesco.generation.models import (
    AuditRecord,
    EvidenceMatch,
    GenerationConfig,
    GenerationStatus,
    LLMMessage,
    LLMResponse,
    StructuredModelOutput,
)
from hiver_tesco.generation.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from hiver_tesco.generation.providers import BaseLLMProvider, get_provider
from hiver_tesco.policy.models import PolicyAction, PolicyDecision
from hiver_tesco.retrieval.pii import sanitize_evidence_text


class GroundedReplyGenerator:
    """End-to-end evidence-grounded reply generator with strict safety gates."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        provider: Optional[BaseLLMProvider] = None,
        cache: Optional[ResponseCache] = None,
        audit_log_path: Optional[Path] = None,
    ):
        self.config = config or GenerationConfig()
        self.provider = provider or get_provider(self.config)
        self.cache = cache or ResponseCache()
        self.template_baseline = IntentTemplateReplyBaseline()
        self.audit_log_path = audit_log_path or (Path("outputs") / "generation" / "generation_audit.jsonl")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        conversation_id: str,
        query_text: str,
        policy_decision: PolicyDecision,
        predicted_intent: str,
        evidence: Optional[EvidenceMatch] = None,
        evidence_accepted: bool = False,
    ) -> AuditRecord:
        """Process a conversation query through policy verification, grounding, and drafting."""
        now_iso = datetime.now(timezone.utc).isoformat()
        template_pred_esc = "escalate" if policy_decision.action == PolicyAction.ESCALATE else "do_not_escalate"
        template_reply = self.template_baseline.generate_reply(predicted_intent, template_pred_esc)

        evidence_ids = [evidence.source_conversation_id] if evidence else []
        evidence_scores = [round(evidence.score, 4)] if evidence else []

        # --- GATE 1: Deterministic Policy Escalation Bypass ---
        if policy_decision.action == PolicyAction.ESCALATE:
            audit = AuditRecord(
                conversation_id=conversation_id,
                timestamp_utc=now_iso,
                policy_action="escalate",
                policy_rule_id=policy_decision.rule_id,
                policy_reason=policy_decision.primary_reason,
                suggested_routing_category=policy_decision.suggested_routing_category,
                evidence_source_ids=evidence_ids,
                evidence_similarity_scores=evidence_scores,
                prompt_version=self.config.prompt_version,
                model_configuration=self.config.to_dict(),
                sanitized_prompt="SKIPPED_POLICY_ESCALATION",
                sanitized_raw_output="SKIPPED_POLICY_ESCALATION",
                final_draft=None,
                generation_status=GenerationStatus.POLICY_ESCALATED.value,
                template_baseline_reply=template_reply,
                cache_hit=False,
            )
            self._write_audit(audit)
            return audit

        # --- GATE 2: PII Scrubbing of Customer Query ---
        sanitized_query = sanitize_evidence_text(query_text)

        # --- GATE 3: Build Prompts & Check Cache ---
        user_prompt = build_user_prompt(
            query_text=sanitized_query,
            policy_decision=policy_decision,
            evidence=evidence,
            evidence_status="accepted" if evidence_accepted else "insufficient",
        )
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        cache_key = self.cache.make_key(
            provider=self.config.provider,
            model=self.config.model,
            prompt_version=self.config.prompt_version,
            messages=messages,
            temperature=self.config.temperature,
        )

        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            llm_response = cached_response
            cache_hit = True
        else:
            llm_response = self.provider.generate(messages)
            cache_hit = False
            self.cache.set(cache_key, llm_response, metadata={"conversation_id": conversation_id})

        # --- GATE 4: Strict Structured Model Result Parsing ---
        raw_output = llm_response.content.strip()
        sanitized_raw_output = sanitize_evidence_text(raw_output)

        parsed_output = self._parse_structured_output(raw_output)

        if parsed_output.status == "draft" and parsed_output.reply:
            final_draft = sanitize_evidence_text(parsed_output.reply)
            status = GenerationStatus.GENERATED.value
            routing_cat = None
            reason = "grounded_reply_drafted"
        elif parsed_output.status == "escalate":
            final_draft = None
            status = GenerationStatus.MODEL_REFUSED_INSUFFICIENT_EVIDENCE.value
            routing_cat = parsed_output.suggested_routing_category or "general_customer_support"
            reason = parsed_output.reason or "insufficient_evidence_to_answer_accurately"
        else:
            final_draft = None
            status = GenerationStatus.MODEL_ERROR.value
            routing_cat = "general_customer_support"
            reason = "unparseable_model_output_fallback_escalate"

        audit = AuditRecord(
            conversation_id=conversation_id,
            timestamp_utc=now_iso,
            policy_action="respond",
            policy_rule_id=policy_decision.rule_id,
            policy_reason=reason,
            suggested_routing_category=routing_cat,
            evidence_source_ids=evidence_ids,
            evidence_similarity_scores=evidence_scores,
            prompt_version=self.config.prompt_version,
            model_configuration=self.config.to_dict(),
            sanitized_prompt=user_prompt,
            sanitized_raw_output=sanitized_raw_output,
            final_draft=final_draft,
            generation_status=status,
            template_baseline_reply=template_reply,
            cache_hit=cache_hit,
        )
        self._write_audit(audit)
        return audit

    @staticmethod
    def _parse_structured_output(text: str) -> StructuredModelOutput:
        """Parse and strictly validate structured JSON output from model."""
        # Strip potential markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                return StructuredModelOutput(status="escalate", reason="output_not_json_dict")

            status = str(data.get("status", "")).lower().strip()
            if status == "draft":
                return StructuredModelOutput(
                    status="draft",
                    reply=data.get("reply"),
                    grounded_evidence_id=str(data.get("grounded_evidence_id", "")),
                )
            elif status == "escalate":
                return StructuredModelOutput(
                    status="escalate",
                    reason=data.get("reason", "insufficient_evidence_to_answer_accurately"),
                    suggested_routing_category=data.get("suggested_routing_category", "general_customer_support"),
                )
            else:
                return StructuredModelOutput(
                    status="escalate",
                    reason=f"invalid_status_{status}_in_structured_output",
                )
        except Exception as e:
            return StructuredModelOutput(
                status="escalate",
                reason=f"malformed_json_parse_error: {e}",
            )

    def _write_audit(self, audit: AuditRecord) -> None:
        """Append record to inspectable JSONL audit file."""
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass
