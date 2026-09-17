"""Data models for evidence-grounded LLM reply generation and auditing."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GenerationStatus(str, Enum):
    """Execution status of the reply generation pipeline."""
    POLICY_ESCALATED = "policy_escalated"
    GENERATED = "generated"
    MODEL_REFUSED_INSUFFICIENT_EVIDENCE = "model_refused_insufficient_evidence"
    MODEL_ERROR = "model_error"


@dataclass
class EvidenceMatch:
    """Convenience record for evidence passed to prompt generation."""
    source_conversation_id: str
    score: float
    sanitized_customer_issue: str
    sanitized_tesco_resolution: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_candidate_match(cls, candidate: Any) -> "EvidenceMatch":
        """Factory from CandidateMatch in retrieval package."""
        doc = candidate.document
        return cls(
            source_conversation_id=str(doc.conversation_id),
            score=float(candidate.score),
            sanitized_customer_issue=doc.sanitized_customer_issue,
            sanitized_tesco_resolution=doc.sanitized_tesco_resolution,
            metadata={"retriever_type": candidate.retriever_type, "rank": candidate.rank},
        )


@dataclass
class StructuredModelOutput:
    """Strict structured schema returned by the model."""
    status: str  # Must be 'draft' or 'escalate'
    reply: Optional[str] = None
    grounded_evidence_id: Optional[str] = None
    reason: Optional[str] = None
    suggested_routing_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMMessage:
    """A single prompt message."""
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Raw response returned by an LLM provider."""
    content: str
    raw_response: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = "unknown"
    provider_name: str = "mock"


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    provider: str = "mock"
    model: str = "mock-model"
    temperature: float = 0.0
    max_tokens: int = 150
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    prompt_version: str = "v1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditRecord:
    """Comprehensive, PII-sanitized audit record for reply generation."""
    conversation_id: str
    timestamp_utc: str
    policy_action: str
    policy_rule_id: str
    policy_reason: str
    suggested_routing_category: Optional[str]
    evidence_source_ids: List[str]
    evidence_similarity_scores: List[float]
    prompt_version: str
    model_configuration: Dict[str, Any]
    sanitized_prompt: str
    sanitized_raw_output: str
    final_draft: Optional[str]
    generation_status: str
    template_baseline_reply: str
    cache_hit: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
