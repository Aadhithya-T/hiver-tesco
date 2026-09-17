"""Data models for deterministic escalation and response policy."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PolicyAction(str, Enum):
    """The two definitive routing decisions made by the policy engine."""
    RESPOND = "respond"
    ESCALATE = "escalate"


class EscalationCategory(str, Enum):
    """Categories of deterministic escalation rules."""
    SAFETY_AND_HARASSMENT = "safety_and_harassment"
    PRODUCT_SAFETY_AND_HEALTH = "product_safety_and_health"
    FINANCIAL_AND_REFUNDS = "financial_and_refunds"
    ACCOUNT_AND_SECURITY = "account_and_security"
    REPEATED_CONTACT_AND_FRUSTRATION = "repeated_contact_and_frustration"
    LEGAL_AND_REGULATORY = "legal_and_regulatory"
    RETRIEVAL_FAILURE = "retrieval_failure"
    MODEL_UNCERTAINTY = "model_uncertainty"
    STORE_AND_STAFF_POLICY = "store_and_staff_policy"
    DEFAULT_AUTOMATED = "default_automated"


@dataclass(frozen=True)
class ResponseGuidance:
    """Safe response boundaries when policy decides to respond."""
    allowed_actions: List[str]
    disallowed_actions: List[str]
    permitted_topic: str
    suggested_template: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyContext:
    """Multi-signal inputs available when customer query arrives."""
    query_text: str
    model_confidence: Optional[float] = None
    predicted_intent: Optional[str] = None
    retrieval_score: Optional[float] = None
    retrieval_accepted: Optional[bool] = None
    inbound_turn_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    """Inspectable routing decision emitted by the policy engine."""
    action: PolicyAction
    primary_reason: str
    rule_id: str
    rule_category: str
    priority: int
    matched_triggers: List[str]
    response_guidance: Optional[ResponseGuidance] = None
    suggested_routing_category: Optional[str] = None  # Project routing labels, not verified Tesco team names
    version: str = "1.0.0"

    @property
    def suggested_queue(self) -> Optional[str]:
        """Deprecated alias for suggested_routing_category."""
        return self.suggested_routing_category

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["action"] = self.action.value
        return res
