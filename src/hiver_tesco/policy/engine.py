"""Deterministic policy engine evaluating queries across prioritized rule chain."""

from typing import List, Optional

from hiver_tesco.policy.models import PolicyAction, PolicyContext, PolicyDecision
from hiver_tesco.policy.rules import (
    create_safe_respond_decision,
    evaluate_account_security_pii,
    evaluate_food_safety_health,
    evaluate_harassment_abuse,
    evaluate_legal_regulatory,
    evaluate_low_model_confidence,
    evaluate_payment_refund,
    evaluate_poor_retrieval_evidence,
    evaluate_staff_incident,
    evaluate_unresolved_repetition,
)


class DeterministicPolicyEngine:
    """Deterministic routing and response-policy engine for customer service."""

    def __init__(
        self,
        retrieval_threshold: float = 0.55,
        confidence_threshold: float = 0.20,
        retrieval_safety_threshold: float = 0.65,
    ):
        self.retrieval_threshold = retrieval_threshold
        self.confidence_threshold = confidence_threshold
        self.retrieval_safety_threshold = retrieval_safety_threshold

    def evaluate(self, context: PolicyContext) -> PolicyDecision:
        """Evaluate a single policy context against the prioritized rule chain.

        Priority Ordering:
        1. Harassment / Severe Safety (Priority 1)
        2. Food Safety / Health Hazard (Priority 1)
        3. Payment / Financial Loss / Refund (Priority 2)
        4. Account Security / PII Handling (Priority 2)
        5. Unresolved Repetition / Long Wait (Priority 3)
        6. Legal / Regulatory Threat (Priority 3)
        7. Poor / No Retrieval Evidence (Priority 4)
        8. Low Model Confidence (Priority 4)
        9. Store & Staff Policy Incident (Priority 5)
        10. Default Safe Respond (Fallback)
        """
        # Priority 1: Severe Safety & Harassment
        if decision := evaluate_harassment_abuse(context):
            return decision
        if decision := evaluate_food_safety_health(context):
            return decision

        # Priority 2: Financial, Account & PII
        if decision := evaluate_payment_refund(context):
            return decision
        if decision := evaluate_account_security_pii(context):
            return decision

        # Priority 3: Operational Failures & Legal
        if decision := evaluate_unresolved_repetition(context):
            return decision
        if decision := evaluate_legal_regulatory(context):
            return decision

        # Priority 4: Model & Retrieval Failure
        if decision := evaluate_poor_retrieval_evidence(context, self.retrieval_threshold):
            return decision
        if decision := evaluate_low_model_confidence(
            context, self.confidence_threshold, self.retrieval_safety_threshold
        ):
            return decision

        # Priority 5: Policy-Sensitive Store Incidents
        if decision := evaluate_staff_incident(context):
            return decision

        # Default: Safe automated response guidance
        return create_safe_respond_decision(context)

    def evaluate_batch(self, contexts: List[PolicyContext]) -> List[PolicyDecision]:
        """Evaluate a list of contexts."""
        return [self.evaluate(ctx) for ctx in contexts]
