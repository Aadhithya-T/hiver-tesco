"""Deterministic escalation and response rules.

Every rule is explicit, versioned, prioritized, and testable. Rules inspect
customer query text, model confidence, retrieval evidence quality, and
inbound turn signals available before Tesco response.
"""

import re
from typing import Callable, Dict, List, Optional, Pattern, Tuple

from hiver_tesco.policy.models import (
    EscalationCategory,
    PolicyAction,
    PolicyContext,
    PolicyDecision,
    ResponseGuidance,
)

POLICY_VERSION = "1.0.0"

# --- Compiled Regex Patterns for Risk Detection ---

# Priority 1: Harassment, severe abuse, police involvement
RE_HARASSMENT_ABUSE = re.compile(
    r"(?i)\b("
    r"fuck\w*|shit\w*|bitch\w*|bastard\w*|asshole\w*|"
    r"threat\w*|assault\w*|police|arrest\w*|crim\w*|"
    r"sue you|press charges|harass\w*|abuse\w*|violent|violence"
    r")\b"
)

# Priority 1: Food safety, contamination, physical hazards, illness, and product defects
RE_FOOD_SAFETY_HEALTH = re.compile(
    r"(?i)\b("
    r"mold\w*|mould\w*|glass|plastic|metal|insect\w*|maggot\w*|beetle\w*|worm\w*|"
    r"foreign body|foreign|poison\w*|vomit\w*|sick|nausea|diarrh\w*|allergic|allergy|"
    r"raw chicken|raw poultry|undercooked|pink chicken|salmonella|rotten|"
    r"out of date|past.*expir\w*|expir\w*.*date|expired|bad meat|sour milk|stale|"
    r"baby food.*hazard|chok\w* hazard|broken|faulty|damaged|smashed|leaking|"
    r"disgust\w*|not happy with|poor quality|terrible quality"
    r")\b"
)

# Priority 2: Financial loss, payment failure, overcharges, refunds
RE_PAYMENT_REFUND = re.compile(
    r"(?i)\b("
    r"refund\w*|reimburse\w*|overcharg\w*|double charg\w*|charged twice|"
    r"charged me £\d|charged £\d|charged \d+\.\d{2}|money back|compensation|"
    r"unauthorized|fraud\w*|card.*deducted|bank.*deducted|wasted.*voucher|"
    r"voucher.*wasted|cannot use.*voucher|can't use.*voucher|voucher.*not work\w*|"
    r"cannot use any.*voucher|can't use any.*voucher|"
    r"moneycard|stolen.*card"
    r")\b"
)

# Priority 2: Account security, password reset, PII/DM handling
RE_ACCOUNT_SECURITY_PII = re.compile(
    r"(?i)\b("
    r"hacked|compromised|login.*fail\w*|can't log\w* in|cannot log\w* in|"
    r"reset.*password|password.*reset|locked out|unlock.*account|"
    r"merg\w*.*account|account.*merg\w*|"
    r"gdpr|data protection|personal details|private data|"
    r"please dm|check my account|look into my account"
    r")\b"
)

# Priority 3: Unresolved repeated contacts & long wait times
RE_REPEATED_CONTACT = re.compile(
    r"(?i)\b("
    r"still wait\w*|waiting a week|waiting for weeks|waiting \d+ days|"
    r"called \d+ times|emailed \d+ times|contacted \d+ times|"
    r"no response|no reply|nobody.*answer\w*|ignored me|second time|"
    r"third time|4th time|5th time|mess\w* me around"
    r")\b"
)

# Priority 3: Legal & regulatory threats
RE_LEGAL_REGULATORY = re.compile(
    r"(?i)\b("
    r"trading standards|ombudsman|small claims|solicitor|legal action|"
    r"take you to court|consumer rights act"
    r")\b"
)

# Priority 5: Staff misconduct and store policy disputes
RE_STAFF_INCIDENT = re.compile(
    r"(?i)\b("
    r"rude staff|rude cashier|member of staff.*rude|staff.*abusive|"
    r"manager.*useless|disgusting service|appalling service|shocking service|"
    r"parking charge|parking fine|park watch|car park.*fine|pcb ticket"
    r")\b"
)


# --- Deterministic Rule Functions ---

def evaluate_harassment_abuse(context: PolicyContext) -> Optional[PolicyDecision]:
    matches = RE_HARASSMENT_ABUSE.findall(context.query_text)
    if matches:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            primary_reason="harassment_threat_or_safety_concern",
            rule_id="RULE_01_HARASSMENT_ABUSE",
            rule_category=EscalationCategory.SAFETY_AND_HARASSMENT.value,
            priority=1,
            matched_triggers=sorted(list(set(matches))),
            suggested_routing_category="safety_and_vulnerable_customer",
            version=POLICY_VERSION,
        )
    return None


def evaluate_food_safety_health(context: PolicyContext) -> Optional[PolicyDecision]:
    matches = RE_FOOD_SAFETY_HEALTH.findall(context.query_text)
    if matches:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            primary_reason="product_safety_contamination_or_health_risk",
            rule_id="RULE_02_FOOD_SAFETY_HEALTH",
            rule_category=EscalationCategory.PRODUCT_SAFETY_AND_HEALTH.value,
            priority=1,
            matched_triggers=sorted(list(set(matches))),
            suggested_routing_category="product_safety_investigation",
            version=POLICY_VERSION,
        )
    return None


def evaluate_payment_refund(context: PolicyContext) -> Optional[PolicyDecision]:
    matches = RE_PAYMENT_REFUND.findall(context.query_text)
    if matches:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            primary_reason="payment_dispute_overcharge_or_refund_demand",
            rule_id="RULE_03_PAYMENT_FINANCIAL_LOSS",
            rule_category=EscalationCategory.FINANCIAL_AND_REFUNDS.value,
            priority=2,
            matched_triggers=sorted(list(set(matches))),
            suggested_routing_category="billing_and_refunds",
            version=POLICY_VERSION,
        )
    return None


def evaluate_account_security_pii(context: PolicyContext) -> Optional[PolicyDecision]:
    matches = RE_ACCOUNT_SECURITY_PII.findall(context.query_text)
    if matches:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            primary_reason="account_security_login_lock_or_pii_request",
            rule_id="RULE_04_ACCOUNT_SECURITY_PII",
            rule_category=EscalationCategory.ACCOUNT_AND_SECURITY.value,
            priority=2,
            matched_triggers=sorted(list(set(matches))),
            suggested_routing_category="account_and_security",
            version=POLICY_VERSION,
        )
    return None


def evaluate_unresolved_repetition(context: PolicyContext) -> Optional[PolicyDecision]:
    matches = RE_REPEATED_CONTACT.findall(context.query_text)
    triggers = list(set(matches))
    if context.inbound_turn_count >= 2:
        triggers.append(f"inbound_turns_{context.inbound_turn_count}")

    if triggers:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            primary_reason="unresolved_repeated_contact_or_long_waiting_time",
            rule_id="RULE_05_UNRESOLVED_REPETITION",
            rule_category=EscalationCategory.REPEATED_CONTACT_AND_FRUSTRATION.value,
            priority=3,
            matched_triggers=sorted(triggers),
            suggested_routing_category="senior_customer_resolution",
            version=POLICY_VERSION,
        )
    return None


def evaluate_legal_regulatory(context: PolicyContext) -> Optional[PolicyDecision]:
    matches = RE_LEGAL_REGULATORY.findall(context.query_text)
    if matches:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            primary_reason="legal_or_regulatory_complaint_threat",
            rule_id="RULE_06_LEGAL_REGULATORY",
            rule_category=EscalationCategory.LEGAL_AND_REGULATORY.value,
            priority=3,
            matched_triggers=sorted(list(set(matches))),
            suggested_routing_category="legal_and_regulatory_affairs",
            version=POLICY_VERSION,
        )
    return None


def evaluate_poor_retrieval_evidence(
    context: PolicyContext,
    retrieval_threshold: float = 0.55,
) -> Optional[PolicyDecision]:
    triggers = []
    if context.retrieval_score is not None and context.retrieval_score < retrieval_threshold:
        triggers.append(f"retrieval_score_{context.retrieval_score:.3f}_below_{retrieval_threshold}")
    if context.retrieval_accepted is False:
        triggers.append("retrieval_evidence_rejected_by_filter")

    if triggers:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            primary_reason="poor_or_unreliable_historical_retrieval_evidence",
            rule_id="RULE_07_POOR_OR_NO_EVIDENCE",
            rule_category=EscalationCategory.RETRIEVAL_FAILURE.value,
            priority=4,
            matched_triggers=triggers,
            suggested_routing_category="general_customer_support",
            version=POLICY_VERSION,
        )
    return None


def evaluate_low_model_confidence(
    context: PolicyContext,
    confidence_threshold: float = 0.20,
    retrieval_safety_threshold: float = 0.65,
) -> Optional[PolicyDecision]:
    """Escalate when model is uncertain, unless strong historical retrieval evidence compensates."""
    if context.model_confidence is not None and context.model_confidence < confidence_threshold:
        has_strong_evidence = (
            context.retrieval_score is not None
            and context.retrieval_score >= retrieval_safety_threshold
            and context.retrieval_accepted is True
        )
        if not has_strong_evidence:
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                primary_reason=f"intent_model_uncertainty_conf_{context.model_confidence:.3f}_below_{confidence_threshold}",
                rule_id="RULE_08_LOW_MODEL_CONFIDENCE",
                rule_category=EscalationCategory.MODEL_UNCERTAINTY.value,
                priority=4,
                matched_triggers=[f"model_confidence_{context.model_confidence:.3f}"],
                suggested_routing_category="general_customer_support",
                version=POLICY_VERSION,
            )
    return None


def evaluate_staff_incident(context: PolicyContext) -> Optional[PolicyDecision]:
    matches = RE_STAFF_INCIDENT.findall(context.query_text)
    if matches:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            primary_reason="store_policy_staff_conduct_or_parking_dispute",
            rule_id="RULE_09_STAFF_INCIDENT",
            rule_category=EscalationCategory.STORE_AND_STAFF_POLICY.value,
            priority=5,
            matched_triggers=sorted(list(set(matches))),
            suggested_routing_category="store_manager_escalations",
            version=POLICY_VERSION,
        )
    return None


def create_safe_respond_decision(context: PolicyContext) -> PolicyDecision:
    """Fallback decision when all safety and escalation checks pass."""
    return PolicyDecision(
        action=PolicyAction.RESPOND,
        primary_reason="inquiry_within_safe_automated_response_boundaries",
        rule_id="DEFAULT_SAFE_RESPOND",
        rule_category=EscalationCategory.DEFAULT_AUTOMATED.value,
        priority=99,
        matched_triggers=[],
        response_guidance=ResponseGuidance(
            allowed_actions=[
                "provide_store_opening_hours_or_stock_status",
                "explain_general_company_policy_or_feature_timeline",
                "acknowledge_positive_customer_feedback_or_chitchat",
                "provide_standard_website_troubleshooting_guidance",
            ],
            disallowed_actions=[
                "do_not_promise_refunds_or_compensation",
                "do_not_request_sensitive_financial_or_personal_pii",
                "do_not_admit_legal_liability_or_negligence",
            ],
            permitted_topic=context.predicted_intent or "general_customer_inquiry",
            suggested_template="standard_informative_reply",
        ),
        suggested_routing_category=None,
        version=POLICY_VERSION,
    )
