"""Escalation baselines: AlwaysDoNotEscalate, MajorityClassEscalate, and RiskKeywordEscalation."""

from collections import Counter
import re
from typing import Dict, List, Optional, Pattern, Tuple


class AlwaysDoNotEscalateBaseline:
    """Trivial baseline that never escalates (always predicts 'do_not_escalate')."""

    def fit(self, X_texts: List[str], y_labels: List[str]) -> "AlwaysDoNotEscalateBaseline":
        return self

    def predict(self, X_texts: List[str]) -> List[str]:
        return ["do_not_escalate"] * len(X_texts)


class MajorityClassEscalateBaseline:
    """Majority baseline that predicts the most frequent escalation label in the training set."""

    def __init__(self):
        self.majority_label: Optional[str] = None

    def fit(self, X_texts: List[str], y_labels: List[str]) -> "MajorityClassEscalateBaseline":
        counts = Counter(y_labels)
        self.majority_label = counts.most_common(1)[0][0]
        return self

    def predict(self, X_texts: List[str]) -> List[str]:
        if self.majority_label is None:
            raise RuntimeError("Model must be fitted before predict.")
        return [self.majority_label] * len(X_texts)


# Transparent risk triggers (case-insensitive regex patterns)
RISK_PATTERNS: Dict[str, Pattern] = {
    "food_safety_or_foreign_body": re.compile(
        r"(?i)\b(mold\w*|mould\w*|glass|plastic|metal|insect\w*|maggot\w*|beetle\w*|worm\w*|foreign|poison\w*|sick|vomit\w*|rotten|raw\b|sour|smell\w*|stale|taste\w* awful|undercook\w*|pink chicken|contaminat\w*|hair\b|off\b|shrink\w*|out of date|date.*pass|expir\w*)\b"
    ),
    "broken_or_faulty_product": re.compile(
        r"(?i)\b(broken|faulty|damaged|dent\w*|rip\w*|smashed|cracked|leak\w*)\b"
    ),
    "financial_loss_or_refund": re.compile(
        r"(?i)\b(refund\w*|moneycard|overcharg\w*|double charg\w*|charged twice|compensation|voucher.*waste|wasted.*voucher|cannot use.*voucher|can't use.*voucher|voucher.*not work\w*|rip ?off|charged £|charged \d)\b"
    ),
    "order_or_delivery_failure": re.compile(
        r"(?i)\b(cancel\w*.*order|order.*cancel\w*|driver.*never|where.*delivery|where.*order|missing.*item\w*|wrong.*item\w*|wait\w*.*order|wait\w*.*delivery|deliver\w*.*late|late.*deliver\w*|no delivery slot\w*)\b"
    ),
    "explicit_complaint_or_dm": re.compile(
        r"(?i)\b(dm\b|direct message|store manager|head office|formal complaint|unacceptable|sort it out|appall\w*|disgrace\w*|disgust\w*|rubbish|trading standards|ombudsman|mess\w* me around|awful service|shocking service|terrible service)\b"
    ),
}


class RiskKeywordEscalationBaseline:
    """Transparent, rule-based escalation classifier based on risk triggers.

    Evaluated strictly on leakage-safe customer query_text.
    """

    def __init__(self):
        self.patterns = RISK_PATTERNS

    def fit(self, X_texts: List[str], y_labels: List[str]) -> "RiskKeywordEscalationBaseline":
        # Rule-based model: no learned parameters, fit is a no-op for API uniformity
        return self

    def predict_single(self, text: str) -> Tuple[str, List[str]]:
        """Predict escalation decision and return matched trigger categories."""
        matched_categories = []
        for cat, pat in self.patterns.items():
            if pat.search(text):
                matched_categories.append(cat)

        if matched_categories:
            return "escalate", matched_categories
        return "do_not_escalate", []

    def predict(self, X_texts: List[str]) -> List[str]:
        return [self.predict_single(t)[0] for t in X_texts]

    def predict_with_triggers(self, X_texts: List[str]) -> List[Tuple[str, List[str]]]:
        return [self.predict_single(t) for t in X_texts]
