"""Template-based reply baseline conditioned strictly on predicted intent and policy output."""

from typing import Any, Dict, List, Optional


TEMPLATES_ESCALATE: Dict[str, str] = {
    "delivery_and_orders": (
        "Hi there, I'm really sorry to hear about the trouble with your delivery/order. "
        "Could you please send us a DM with your full name, delivery postcode, and order number "
        "so we can look into this and put things right for you? Thanks."
    ),
    "product_quality_and_safety": (
        "Hi, I'm very sorry to see the poor quality of this item. "
        "Please send us a DM with your full name, address, store of purchase, and the barcode/batch code "
        "from the packaging so we can log this with our supplier and arrange a refund. Thanks."
    ),
    "stock_and_availability": (
        "Hi, thanks for checking with us. "
        "Please send us a DM with your nearest store location and postcode, and our team will check "
        "the exact stock levels and next delivery date for you."
    ),
    "store_experience_and_staff": (
        "Hi, thank you for bringing this to our attention. "
        "Please send us a DM with the store location, date/time of your visit, and any colleague details "
        "so we can pass this directly to the store manager to look into."
    ),
    "pricing_promotions_and_vouchers": (
        "Hi, sorry for the pricing/voucher confusion. "
        "Please send us a DM with a photo of your till receipt, store visited, and offer details "
        "so we can investigate and refund any difference."
    ),
    "clubcard_and_loyalty": (
        "Hi, sorry for the trouble with your Clubcard account. "
        "Could you please send us a DM with your full name, registered email address, and Clubcard number "
        "so our loyalty team can look into this for you? Thanks."
    ),
    "website_and_app_technical": (
        "Hi, sorry you're experiencing technical difficulties on our site/app. "
        "Please send us a DM with your account email, device/browser details, and a screenshot of the error "
        "so our technical team can assist."
    ),
    "general_feedback_and_chitchat": (
        "Hi there, thanks for getting in touch with Tesco! "
        "If there is an ongoing issue we can help you with, please send us a DM with more details. "
        "Otherwise, we hope you have a great day!"
    ),
}

TEMPLATES_DO_NOT_ESCALATE: Dict[str, str] = {
    "delivery_and_orders": (
        "Hi there, thank you for reaching out regarding Tesco delivery services. "
        "Delivery slots are released on our website and app. Please check online for current availability."
    ),
    "product_quality_and_safety": (
        "Hi, thanks for your feedback regarding our product range. "
        "We take product standards seriously and appreciate you taking the time to let us know."
    ),
    "stock_and_availability": (
        "Hi, thanks for checking with us! Stock availability can vary by branch format. "
        "You can check general product lines on our website or check in-store with our colleagues."
    ),
    "store_experience_and_staff": (
        "Hi, thank you so much for taking the time to share your feedback about our store and colleagues! "
        "We really appreciate your kind words. #EveryLittleHelps"
    ),
    "pricing_promotions_and_vouchers": (
        "Hi, thank you for checking with us. Promotional offers and terms can vary by region and store format. "
        "Full promotion terms and conditions are available on our website."
    ),
    "clubcard_and_loyalty": (
        "Hi, thanks for getting in touch about Clubcard. You can view your points balance, vouchers, "
        "and statements anytime by logging into the Clubcard app or website."
    ),
    "website_and_app_technical": (
        "Hi, thanks for letting us know. If you're experiencing website glitches, clearing your browser cache "
        "and cookies or updating the app often resolves the issue. Have a great day!"
    ),
    "general_feedback_and_chitchat": (
        "Hi there, thanks for reaching out to Tesco! We appreciate your tweet. "
        "Have a fantastic day! #EveryLittleHelps"
    ),
}


class IntentTemplateReplyBaseline:
    """Minimal deterministic template baseline conditioned on predicted intent and escalation policy."""

    def generate_reply(self, predicted_intent: str, predicted_escalation: str) -> str:
        """Generate a response conditioned strictly on the baseline's predicted intent and escalation."""
        intent = predicted_intent if predicted_intent in TEMPLATES_ESCALATE else "general_feedback_and_chitchat"
        if predicted_escalation == "escalate":
            return TEMPLATES_ESCALATE[intent]
        return TEMPLATES_DO_NOT_ESCALATE[intent]

    def generate_oracle_reply(self, true_intent: str, true_escalation: str) -> Dict[str, Any]:
        """Generate an oracle response based on true human labels (diagnostic only, excluded from evaluation)."""
        reply = self.generate_reply(true_intent, true_escalation)
        return {
            "oracle_reply": reply,
            "oracle_diagnostic_only": True,
            "conditioning": {"intent": true_intent, "escalation": true_escalation},
        }
