"""Safety checks and tone-alignment validation for generated replies."""

import re
from typing import Optional, Tuple

RE_POSITIVE_QUERY = re.compile(
    r"(?i)\b("
    r"good service|great service|excellent service|superb service|fantastic service|"
    r"good experience|great experience|best service|well done|kudos|shout out|"
    r"credit to|prefer not to use self service|thank you|thanks for.*help|"
    r"love|loves|loving|favourite|favorite|cute|adorable|tasty|delicious|"
    r"happy with|pleased with|impressed with|"
    r"argue over|missus and i|funny|haha|brownies in a pack|#santapaws"
    r")\b|[😂😍😊❤️👍👏🎉]"
)

RE_COMPLAINT_WORDS_QUERY = re.compile(
    r"(?i)\b("
    r"mold|mould|broken|damaged|rotten|stale|sick|vomit|disgusting|rubbish|"
    r"late|missing|refund|overcharge|stolen|hacked"
    r")\b"
)

RE_APOLOGY_OR_DEFECT_DRAFT = re.compile(
    r"(?i)\b("
    r"sorry.*poor experience|poor experience|poor quality|bad day|bad experience|"
    r"sorry.*to hear|sorry.*to learn|sorry.*about the quality|sorry there was|sorry that we|"
    r"apologi\w*|defect|caeser salad|slug on your|refund you|issue you with a refund|"
    r"discontinued"
    r")\b"
)

RE_CHEERFUL_DRAFT = re.compile(
    r"(?i)\b("
    r"awww|how cute|fantastic|wonderful|celebrat|happy to hear"
    r")\b|[😍🎉😺]"
)


def detect_sentiment_conflict(
    query_text: str,
    draft_reply: str,
    predicted_intent: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Heuristic safety guard for tone and sentiment conflicts between customer query and drafted reply.

    Note: This is a heuristic pattern-matching safety guard rather than a guaranteed semantic classifier.
    It catches high-discrepancy failure modes where:
    1. Customer praised Tesco or engaged in banter/chitchat, but draft contains an apology
       for a defect, poor experience, or offers an unprompted refund.
    2. Customer reported a complaint/defect, but draft responds with cheerful emojis or celebration.

    Returns:
        (is_conflict, conflict_reason)
    """
    if not query_text or not draft_reply:
        return False, None

    is_positive_or_banter = bool(RE_POSITIVE_QUERY.search(query_text)) and not bool(RE_COMPLAINT_WORDS_QUERY.search(query_text))
    has_apology_or_defect_draft = bool(RE_APOLOGY_OR_DEFECT_DRAFT.search(draft_reply))

    if is_positive_or_banter and has_apology_or_defect_draft:
        return True, "query_positive_or_banter_vs_draft_apology"

    is_complaint = bool(RE_COMPLAINT_WORDS_QUERY.search(query_text))
    has_cheerful_draft = bool(RE_CHEERFUL_DRAFT.search(draft_reply))

    if is_complaint and has_cheerful_draft:
        return True, "query_complaint_vs_draft_celebration"

    return False, None
