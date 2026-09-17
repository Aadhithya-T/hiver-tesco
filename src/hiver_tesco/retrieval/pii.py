"""PII sanitization and competitor detection for customer-support retrieval evidence.

Ensures that competitor mentions are detected before handles are masked, and
that all customer personal identifiers (handles, order numbers, postcodes,
phones, emails) are redacted before storing or presenting retrieved evidence.
"""

import re
from typing import List, Set

# Known UK retail / grocery competitors to detect in TWCS dialogues
COMPETITOR_HANDLES = {
    "@sainsburys",
    "@asda",
    "@asdaservice",
    "@morrisons",
    "@aldiuk",
    "@lidlgb",
    "@marksandspencer",
    "@waitrose",
    "@icelandfoods",
    "@coopuk",
    "@ocado",
    "@amazonuk",
    "@bootsuk",
}

COMPETITOR_KEYWORDS = {
    "sainsburys",
    "sainsbury's",
    "asda",
    "morrisons",
    "aldi",
    "lidl",
    "waitrose",
    "marks and spencer",
    "marks & spencer",
    "iceland foods",
    "ocado",
}

# Regex patterns
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# UK postcodes e.g., SW1A 1AA, W1A 0AX, EC1A 1BB, B33 8TH, M1 1AE, etc.
RE_UK_POSTCODE = re.compile(r"\b[A-Z]{1,2}[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}\b", re.IGNORECASE)
# Order / reference / tracking numbers (6 to 16 consecutive digits)
RE_ORDER_REF = re.compile(r"\b\d{6,16}\b")
# Phone numbers (UK landline/mobile/international format)
RE_PHONE = re.compile(r"(?:\+44\s?|0)(?:7\d{3}|\d{4}|\d{3})\s?\d{3}\s?\d{3,4}\b")
# Twitter handles
RE_HANDLE = re.compile(r"@[A-Za-z0-9_]+")


def detect_competitor_mentions(text: str) -> List[str]:
    """Detect competitor supermarket handles or keywords in text BEFORE redaction.

    Returns:
        List of identified competitor mentions.
    """
    if not text:
        return []

    found = []
    text_lower = text.lower()

    # Check handles
    for handle in RE_HANDLE.findall(text_lower):
        if handle in COMPETITOR_HANDLES:
            found.append(handle)

    # Check keyword mentions
    for kw in COMPETITOR_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            if kw not in found:
                found.append(kw)

    return sorted(list(set(found)))


EXEMPT_GREETING_WORDS = {
    "there", "team", "everyone", "all", "tesco", "@tesco", "customer", "[customer]",
    "colleague", "[colleague]", "again", "you", "for", "to", "that", "if", "we", "i", "the",
}


def sanitize_evidence_text(text: str) -> str:
    """Redact customer personal identifiers from customer-support text.

    Rules:
    1. Competitor detection is assumed to be run prior to this step.
    2. Any Twitter handle other than @Tesco (case-insensitive) is replaced with [CUSTOMER].
    3. Emails are replaced with [EMAIL].
    4. UK postcodes are replaced with [POSTCODE].
    5. Order / reference numbers (6-16 digits) are replaced with [ORDER_REF].
    6. Phone numbers are replaced with [PHONE].
    7. Replace named greetings with a generic "Hi" or "Hello".
    8. Redact customer/colleague personal names to [CUSTOMER] or normalize agent sign-offs.
    """
    if not text:
        return ""

    sanitized = text

    # 1. Redact emails
    sanitized = RE_EMAIL.sub("[EMAIL]", sanitized)

    # 2. Redact phone numbers
    sanitized = RE_PHONE.sub("[PHONE]", sanitized)

    # 3. Redact UK postcodes
    sanitized = RE_UK_POSTCODE.sub("[POSTCODE]", sanitized)

    # 4. Redact order / reference numbers
    sanitized = RE_ORDER_REF.sub("[ORDER_REF]", sanitized)

    # 5. Redact handles except @Tesco
    def _mask_handle(match: re.Match) -> str:
        handle = match.group(0)
        if handle.lower() == "@tesco":
            return "@Tesco"
        return "[CUSTOMER]"

    sanitized = RE_HANDLE.sub(_mask_handle, sanitized)

    # 6. Replace named greetings with generic 'Hi' or 'Hello'
    def _sub_greeting(m: re.Match) -> str:
        greet = m.group(1).capitalize()
        name = m.group(2)
        punct = m.group(3) or ","
        if name.lower() in EXEMPT_GREETING_WORDS:
            return m.group(0)
        if greet.lower() in ("hi", "hey"):
            return f"Hi{punct}"
        return f"{greet}{punct}"

    sanitized = re.sub(
        r"\b(Hi|Hello|Hey|Good\s+(?:morning|afternoon|evening))\s+([A-Z][a-z]+)(\s*[,!.:]?)",
        _sub_greeting,
        sanitized,
    )

    def _sub_sorry(m: re.Match) -> str:
        name = m.group(1)
        punct = m.group(2) or ","
        if name.lower() in EXEMPT_GREETING_WORDS:
            return m.group(0)
        return f"Sorry{punct}"

    sanitized = re.sub(r"\bSorry\s+([A-Z][a-z]+)(\s*[,!.:])", _sub_sorry, sanitized)

    # 7. Redact personal names addressed mid-sentence to [CUSTOMER]
    def _sub_address(m: re.Match) -> str:
        prefix = m.group(1)
        name = m.group(2)
        punct = m.group(3) or ""
        if name.lower() not in EXEMPT_GREETING_WORDS:
            return f"{prefix} [CUSTOMER]{punct}"
        return m.group(0)

    sanitized = re.sub(
        r"\b(for you|hear you|with you|help you|ask for|speaking with|speaking to|served by|coming back to me|back to me|back to us)\s+([A-Z][a-z]+)(\s*[,!.:]?)",
        _sub_address,
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"\b(Mr|Mrs|Ms|Miss|Dr)\.?\s+([A-Z][a-z]+)\b", "[CUSTOMER]", sanitized)
    sanitized = re.sub(r"\b(my name is|i am|this is)\s+([A-Z][a-z]+)\b", r"\1 [CUSTOMER]", sanitized, flags=re.IGNORECASE)

    # 8. Normalize agent sign-offs to prevent agent name leakage
    sanitized = re.sub(r"[-–—]\s*[A-Z][a-z]+\b", "- Team", sanitized)
    sanitized = re.sub(r"\bTY\s+[A-Z][a-z]+\b", "TY - Team", sanitized)
    sanitized = re.sub(
        r"(?i)\b(thanks|regards|best wishes|best|cheers)\s*[,.]?\s*[-–—]?\s+([A-Z][a-z]+)(?:\.|\b)(?=\s*(?:https?://|#|\Z|[.!?]))",
        r"\1 - Team",
        sanitized,
    )
    sanitized = re.sub(r"(?<=\.\s)[A-Z][a-z]+(?=\s*(?:https?://|#|\Z))", "- Team", sanitized)

    # Clean up multiple whitespaces
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    return sanitized
