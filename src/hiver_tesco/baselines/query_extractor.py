"""Leakage-safe customer query text extraction.

Extracts ONLY inbound customer messages created before any Tesco response or resolution.
Tesco replies, DM requests, and agent text are strictly excluded from query_text.
"""

from typing import Any, Dict, List


def extract_leakage_safe_query(messages: List[Dict[str, Any]]) -> str:
    """Extract a leakage-safe query string from a conversation's messages.

    Rules:
    1. Identify all messages with direction == 'inbound' (customer messages).
    2. Find the index of the first customer message.
    3. Find the index of the first Tesco outbound message that occurs AFTER the first customer message.
    4. Collect all customer messages from the first customer message up to (exclusive) that first Tesco reply.
    5. Concatenate their text content with spaces.
    6. Tesco outbound text is NEVER included.

    Returns:
        Clean, leakage-safe query text representing the customer's pre-resolution inquiry.
    """
    if not messages:
        return ""

    first_cust_idx = None
    for idx, m in enumerate(messages):
        if m.get("direction") == "inbound":
            first_cust_idx = idx
            break

    if first_cust_idx is None:
        # Fallback if no inbound message exists
        return ""

    # Find first Tesco outbound message occurring after the first customer message
    first_tesco_after_cust_idx = None
    for idx in range(first_cust_idx + 1, len(messages)):
        if m_dir := messages[idx].get("direction"):
            if m_dir == "outbound":
                first_tesco_after_cust_idx = idx
                break

    # Cutoff at first Tesco reply (or end of messages if Tesco never replied)
    cutoff = first_tesco_after_cust_idx if first_tesco_after_cust_idx is not None else len(messages)

    customer_texts = [
        messages[i].get("text", "").strip()
        for i in range(first_cust_idx, cutoff)
        if messages[i].get("direction") == "inbound" and messages[i].get("text")
    ]

    return " ".join(customer_texts).strip()
