"""Historical resolution corpus extraction and document models.

Constructs resolution documents strictly from eligible Tesco conversations
outside the golden evaluation set. Documents retain provenance IDs, timestamps,
and pre-resolution customer issue text alongside subsequent Tesco replies.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from hiver_tesco.baselines.query_extractor import extract_leakage_safe_query
from hiver_tesco.retrieval.pii import detect_competitor_mentions, sanitize_evidence_text


class CorpusLeakageError(Exception):
    """Raised when an excluded golden conversation ID is attempted to be indexed."""
    pass


@dataclass(frozen=True)
class HistoricalResolutionDoc:
    """Structured historical resolution record for retrieval."""

    conversation_id: str
    source_root_tweet_id: str
    started_at: str
    ended_at: str
    customer_issue: str
    tesco_resolution: str
    sanitized_customer_issue: str
    sanitized_tesco_resolution: str
    outbound_tweet_ids: List[str]
    outbound_timestamps: List[str]
    quality_flags: Dict[str, Any]
    competitor_mentions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoricalResolutionDoc":
        return cls(
            conversation_id=str(data["conversation_id"]),
            source_root_tweet_id=str(data["source_root_tweet_id"]),
            started_at=str(data["started_at"]),
            ended_at=str(data["ended_at"]),
            customer_issue=str(data["customer_issue"]),
            tesco_resolution=str(data["tesco_resolution"]),
            sanitized_customer_issue=str(data["sanitized_customer_issue"]),
            sanitized_tesco_resolution=str(data["sanitized_tesco_resolution"]),
            outbound_tweet_ids=[str(x) for x in data["outbound_tweet_ids"]],
            outbound_timestamps=[str(x) for x in data["outbound_timestamps"]],
            quality_flags=dict(data.get("quality_flags", {})),
            competitor_mentions=[str(x) for x in data.get("competitor_mentions", [])],
        )


def build_historical_corpus(
    conversations_path: Path,
    excluded_golden_ids: Set[str],
    require_clean_flags: bool = True,
) -> List[HistoricalResolutionDoc]:
    """Extract and validate eligible historical Tesco resolution documents.

    Args:
        conversations_path: Path to outputs/conversations.jsonl
        excluded_golden_ids: Set of conversation IDs to strictly exclude (the entire 200 golden set)
        require_clean_flags: If True, filters out cross-brand, broadcast roots, cycles, and missing parents.

    Returns:
        List of HistoricalResolutionDoc instances.
    """
    documents: List[HistoricalResolutionDoc] = []
    excluded_count = 0

    with open(conversations_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cid = str(c["conversation_id"])

            # 1. Golden set quarantine check
            if cid in excluded_golden_ids:
                excluded_count += 1
                continue

            # 2. Quality and eligibility checks
            flags = c.get("quality_flags", {})
            if require_clean_flags:
                if flags.get("has_cross_brand_mention"):
                    continue
                if flags.get("is_outbound_root"):
                    continue
                if flags.get("has_missing_parent") or flags.get("has_cycle"):
                    continue

            # 3. Must have outbound Tesco replies
            outbound_msgs = [
                m for m in c.get("messages", [])
                if m.get("direction") == "outbound"
            ]
            if not outbound_msgs:
                continue

            # 4. Extract customer issue text
            customer_issue = extract_leakage_safe_query(c.get("messages", []))
            if not customer_issue.strip():
                continue

            # 5. Extract subsequent Tesco outbound resolution text
            outbound_texts = [m.get("text", "").strip() for m in outbound_msgs if m.get("text")]
            tesco_resolution = " ".join(outbound_texts).strip()
            if not tesco_resolution:
                continue

            # 6. Competitor detection BEFORE handle redaction
            full_text_for_comp = f"{customer_issue} {tesco_resolution}"
            competitors = detect_competitor_mentions(full_text_for_comp)
            if require_clean_flags and competitors:
                # Reject if competitor brands/handles were mentioned
                continue

            # 7. PII Sanitization
            sanitized_cust = sanitize_evidence_text(customer_issue)
            sanitized_tesco = sanitize_evidence_text(tesco_resolution)

            outbound_ids = [str(m.get("tweet_id")) for m in outbound_msgs if m.get("tweet_id")]
            outbound_times = [str(m.get("created_at")) for m in outbound_msgs if m.get("created_at")]

            doc = HistoricalResolutionDoc(
                conversation_id=cid,
                source_root_tweet_id=str(c.get("source_root_tweet_id", cid)),
                started_at=str(c.get("started_at", "")),
                ended_at=str(c.get("ended_at", "")),
                customer_issue=customer_issue,
                tesco_resolution=tesco_resolution,
                sanitized_customer_issue=sanitized_cust,
                sanitized_tesco_resolution=sanitized_tesco,
                outbound_tweet_ids=outbound_ids,
                outbound_timestamps=outbound_times,
                quality_flags=flags,
                competitor_mentions=competitors,
            )
            documents.append(doc)

    return documents


def save_corpus_jsonl(documents: Iterable[HistoricalResolutionDoc], output_path: Path) -> None:
    """Save resolution documents to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict()) + "\n")


def load_corpus_jsonl(corpus_path: Path) -> List[HistoricalResolutionDoc]:
    """Load resolution documents from JSONL."""
    documents: List[HistoricalResolutionDoc] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(HistoricalResolutionDoc.from_dict(json.loads(line)))
    return documents
