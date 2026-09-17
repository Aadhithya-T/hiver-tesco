"""Data models representing tweets, messages, quality flags, and conversations."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Direction(str, Enum):
    """Direction of the message in the customer support context."""

    INBOUND = "inbound"      # Customer -> Tesco
    OUTBOUND = "outbound"    # Tesco -> Customer
    UNKNOWN = "unknown"      # Ambiguous / unclassified


@dataclass(frozen=True)
class RawTweet:
    """A raw tweet as ingested and validated from the dataset."""

    tweet_id: str
    author_id: str
    inbound: bool
    created_at_raw: str
    created_at: datetime
    text: str
    response_tweet_id: Optional[str]
    in_response_to_tweet_id: Optional[str]
    conversation_id: str


@dataclass
class QualityFlags:
    """Quality and anomaly flags evaluated for each conversation."""

    has_missing_parent: bool = False
    has_cycle: bool = False
    has_duplicate_tweet_id: bool = False
    has_cross_brand_mention: bool = False
    is_outbound_root: bool = False
    is_multi_customer: bool = False
    is_branching: bool = False
    has_empty_text: bool = False
    is_ambiguous_ownership: bool = False
    missing_parent_ids: List[str] = field(default_factory=list)
    cross_brand_handles: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True if the thread has no structural or semantic anomalies."""
        return not (
            self.has_missing_parent
            or self.has_cycle
            or self.has_duplicate_tweet_id
            or self.has_cross_brand_mention
            or self.is_outbound_root
            or self.has_empty_text
            or self.is_ambiguous_ownership
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThreadMessage:
    """An ordered, contextually enriched message inside a reconstructed conversation."""

    tweet_id: str
    parent_tweet_id: Optional[str]
    child_tweet_ids: List[str]
    author_id: str
    direction: Direction
    created_at: str
    text: str
    turn_index: int
    depth: int
    is_root: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["direction"] = self.direction.value
        return d


@dataclass
class ReconstructedConversation:
    """A fully reconstructed, ordered conversation ready for taxonomy discovery."""

    conversation_id: str
    source_root_tweet_id: Optional[str]
    message_count: int
    inbound_count: int
    outbound_count: int
    customer_author_ids: List[str]
    started_at: str
    ended_at: str
    duration_seconds: float
    quality_flags: QualityFlags
    messages: List[ThreadMessage]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "source_root_tweet_id": self.source_root_tweet_id,
            "message_count": self.message_count,
            "inbound_count": self.inbound_count,
            "outbound_count": self.outbound_count,
            "customer_author_ids": self.customer_author_ids,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "quality_flags": self.quality_flags.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "provenance": self.provenance,
        }
