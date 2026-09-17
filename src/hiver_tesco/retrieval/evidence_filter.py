"""Deterministic evidence filtering for historical Tesco resolutions.

Applies configurable semantic threshold rejection, cross-brand rejection,
broadcast root rejection, and malformed conversation rejection while exposing
source IDs, timestamps, and similarity scores.
"""

from dataclasses import dataclass
from typing import List

from hiver_tesco.retrieval.retrievers import CandidateMatch


@dataclass
class FilteredEvidence:
    """Retrieved evidence record post deterministic filtering."""

    candidate: CandidateMatch
    is_accepted: bool
    rejection_reasons: List[str]
    source_conversation_id: str
    source_root_tweet_id: str
    outbound_tweet_ids: List[str]
    outbound_timestamps: List[str]
    similarity_score: float
    retrieved_evidence_text: str  # sanitized Tesco resolution
    retrieved_customer_issue: str  # sanitized customer issue


class EvidenceFilter:
    """Deterministic filter validating retrieved resolution candidates."""

    def __init__(
        self,
        semantic_threshold: float = 0.45,
        reject_cross_brand: bool = True,
        reject_broadcast_root: bool = True,
        reject_malformed: bool = True,
    ):
        self.semantic_threshold = semantic_threshold
        self.reject_cross_brand = reject_cross_brand
        self.reject_broadcast_root = reject_broadcast_root
        self.reject_malformed = reject_malformed

    def evaluate(self, candidate: CandidateMatch) -> FilteredEvidence:
        """Evaluate a candidate match against all deterministic filtering rules."""
        doc = candidate.document
        reasons = []

        # 1. Semantic score threshold check
        if candidate.retriever_type == "semantic":
            if candidate.score < self.semantic_threshold:
                reasons.append(
                    f"weak_semantic_match: score {candidate.score:.4f} < threshold {self.semantic_threshold:.4f}"
                )

        # 2. Cross-brand and competitor checks
        if self.reject_cross_brand:
            flags = doc.quality_flags
            if flags.get("has_cross_brand_mention"):
                reasons.append("cross_brand_mention_flagged")
            if doc.competitor_mentions:
                reasons.append(f"competitor_mentioned: {', '.join(doc.competitor_mentions)}")

        # 3. Broadcast root check
        if self.reject_broadcast_root:
            if doc.quality_flags.get("is_outbound_root"):
                reasons.append("broadcast_outbound_root")

        # 4. Malformed thread checks
        if self.reject_malformed:
            if doc.quality_flags.get("has_missing_parent"):
                reasons.append("malformed_missing_parent")
            if doc.quality_flags.get("has_cycle"):
                reasons.append("malformed_cycle")

        # 5. Missing outbound resolution
        if not doc.outbound_tweet_ids or not doc.sanitized_tesco_resolution.strip():
            reasons.append("missing_outbound_resolution")

        is_accepted = (len(reasons) == 0)

        return FilteredEvidence(
            candidate=candidate,
            is_accepted=is_accepted,
            rejection_reasons=reasons,
            source_conversation_id=doc.conversation_id,
            source_root_tweet_id=doc.source_root_tweet_id,
            outbound_tweet_ids=doc.outbound_tweet_ids,
            outbound_timestamps=doc.outbound_timestamps,
            similarity_score=candidate.score,
            retrieved_evidence_text=doc.sanitized_tesco_resolution,
            retrieved_customer_issue=doc.sanitized_customer_issue,
        )

    def filter_candidates(self, candidates: List[CandidateMatch]) -> List[FilteredEvidence]:
        """Evaluate and return all filtered evidence items."""
        return [self.evaluate(cand) for cand in candidates]
