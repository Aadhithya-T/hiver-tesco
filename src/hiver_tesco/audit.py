"""Audit reporting for tracking funnel metrics and data quality findings."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import csv
import json
from typing import Any, Dict, List
from hiver_tesco.models import ReconstructedConversation


@dataclass
class AuditReport:
    """Audit report holding pipeline stage funnel counts and data quality metrics."""

    total_input_rows: int = 0
    unique_tweets: int = 0
    duplicate_tweet_ids: int = 0
    total_conversations_discovered: int = 0
    conversations_with_brand_participation: int = 0
    conversations_reconstructed: int = 0
    clean_conversations: int = 0
    flagged_conversations: int = 0

    # Anomaly breakdown
    anomalies: Dict[str, int] = field(default_factory=lambda: {
        "missing_parents": 0,
        "cycles": 0,
        "duplicate_tweets": 0,
        "cross_brand_mentions": 0,
        "outbound_roots": 0,
        "multi_customers": 0,
        "branching_threads": 0,
        "empty_text": 0,
        "ambiguous_ownership": 0,
    })

    # Length distribution
    length_distribution: Dict[str, int] = field(default_factory=lambda: {
        "length_2": 0,
        "length_3": 0,
        "length_4": 0,
        "length_5": 0,
        "length_6": 0,
        "length_7_to_10": 0,
        "length_11_plus": 0,
    })

    # Metadata
    pipeline_version: str = "0.1.0"
    brand_id: str = "Tesco"
    seed: int = 42

    def record_conversation(self, conv: ReconstructedConversation) -> None:
        """Record quality flags and length distribution for a reconstructed conversation."""
        self.conversations_reconstructed += 1

        flags = conv.quality_flags
        if flags.is_clean:
            self.clean_conversations += 1
        else:
            self.flagged_conversations += 1

        if flags.has_missing_parent:
            self.anomalies["missing_parents"] += 1
        if flags.has_cycle:
            self.anomalies["cycles"] += 1
        if flags.has_duplicate_tweet_id:
            self.anomalies["duplicate_tweets"] += 1
        if flags.has_cross_brand_mention:
            self.anomalies["cross_brand_mentions"] += 1
        if flags.is_outbound_root:
            self.anomalies["outbound_roots"] += 1
        if flags.is_multi_customer:
            self.anomalies["multi_customers"] += 1
        if flags.is_branching:
            self.anomalies["branching_threads"] += 1
        if flags.has_empty_text:
            self.anomalies["empty_text"] += 1
        if flags.is_ambiguous_ownership:
            self.anomalies["ambiguous_ownership"] += 1

        # Length buckets
        length = conv.message_count
        if length == 2:
            self.length_distribution["length_2"] += 1
        elif length == 3:
            self.length_distribution["length_3"] += 1
        elif length == 4:
            self.length_distribution["length_4"] += 1
        elif length == 5:
            self.length_distribution["length_5"] += 1
        elif length == 6:
            self.length_distribution["length_6"] += 1
        elif 7 <= length <= 10:
            self.length_distribution["length_7_to_10"] += 1
        else:
            self.length_distribution["length_11_plus"] += 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, output_path: Path) -> None:
        """Write hierarchical audit report to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def write_csv(self, output_path: Path) -> None:
        """Write flat key-value summary table to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows: List[Tuple[str, Any]] = [
            ("total_input_rows", self.total_input_rows),
            ("unique_tweets", self.unique_tweets),
            ("duplicate_tweet_ids", self.duplicate_tweet_ids),
            ("total_conversations_discovered", self.total_conversations_discovered),
            ("conversations_with_brand_participation", self.conversations_with_brand_participation),
            ("conversations_reconstructed", self.conversations_reconstructed),
            ("clean_conversations", self.clean_conversations),
            ("flagged_conversations", self.flagged_conversations),
            # Anomalies
            ("anomaly_missing_parents", self.anomalies["missing_parents"]),
            ("anomaly_cycles", self.anomalies["cycles"]),
            ("anomaly_duplicate_tweets", self.anomalies["duplicate_tweets"]),
            ("anomaly_cross_brand_mentions", self.anomalies["cross_brand_mentions"]),
            ("anomaly_outbound_roots", self.anomalies["outbound_roots"]),
            ("anomaly_multi_customers", self.anomalies["multi_customers"]),
            ("anomaly_branching_threads", self.anomalies["branching_threads"]),
            ("anomaly_empty_text", self.anomalies["empty_text"]),
            ("anomaly_ambiguous_ownership", self.anomalies["ambiguous_ownership"]),
            # Length distribution
            ("length_2_count", self.length_distribution["length_2"]),
            ("length_3_count", self.length_distribution["length_3"]),
            ("length_4_count", self.length_distribution["length_4"]),
            ("length_5_count", self.length_distribution["length_5"]),
            ("length_6_count", self.length_distribution["length_6"]),
            ("length_7_to_10_count", self.length_distribution["length_7_to_10"]),
            ("length_11_plus_count", self.length_distribution["length_11_plus"]),
        ]
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for metric, val in rows:
                writer.writerow([metric, val])
