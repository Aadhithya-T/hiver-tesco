"""Safe CSV loading, schema validation, and data quality inspection."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import csv
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from hiver_tesco.config import TWITTER_DATE_FORMAT
from hiver_tesco.models import RawTweet

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Set[str] = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "conversation_id",
}


@dataclass
class SchemaReport:
    """Summary metrics and data-quality inspection report of the raw input file."""

    file_path: str
    total_rows: int
    column_names: List[str]
    null_counts: Dict[str, int]
    inferred_types: Dict[str, str]
    unique_tweet_ids: int
    duplicate_tweet_ids: int
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    redacted_sample: List[Dict[str, Any]] = field(default_factory=list)

    def print_summary(self, file=sys.stdout) -> None:
        """Print a clean, concise schema and data-quality report to stdout or stream."""
        file.write("=" * 70 + "\n")
        file.write(f"DATASET SCHEMA & QUALITY REPORT: {self.file_path}\n")
        file.write("=" * 70 + "\n")
        file.write(f"Total Rows Ingested : {self.total_rows:,}\n")
        file.write(f"Unique Tweet IDs    : {self.unique_tweet_ids:,}\n")
        file.write(f"Duplicate Tweet IDs : {self.duplicate_tweet_ids}\n")
        if self.min_date and self.max_date:
            file.write(f"Date Range          : {self.min_date} to {self.max_date}\n")
        file.write("-" * 70 + "\n")
        file.write(f"{'Column Name':<25} {'Type':<12} {'Nulls':<10} {'Null %':<8}\n")
        file.write("-" * 70 + "\n")
        for col in self.column_names:
            nulls = self.null_counts.get(col, 0)
            pct = (nulls / self.total_rows * 100) if self.total_rows > 0 else 0.0
            col_type = self.inferred_types.get(col, "string")
            file.write(f"{col:<25} {col_type:<12} {nulls:<10} {pct:>6.2f}%\n")
        file.write("-" * 70 + "\n")
        file.write("Redacted Sample (first 3 rows):\n")
        for idx, sample in enumerate(self.redacted_sample[:3], 1):
            file.write(f"  [{idx}] tweet_id={sample.get('tweet_id')}, author={sample.get('author_id')}, inbound={sample.get('inbound')}\n")
            text_preview = sample.get("text", "").replace("\n", " ")
            file.write(f"      text: {text_preview[:80]}...\n")
        file.write("=" * 70 + "\n")


def redact_text_for_sample(text: str) -> str:
    """Redact sensitive customer mentions (@123456) for safe diagnostic logging."""
    # Mask numeric customer handles
    redacted = re.sub(r"@\d{4,}", "@[CUSTOMER_ID]", text)
    return redacted


def parse_inbound_flag(val: Any) -> bool:
    """Normalize inbound boolean string."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"true", "1", "t", "yes"}


def load_dataset(
    file_path: Path,
) -> Tuple[List[RawTweet], SchemaReport, Set[str]]:
    """Safely load and validate the raw tweets CSV.

    Returns:
        (parsed_tweets: List[RawTweet], report: SchemaReport, all_tweet_ids: Set[str])
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at path: {file_path}")

    logger.info(f"Loading raw dataset from {file_path}...")

    total_rows = 0
    column_names: List[str] = []
    null_counts: Dict[str, int] = {}
    inferred_types: Dict[str, str] = {}
    seen_tweet_ids: Set[str] = set()
    duplicate_ids_count = 0
    redacted_sample: List[Dict[str, Any]] = []

    parsed_tweets: List[RawTweet] = []
    min_dt: Optional[datetime] = None
    max_dt: Optional[datetime] = None

    with open(file_path, mode="r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or corrupted CSV file: {file_path}")

        column_names = list(reader.fieldnames)
        missing_cols = REQUIRED_COLUMNS - set(column_names)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in dataset: {missing_cols}. Found: {column_names}"
            )

        for col in column_names:
            null_counts[col] = 0

        for line_num, row in enumerate(reader, start=2):
            total_rows += 1

            for col in column_names:
                val = row.get(col)
                if val is None or val == "":
                    null_counts[col] += 1

            tid = (row.get("tweet_id") or "").strip()
            if not tid:
                logger.warning(f"Line {line_num}: Empty tweet_id, skipping.")
                continue

            if tid in seen_tweet_ids:
                duplicate_ids_count += 1
            else:
                seen_tweet_ids.add(tid)

            # Parse created_at
            raw_date = (row.get("created_at") or "").strip()
            try:
                dt = datetime.strptime(raw_date, TWITTER_DATE_FORMAT)
            except Exception as e:
                raise ValueError(
                    f"Line {line_num}: Invalid date format '{raw_date}' in tweet_id {tid}: {e}"
                )

            if min_dt is None or dt < min_dt:
                min_dt = dt
            if max_dt is None or dt > max_dt:
                max_dt = dt

            inbound_bool = parse_inbound_flag(row.get("inbound", ""))
            author = (row.get("author_id") or "").strip()
            text = row.get("text", "")
            resp_id = (row.get("response_tweet_id") or "").strip() or None
            in_reply_id = (row.get("in_response_to_tweet_id") or "").strip() or None
            conv_id = (row.get("conversation_id") or "").strip() or tid

            raw_tweet = RawTweet(
                tweet_id=tid,
                author_id=author,
                inbound=inbound_bool,
                created_at_raw=raw_date,
                created_at=dt,
                text=text,
                response_tweet_id=resp_id,
                in_response_to_tweet_id=in_reply_id,
                conversation_id=conv_id,
            )
            parsed_tweets.append(raw_tweet)

            if len(redacted_sample) < 5:
                redacted_sample.append({
                    "tweet_id": tid,
                    "author_id": author if author == "Tesco" else "[CUSTOMER_ID]",
                    "inbound": inbound_bool,
                    "created_at": raw_date,
                    "text": redact_text_for_sample(text),
                })

    # Type inference summary
    for col in column_names:
        if col in {"inbound"}:
            inferred_types[col] = "boolean"
        elif col in {"created_at"}:
            inferred_types[col] = "datetime"
        elif col in {"tweet_id", "author_id", "conversation_id", "in_response_to_tweet_id", "response_tweet_id"}:
            inferred_types[col] = "string (id)"
        else:
            inferred_types[col] = "string"

    report = SchemaReport(
        file_path=str(file_path),
        total_rows=total_rows,
        column_names=column_names,
        null_counts=null_counts,
        inferred_types=inferred_types,
        unique_tweet_ids=len(seen_tweet_ids),
        duplicate_tweet_ids=duplicate_ids_count,
        min_date=min_dt.isoformat() if min_dt else None,
        max_date=max_dt.isoformat() if max_dt else None,
        redacted_sample=redacted_sample,
    )

    return parsed_tweets, report, seen_tweet_ids
