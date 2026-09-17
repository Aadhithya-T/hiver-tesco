"""Configuration constants and pipeline parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Pattern
import re

DEFAULT_BRAND_ID: str = "Tesco"
TWITTER_DATE_FORMAT: str = "%a %b %d %H:%M:%S %z %Y"

# Known competitor supermarket/support handles frequently mentioned in cross-brand tweets
KNOWN_COMPETITOR_HANDLES: List[str] = [
    "@sainsburys",
    "@asda",
    "@morrisons",
    "@aldiuk",
    "@lidlgb",
    "@marksandspencer",
    "@waitrose",
    "@amazonhelp",
    "@bootsuk",
    "@coopuk",
    "@icelandfoods",
]

# Precompiled regex for competitor handles (note: @ is non-word character, so use (?<!\w) or direct match)
COMPETITOR_REGEX: Pattern = re.compile(
    r"(?i)(?<!\w)(" + "|".join(re.escape(h) for h in KNOWN_COMPETITOR_HANDLES) + r")\b"
)


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration settings for the extraction pipeline."""

    input_path: Path
    output_dir: Path
    brand_id: str = DEFAULT_BRAND_ID
    seed: int = 42
    sample_size: int = 200
    require_brand_participation: bool = True
    export_sample_markdown: bool = True
    export_sample_jsonl: bool = True

    def get_conversations_path(self) -> Path:
        return self.output_dir / "conversations.jsonl"

    def get_audit_json_path(self) -> Path:
        return self.output_dir / "audit_report.json"

    def get_audit_csv_path(self) -> Path:
        return self.output_dir / "audit_report.csv"

    def get_samples_dir(self) -> Path:
        return self.output_dir / "samples"
