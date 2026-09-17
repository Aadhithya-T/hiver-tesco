"""Auditable request/response cache for LLM reply generation."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from hiver_tesco.generation.models import LLMMessage, LLMResponse


class ResponseCache:
    """File-backed, inspectable cache for LLM requests and responses."""

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        self.cache_dir = cache_dir or (Path("outputs") / "generation" / "cache")
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(
        provider: str,
        model: str,
        prompt_version: str,
        messages: List[LLMMessage],
        temperature: float = 0.0,
    ) -> str:
        """Compute deterministic SHA256 digest from request payload."""
        payload = {
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "temperature": round(temperature, 4),
            "messages": [m.to_dict() for m in messages],
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> Optional[LLMResponse]:
        """Look up cached response by hash key."""
        if not self.enabled:
            return None

        file_path = self.cache_dir / f"{key}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return LLMResponse(
                content=data["content"],
                raw_response=data.get("raw_response", {}),
                usage=data.get("usage", {}),
                model=data.get("model", "cached"),
                provider_name=data.get("provider_name", "cached"),
            )
        except Exception:
            return None

    def set(
        self,
        key: str,
        response: LLMResponse,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist response and metadata to inspectable JSON cache file."""
        if not self.enabled:
            return

        file_path = self.cache_dir / f"{key}.json"
        record = {
            "key": key,
            "provider_name": response.provider_name,
            "model": response.model,
            "content": response.content,
            "usage": response.usage,
            "raw_response": response.raw_response,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
