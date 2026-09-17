"""Configurable LLM provider interfaces with strict zero-credential repository guarantees."""

from abc import ABC, abstractmethod
import json
import os
import re
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from pathlib import Path

from hiver_tesco.generation.models import GenerationConfig, LLMMessage, LLMResponse


def load_dotenv():
    """Load key-value pairs from .env file into os.environ if not already set."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k not in os.environ:
                            os.environ[k] = v


class BaseLLMProvider(ABC):
    """Abstract base provider for LLM generation."""

    def __init__(self, config: GenerationConfig):
        self.config = config

    @abstractmethod
    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response from list of messages."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock provider for offline integration checks and testing.

    Proves that the generation pipeline, PII scrubbing, caching, and audit logging
    work end-to-end without incurring API costs or requiring network access.
    Does NOT claim LLM reply quality (termed 'offline integration check').
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        canned_responses: Optional[Dict[str, str]] = None,
    ):
        super().__init__(config or GenerationConfig(provider="mock", model="mock-pipeline-v1"))
        self.canned_responses: Dict[str, str] = canned_responses or {}

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        user_content = ""
        for m in messages:
            if m.role == "user":
                user_content = m.content

        # 1. Check direct canned response triggers
        for trigger, canned in self.canned_responses.items():
            if trigger.lower() in user_content.lower():
                return LLMResponse(
                    content=canned,
                    raw_response={"mock_rule": "canned_trigger", "trigger": trigger},
                    usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
                    model=self.config.model,
                    provider_name="mock",
                )

        # 2. Check if user content explicitly indicates insufficient evidence
        if "NO HISTORICAL EVIDENCE AVAILABLE" in user_content or "evidence_status: insufficient" in user_content:
            refusal_payload = {
                "status": "escalate",
                "reason": "insufficient_evidence_to_answer_accurately",
                "suggested_routing_category": "general_customer_support",
            }
            return LLMResponse(
                content=json.dumps(refusal_payload),
                raw_response={"mock_rule": "insufficient_evidence_fallback"},
                usage={"prompt_tokens": 60, "completion_tokens": 25, "total_tokens": 85},
                model=self.config.model,
                provider_name="mock",
            )

        # 3. Extract evidence source ID and resolution if available in the prompt
        ev_id_match = re.search(r"Evidence Source ID:\s*(\d+)", user_content)
        ev_id = ev_id_match.group(1) if ev_id_match else "generic_guidance"

        ev_res_match = re.search(r"Historical Tesco Resolution:\s*(.+?)(?:\n\n|\n-|\Z)", user_content, re.DOTALL)
        historical_res = ev_res_match.group(1).strip() if ev_res_match else ""

        # Construct grounded Tesco-style response based on evidence
        if historical_res and len(historical_res) > 10:
            # Ground directly in the historical resolution text
            reply_text = historical_res
            if not reply_text.endswith((".", "!", "?")):
                reply_text += "."
            if "#EveryLittleHelps" not in reply_text and len(reply_text) < 220:
                reply_text += " #EveryLittleHelps"
        else:
            reply_text = (
                "Hi there, thanks for getting in touch with Tesco. "
                "Please check our website or app for the latest updates on our services. "
                "#EveryLittleHelps"
            )

        draft_payload = {
            "status": "draft",
            "reply": reply_text,
            "grounded_evidence_id": ev_id,
        }

        return LLMResponse(
            content=json.dumps(draft_payload),
            raw_response={"mock_rule": "grounded_evidence_synthesis", "ev_id": ev_id},
            usage={"prompt_tokens": 80, "completion_tokens": 45, "total_tokens": 125},
            model=self.config.model,
            provider_name="mock",
        )


import time


class OpenAICompatibleProvider(BaseLLMProvider):
    """Configurable HTTP provider connecting to OpenAI-compatible endpoints.

    Supports OpenAI, Groq, Ollama, OpenRouter, vLLM, DeepSeek, and Gemini endpoints.
    Strictly loads credentials from environment variables; NEVER stores keys in repo.
    """

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        load_dotenv()
        self.api_key = os.environ.get(config.api_key_env_var)
        if not self.api_key and not (config.base_url and "localhost" in config.base_url):
            raise ValueError(
                f"Required environment variable '{config.api_key_env_var}' is not set. "
                "Cannot initialize live model provider without API credentials. "
                "Use --provider mock for offline integration checks."
            )
        self.endpoint = config.base_url or "https://api.openai.com/v1/chat/completions"

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data_bytes, headers=headers, method="POST")

        max_retries = 5
        backoff = 3.0
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                choice = result["choices"][0]
                content = choice["message"]["content"]
                usage = result.get("usage", {})
                return LLMResponse(
                    content=content,
                    raw_response=result,
                    usage=usage,
                    model=self.config.model,
                    provider_name="openai_compatible",
                )
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8")
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"OpenAI API HTTP Error {e.code}: {err_msg}")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"Failed to connect to LLM endpoint: {e}")


class GeminiNativeProvider(BaseLLMProvider):
    """Direct provider connecting to Google Gemini REST API.

    Uses native JSON mode (responseMimeType='application/json') with automatic retry.
    """

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        load_dotenv()
        self.api_key = os.environ.get(config.api_key_env_var or "GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                f"Required environment variable '{config.api_key_env_var or 'GEMINI_API_KEY'}' is not set. "
                "Cannot initialize Gemini provider without API credentials."
            )
        self.model = config.model or "gemini-3.5-flash"
        if self.model in ("mock-model", "gpt-4o-mini"):
            self.model = "gemini-3.5-flash"

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        system_text = ""
        user_text = ""
        for m in messages:
            if m.role == "system":
                system_text += m.content + "\n\n"
            else:
                user_text += m.content + "\n\n"

        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_text.strip()}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": 400,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": 0},
            }
        }
        if system_text.strip():
            payload["system_instruction"] = {"parts": [{"text": system_text.strip()}]}

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST")

        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                usage_meta = result.get("usageMetadata", {})
                usage = {
                    "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                    "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                    "total_tokens": usage_meta.get("totalTokenCount", 0),
                }
                return LLMResponse(
                    content=content,
                    raw_response=result,
                    usage=usage,
                    model=self.model,
                    provider_name="gemini_native",
                )
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8")
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"Gemini API HTTP Error {e.code}: {err_msg}")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"Failed to connect to Gemini API: {e}")


def get_provider(config: GenerationConfig) -> BaseLLMProvider:
    """Factory to instantiate the appropriate provider based on configuration."""
    provider_type = config.provider.lower()
    if provider_type == "mock":
        return MockLLMProvider(config)
    elif provider_type == "gemini":
        return GeminiNativeProvider(config)
    elif provider_type in ("openai", "openai_compatible", "groq", "ollama", "openrouter"):
        return OpenAICompatibleProvider(config)
    else:
        raise ValueError(f"Unsupported provider type '{config.provider}'. Choose 'mock', 'openai', or 'gemini'.")
