"""Blinded pairwise LLM judging with position-bias detection.

Applied ONLY to conversations where policy action was RESPOND
(both systems — generated and template — produced a reply).
"""

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PairwiseJudgment:
    """Single-pass judgment from the pairwise judge."""
    conversation_id: str
    pass_label: str  # "forward" or "reversed"
    position_a_source: str  # "generated" or "template"
    position_b_source: str  # "generated" or "template"
    preference: str  # "A", "B", or "Tie"
    confidence: int  # 1-5
    rationale: str
    raw_output: str = ""

    def preferred_source(self) -> str:
        """Return the preferred source ('generated', 'template', or 'tie')."""
        if self.preference == "A":
            return self.position_a_source
        elif self.preference == "B":
            return self.position_b_source
        return "tie"


@dataclass
class MitigatedResult:
    """Consensus result from forward + reversed passes."""
    conversation_id: str
    forward_preference: str  # 'generated', 'template', or 'tie'
    reversed_preference: str  # 'generated', 'template', or 'tie'
    mitigated_winner: str  # 'generated', 'template', 'tie', or 'tie_inconsistent'
    is_consistent: bool
    forward_judgment: Optional[PairwiseJudgment] = None
    reversed_judgment: Optional[PairwiseJudgment] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


JUDGE_SYSTEM_PROMPT = """You are an impartial quality evaluator for customer-support replies.

You will be shown a customer's message and two candidate replies. Your task is to decide which reply would be more helpful, appropriate, and safe for the customer.

Evaluation criteria:
1. Relevance: Does the reply address the customer's specific issue?
2. Tone: Is the tone appropriate for the customer's sentiment (don't apologize for compliments)?
3. Safety: Does the reply avoid making unauthorized promises (refunds, account changes, order status)?
4. Helpfulness: Does the reply provide a useful next step without being generic?
5. Conciseness: Is the reply an appropriate length?

You MUST respond with valid JSON in this exact format:
{"preference": "A" or "B" or "Tie", "confidence": 1-5, "rationale": "brief explanation"}

Do NOT include any text outside the JSON object."""


def build_judge_prompt(query: str, candidate_a: str, candidate_b: str) -> str:
    """Build the blinded user prompt for the pairwise judge."""
    return f"""Customer Message:
\"{query}\"

Candidate A:
\"{candidate_a}\"

Candidate B:
\"{candidate_b}\"

Which candidate reply is better? Respond with JSON only."""


def parse_judge_output(raw: str) -> Dict[str, Any]:
    """Parse strict JSON from judge output. Returns parsed dict or raises ValueError."""
    raw = raw.strip()
    # Try to extract JSON from potential markdown wrapping
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge output is not valid JSON: {e}")

    pref = result.get("preference", "").upper()
    if pref not in ("A", "B", "TIE"):
        raise ValueError(f"Invalid preference '{pref}', must be A, B, or Tie")
    result["preference"] = pref if pref != "TIE" else "Tie"

    conf = result.get("confidence", 3)
    if not isinstance(conf, (int, float)) or conf < 1 or conf > 5:
        result["confidence"] = 3
    else:
        result["confidence"] = int(conf)

    return result


class BasePairwiseJudge(ABC):
    """Abstract base for pairwise judges."""

    @abstractmethod
    def judge(self, query: str, candidate_a: str, candidate_b: str) -> Dict[str, Any]:
        """Return parsed judgment dict with 'preference', 'confidence', 'rationale'."""
        pass


class MockPairwiseJudge(BasePairwiseJudge):
    """Deterministic mock judge for unit testing. Always prefers shorter reply."""

    def judge(self, query: str, candidate_a: str, candidate_b: str) -> Dict[str, Any]:
        len_a = len(candidate_a.split())
        len_b = len(candidate_b.split())
        if len_a < len_b:
            pref = "A"
            rationale = "Candidate A is more concise."
        elif len_b < len_a:
            pref = "B"
            rationale = "Candidate B is more concise."
        else:
            pref = "Tie"
            rationale = "Both candidates are equal length."

        return {"preference": pref, "confidence": 3, "rationale": rationale}


class OpenAIPairwiseJudge(BasePairwiseJudge):
    """Real LLM judge using OpenAI-compatible API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key_env_var: str = "OPENAI_API_KEY",
                 base_url: str = "https://api.openai.com/v1/chat/completions"):
        self.model = model
        self.api_key = os.environ.get(api_key_env_var, "")
        self.base_url = base_url
        if not self.api_key:
            raise ValueError(f"Environment variable '{api_key_env_var}' is not set.")

    def judge(self, query: str, candidate_a: str, candidate_b: str) -> Dict[str, Any]:
        user_prompt = build_judge_prompt(query, candidate_a, candidate_b)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url, data=data_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            raw_content = result["choices"][0]["message"]["content"]
            parsed = parse_judge_output(raw_content)
            parsed["raw_output"] = raw_content
            return parsed
        except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
            return {
                "preference": "Tie",
                "confidence": 1,
                "rationale": f"Judge API error: {e}",
                "raw_output": str(e),
            }


class BlindedPairwiseJudge:
    """Orchestrates two-pass blinded pairwise evaluation with position-bias detection.

    Pass 1 (Forward):  Position A = Generated,  Position B = Template
    Pass 2 (Reversed): Position A = Template,   Position B = Generated
    """

    def __init__(self, judge: BasePairwiseJudge):
        self.judge = judge

    def evaluate_pair(
        self,
        conversation_id: str,
        query: str,
        generated_reply: str,
        template_reply: str,
    ) -> MitigatedResult:
        """Run forward and reversed evaluation, compute mitigated result."""
        # Forward pass: A=Generated, B=Template
        fwd_result = self.judge.judge(query, generated_reply, template_reply)
        fwd_judgment = PairwiseJudgment(
            conversation_id=conversation_id,
            pass_label="forward",
            position_a_source="generated",
            position_b_source="template",
            preference=fwd_result["preference"],
            confidence=fwd_result["confidence"],
            rationale=fwd_result.get("rationale", ""),
            raw_output=fwd_result.get("raw_output", ""),
        )

        # Reversed pass: A=Template, B=Generated
        rev_result = self.judge.judge(query, template_reply, generated_reply)
        rev_judgment = PairwiseJudgment(
            conversation_id=conversation_id,
            pass_label="reversed",
            position_a_source="template",
            position_b_source="generated",
            preference=rev_result["preference"],
            confidence=rev_result["confidence"],
            rationale=rev_result.get("rationale", ""),
            raw_output=rev_result.get("raw_output", ""),
        )

        # Compute mitigated winner
        fwd_pref = fwd_judgment.preferred_source()
        rev_pref = rev_judgment.preferred_source()

        if fwd_pref == rev_pref:
            mitigated = fwd_pref
            consistent = True
        else:
            mitigated = "tie_inconsistent"
            consistent = False

        return MitigatedResult(
            conversation_id=conversation_id,
            forward_preference=fwd_pref,
            reversed_preference=rev_pref,
            mitigated_winner=mitigated,
            is_consistent=consistent,
            forward_judgment=fwd_judgment,
            reversed_judgment=rev_judgment,
        )


def compute_position_bias_metrics(results: List[MitigatedResult]) -> Dict[str, Any]:
    """Compute position-bias statistics from two-pass evaluation results.

    Returns:
        - position_a_win_rate: % of all individual trials won by Position A
        - swap_consistency_rate: % of pairs where both passes agree on winner
        - flip_rate: % of pairs where the judge favored Position A in both passes
          (regardless of which candidate was there)
    """
    if not results:
        return {
            "total_pairs": 0,
            "position_a_win_rate": 0.0,
            "swap_consistency_rate": 0.0,
            "flip_rate": 0.0,
        }

    total_pairs = len(results)
    total_trials = total_pairs * 2  # forward + reversed

    # Count Position A wins across all individual trials
    position_a_wins = 0
    for r in results:
        if r.forward_judgment and r.forward_judgment.preference == "A":
            position_a_wins += 1
        if r.reversed_judgment and r.reversed_judgment.preference == "A":
            position_a_wins += 1

    # Consistency: both passes agree on the same underlying candidate
    consistent_count = sum(1 for r in results if r.is_consistent)

    # Flip: judge picked "A" in BOTH passes (position bias)
    flip_count = 0
    for r in results:
        if (r.forward_judgment and r.reversed_judgment and
                r.forward_judgment.preference == "A" and r.reversed_judgment.preference == "A"):
            flip_count += 1

    return {
        "total_pairs": total_pairs,
        "total_trials": total_trials,
        "position_a_wins": position_a_wins,
        "position_a_win_rate": round(position_a_wins / total_trials, 4) if total_trials > 0 else 0.0,
        "consistent_pairs": consistent_count,
        "swap_consistency_rate": round(consistent_count / total_pairs, 4) if total_pairs > 0 else 0.0,
        "flip_pairs": flip_count,
        "flip_rate": round(flip_count / total_pairs, 4) if total_pairs > 0 else 0.0,
    }


def summarize_judge_preferences(results: List[MitigatedResult]) -> Dict[str, Any]:
    """Summarize mitigated preference counts."""
    if not results:
        return {"total": 0, "generated_wins": 0, "template_wins": 0, "ties": 0, "inconsistent_ties": 0}

    generated_wins = sum(1 for r in results if r.mitigated_winner == "generated")
    template_wins = sum(1 for r in results if r.mitigated_winner == "template")
    ties = sum(1 for r in results if r.mitigated_winner == "tie")
    inconsistent = sum(1 for r in results if r.mitigated_winner == "tie_inconsistent")

    total = len(results)
    return {
        "total": total,
        "generated_wins": generated_wins,
        "template_wins": template_wins,
        "ties": ties,
        "inconsistent_ties": inconsistent,
        "generated_win_rate": round(generated_wins / total, 4) if total > 0 else 0.0,
        "template_win_rate": round(template_wins / total, 4) if total > 0 else 0.0,
    }
