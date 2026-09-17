"""Human pairwise comparison: blinded annotation, preference statistics, and agreement.

Produces a truly blinded annotation CSV (no intent, escalation, or split labels)
and a separate internal mapping file for scoring.

Single-reviewer disclosure: All human annotations in this project are performed
by a single reviewer. No inter-annotator agreement metrics are reported.
"""

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _wilson_score_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a proportion.

    Returns (lower, upper) bounds.
    """
    if total == 0:
        return (0.0, 0.0)

    p_hat = successes / total
    denominator = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def generate_blinded_annotation_csv(
    eval_df: pd.DataFrame,
    audit_records: List[Dict[str, Any]],
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate truly blinded human annotation CSV and internal mapping.

    Only includes RESPOND cases where both generated and template replies exist.

    Returns:
        (blinded_df, mapping_dict)
        - blinded_df: Reviewer-facing CSV with NO intent/escalation/split columns.
        - mapping_dict: Internal mapping from pair_id to metadata (for scoring).
    """
    rng = random.Random(seed)

    audit_map = {}
    for rec in audit_records:
        audit_map[str(rec["conversation_id"])] = rec

    blinded_rows = []
    mapping = {}

    for _, row in eval_df.iterrows():
        conv_id = str(row["conversation_id"])
        rec = audit_map.get(conv_id)

        if rec is None:
            continue

        # Only include RESPOND cases with both a generated draft and template
        if rec.get("policy_action") != "respond":
            continue
        if not rec.get("final_draft"):
            continue

        generated_reply = rec["final_draft"]
        template_reply = rec.get("template_baseline_reply", "")

        if not template_reply:
            continue

        # Deterministic pair_id
        pair_id = f"pair_{conv_id}"

        # Randomize order deterministically
        order_hash = int(hashlib.sha256(f"{seed}_{conv_id}".encode()).hexdigest(), 16)
        if order_hash % 2 == 0:
            candidate_a = generated_reply
            candidate_b = template_reply
            which_is_generated = "A"
        else:
            candidate_a = template_reply
            candidate_b = generated_reply
            which_is_generated = "B"

        # Sanitized query from the audit prompt
        sanitized_query = ""
        prompt = rec.get("sanitized_prompt", "")
        if "CUSTOMER INQUIRY (Sanitized):" in prompt:
            parts = prompt.split("CUSTOMER INQUIRY (Sanitized):")
            if len(parts) > 1:
                query_part = parts[1].split("POLICY GUIDANCE:")[0].strip().strip('"')
                sanitized_query = query_part

        # Blinded row: NO primary_intent, escalation_needed, or split
        blinded_rows.append({
            "pair_id": pair_id,
            "conversation_id": conv_id,
            "sanitized_query": sanitized_query,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "human_preference": "",  # To be filled by reviewer
            "human_rationale": "",   # To be filled by reviewer
        })

        # Internal mapping (NOT shown to reviewer)
        mapping[pair_id] = {
            "conversation_id": conv_id,
            "split": row.get("split", "unknown"),
            "primary_intent": row.get("primary_intent", "unknown"),
            "escalation_needed": row.get("escalation_needed", "unknown"),
            "which_is_generated": which_is_generated,
        }

    blinded_df = pd.DataFrame(blinded_rows)
    return blinded_df, mapping


def compute_preference_stats(
    human_df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute preference statistics from completed human annotations.

    Dev and Test results are computed and reported SEPARATELY.

    Returns dict with per-split stats and per-intent breakdowns.
    """
    results = {"dev": {}, "test": {}, "overall_note": "Dev and Test are reported separately. Do not pool."}

    for split in ("dev", "test"):
        split_pairs = [pid for pid, m in mapping.items() if m.get("split") == split]
        split_df = human_df[human_df["pair_id"].isin(split_pairs)].copy()

        if len(split_df) == 0:
            results[split] = {"total_pairs": 0, "note": "no_pairs_in_split"}
            continue

        total = len(split_df)
        gen_wins = 0
        template_wins = 0
        ties = 0

        for _, row in split_df.iterrows():
            pair_id = row["pair_id"]
            pref = str(row.get("human_preference", "")).strip().upper()
            m = mapping.get(pair_id, {})
            which_gen = m.get("which_is_generated", "A")

            if pref == which_gen:
                gen_wins += 1
            elif pref == "TIE" or pref == "":
                ties += 1
            else:
                template_wins += 1

        gen_ci = _wilson_score_ci(gen_wins, total)
        template_ci = _wilson_score_ci(template_wins, total)
        tie_ci = _wilson_score_ci(ties, total)

        results[split] = {
            "total_pairs": total,
            "generated_wins": gen_wins,
            "template_wins": template_wins,
            "ties": ties,
            "generated_win_rate": round(gen_wins / total, 4) if total > 0 else 0.0,
            "generated_win_ci_95": gen_ci,
            "template_win_rate": round(template_wins / total, 4) if total > 0 else 0.0,
            "template_win_ci_95": template_ci,
            "tie_rate": round(ties / total, 4) if total > 0 else 0.0,
            "tie_ci_95": tie_ci,
        }

        # Per-intent breakdown
        intent_breakdown = {}
        for intent in set(m.get("primary_intent", "unknown") for m in mapping.values()
                          if m.get("split") == split):
            intent_pairs = [pid for pid, m2 in mapping.items()
                            if m2.get("split") == split and m2.get("primary_intent") == intent]
            intent_df = split_df[split_df["pair_id"].isin(intent_pairs)]
            if len(intent_df) == 0:
                continue

            ig = it = itie = 0
            for _, r2 in intent_df.iterrows():
                p = str(r2.get("human_preference", "")).strip().upper()
                m2 = mapping.get(r2["pair_id"], {})
                wg = m2.get("which_is_generated", "A")
                if p == wg:
                    ig += 1
                elif p == "TIE" or p == "":
                    itie += 1
                else:
                    it += 1

            intent_breakdown[intent] = {
                "pairs": len(intent_df),
                "generated_wins": ig,
                "template_wins": it,
                "ties": itie,
            }

        results[split]["by_intent"] = intent_breakdown

    return results


def compute_judge_human_agreement(
    human_df: pd.DataFrame,
    judge_results: List[Dict[str, Any]],
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute raw agreement rate between human and LLM judge.

    No Cohen's Kappa — single reviewer, explicitly stated.

    Returns:
        Dict with agreement rate, disagreement list, and disclosure.
    """
    judge_map = {}
    for jr in judge_results:
        judge_map[jr["conversation_id"]] = jr.get("mitigated_winner", "tie")

    agreements = 0
    disagreements = []
    total = 0

    for _, row in human_df.iterrows():
        pair_id = row["pair_id"]
        m = mapping.get(pair_id)
        if m is None:
            continue

        conv_id = m["conversation_id"]
        which_gen = m["which_is_generated"]

        human_pref = str(row.get("human_preference", "")).strip().upper()
        if human_pref == which_gen:
            human_source = "generated"
        elif human_pref == "TIE" or human_pref == "":
            human_source = "tie"
        else:
            human_source = "template"

        judge_source = judge_map.get(conv_id, "unknown")

        total += 1
        if human_source == judge_source:
            agreements += 1
        else:
            disagreements.append({
                "conversation_id": conv_id,
                "pair_id": pair_id,
                "human_preference": human_source,
                "judge_preference": judge_source,
                "human_rationale": row.get("human_rationale", ""),
            })

    return {
        "total_compared": total,
        "agreements": agreements,
        "disagreements_count": len(disagreements),
        "raw_agreement_rate": round(agreements / total, 4) if total > 0 else 0.0,
        "disagreements": disagreements,
        "single_reviewer_disclosure": (
            "All human annotations were performed by a single reviewer. "
            "No inter-annotator agreement metrics (Cohen's Kappa, etc.) are reported. "
            "This is an acknowledged limitation."
        ),
    }
