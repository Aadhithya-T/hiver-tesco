"""Deep-dive analysis of human–judge disagreements.

When human and LLM judge disagree, human evaluation is foregrounded
as the primary ground truth.
"""

from typing import Any, Dict, List, Optional

import pandas as pd


FAILURE_MODES = {
    "generated_verbose": "Generated reply is unnecessarily verbose",
    "generated_near_copy": "Generated reply is a near-verbatim copy of evidence",
    "generated_tone_mismatch": "Generated reply has wrong tone for the query",
    "generated_hallucination": "Generated reply contains unsupported claims",
    "template_too_generic": "Template reply is too generic for the specific issue",
    "template_wrong_tone": "Template reply has inappropriate tone (e.g., apology for compliment)",
    "template_lacks_routing": "Template doesn't guide customer to the right next step",
    "judge_position_bias": "Judge flipped on position swap, suggesting position-1 bias",
    "judge_prefers_boilerplate": "Judge prefers safe boilerplate over contextual response",
    "judge_misread_sentiment": "Judge misread customer sentiment/intent",
    "ambiguous_both_acceptable": "Both replies are reasonable; disagreement is subjective",
    "unclassified": "Disagreement does not fit a clear pattern",
}


def classify_disagreements(
    human_df: pd.DataFrame,
    judge_results: List[Dict[str, Any]],
    mapping: Dict[str, Any],
    audit_records: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Classify all human–judge disagreements by failure mode.

    Returns a DataFrame with one row per disagreement.
    """
    # Build lookup maps
    judge_map = {}
    judge_rationale_map = {}
    judge_consistency_map = {}
    for jr in judge_results:
        cid = jr["conversation_id"]
        judge_map[cid] = jr.get("mitigated_winner", "tie")
        judge_rationale_map[cid] = jr.get("forward_judgment", {})
        judge_consistency_map[cid] = jr.get("is_consistent", True)

    audit_map = {}
    for rec in audit_records:
        audit_map[str(rec["conversation_id"])] = rec

    rows = []
    for _, row in human_df.iterrows():
        pair_id = row["pair_id"]
        m = mapping.get(pair_id)
        if m is None:
            continue

        conv_id = m["conversation_id"]
        which_gen = m["which_is_generated"]

        human_pref_raw = str(row.get("human_preference", "")).strip().upper()
        if human_pref_raw == which_gen:
            human_source = "generated"
        elif human_pref_raw in ("TIE", ""):
            human_source = "tie"
        else:
            human_source = "template"

        judge_source = judge_map.get(conv_id, "unknown")

        # Only process disagreements
        if human_source == judge_source:
            continue

        # Attempt to classify the failure mode
        rec = audit_map.get(conv_id, {})
        failure_mode = "unclassified"

        is_judge_inconsistent = not judge_consistency_map.get(conv_id, True)

        if is_judge_inconsistent:
            failure_mode = "judge_position_bias"
        elif human_source == "generated" and judge_source == "template":
            # Human prefers generated but judge prefers template
            draft = rec.get("final_draft", "")
            template = rec.get("template_baseline_reply", "")
            if len(draft.split()) > len(template.split()) * 1.5:
                failure_mode = "judge_prefers_boilerplate"
            else:
                failure_mode = "judge_misread_sentiment"
        elif human_source == "template" and judge_source == "generated":
            # Human prefers template but judge prefers generated
            draft = rec.get("final_draft", "")
            if draft and len(draft.split()) > 60:
                failure_mode = "generated_verbose"
            else:
                failure_mode = "template_too_generic"
        elif human_source == "tie" or judge_source == "tie" or judge_source == "tie_inconsistent":
            failure_mode = "ambiguous_both_acceptable"

        rows.append({
            "conversation_id": conv_id,
            "pair_id": pair_id,
            "split": m.get("split", "unknown"),
            "primary_intent": m.get("primary_intent", "unknown"),
            "human_preference": human_source,
            "judge_preference": judge_source,
            "human_rationale": row.get("human_rationale", ""),
            "judge_consistent": not is_judge_inconsistent,
            "failure_mode": failure_mode,
            "failure_description": FAILURE_MODES.get(failure_mode, "Unknown"),
        })

    return pd.DataFrame(rows)


def build_disagreement_summary(disagreements_df: pd.DataFrame) -> str:
    """Build a prioritized disagreement summary foregrounding human evaluation.

    Human preference is stated first as the primary ground truth.
    """
    if disagreements_df.empty:
        return (
            "## Disagreement Analysis\n\n"
            "No disagreements between human evaluator and LLM judge.\n\n"
            "> **Single-Reviewer Disclosure**: All human annotations were performed by a single reviewer. "
            "No inter-annotator agreement metrics are reported.\n"
        )

    total = len(disagreements_df)
    mode_counts = disagreements_df["failure_mode"].value_counts()
    top_mode = mode_counts.index[0] if len(mode_counts) > 0 else "unclassified"
    top_desc = FAILURE_MODES.get(top_mode, "Unknown")

    # Count by human preference direction
    human_gen = (disagreements_df["human_preference"] == "generated").sum()
    human_template = (disagreements_df["human_preference"] == "template").sum()
    human_tie = (disagreements_df["human_preference"] == "tie").sum()

    md = f"""## Disagreement Analysis

**Human preference is the primary ground truth.** The LLM judge disagreed on **{total}** case(s).

The most common disagreement pattern was: **{top_desc}** ({mode_counts.iloc[0]} case(s)).

### Disagreement Direction
| Human Preferred | Count |
|---|:---:|
| Generated | {human_gen} |
| Template | {human_template} |
| Tie | {human_tie} |

### Failure Mode Breakdown
| Failure Mode | Count | Description |
|---|:---:|---|
"""
    for mode, count in mode_counts.items():
        desc = FAILURE_MODES.get(mode, "Unknown")
        md += f"| `{mode}` | {count} | {desc} |\n"

    md += f"""
### Per-Case Disagreements

| Conv ID | Split | Intent | Human | Judge | Failure Mode | Human Rationale |
|---|---|---|---|---|---|---|
"""
    for _, row in disagreements_df.iterrows():
        md += (
            f"| `{row['conversation_id']}` | {row['split']} | {row['primary_intent']} | "
            f"{row['human_preference']} | {row['judge_preference']} | "
            f"`{row['failure_mode']}` | {row.get('human_rationale', '')[:80]} |\n"
        )

    md += """
> **Single-Reviewer Disclosure**: All human annotations were performed by a single reviewer. \
No inter-annotator agreement metrics (Cohen's Kappa, etc.) are reported. \
This is an acknowledged limitation.
"""
    return md
