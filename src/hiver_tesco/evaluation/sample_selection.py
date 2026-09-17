"""Deterministic stratified sampling for the Phase 7 evaluation set.

Selects exactly 60 conversations: all 40 Test + 20 stratified Dev.
Fully reproducible with fixed seed=42.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


SEED = 42


def _stratified_dev_selection(
    dev_df: pd.DataFrame,
    n_select: int = 20,
    seed: int = SEED,
    phase6_sentiment_conflict_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Select n_select Dev conversations using deterministic stratified sampling.

    Strategy:
    1. Allocate slots proportional to intent distribution, minimum 1 per intent.
    2. Within each intent, prefer escalation balance.
    3. Prioritize difficulty: long conversations (>=5 msgs), ambiguity_notes present,
       sentiment conflict fallbacks from Phase 6.
    4. Tie-break by conversation_id ascending (deterministic).
    """
    if len(dev_df) <= n_select:
        return dev_df.copy()

    sentiment_ids = set(phase6_sentiment_conflict_ids or [])
    dev_df = dev_df.copy()
    dev_df["conversation_id_str"] = dev_df["conversation_id"].astype(str)

    # Score each conversation for difficulty enrichment
    dev_df["difficulty_score"] = 0
    dev_df.loc[dev_df["message_count"] >= 5, "difficulty_score"] += 1
    dev_df.loc[dev_df["ambiguity_notes"].notna() & (dev_df["ambiguity_notes"] != ""), "difficulty_score"] += 1
    dev_df.loc[dev_df["conversation_id_str"].isin(sentiment_ids), "difficulty_score"] += 1

    # Sort within each intent by: difficulty_score desc, then conversation_id asc
    dev_df = dev_df.sort_values(
        ["primary_intent", "difficulty_score", "conversation_id"],
        ascending=[True, False, True],
    )

    # Proportional allocation per intent
    intent_counts = dev_df["primary_intent"].value_counts()
    intents = sorted(intent_counts.index.tolist())

    # Minimum 1 per intent, then proportional for remaining
    allocation: Dict[str, int] = {intent: 1 for intent in intents}
    remaining = n_select - len(intents)

    if remaining > 0:
        total_for_prop = intent_counts.sum() - len(intents)  # subtract the 1 already allocated
        if total_for_prop > 0:
            for intent in intents:
                extra = int(round((intent_counts[intent] - 1) / total_for_prop * remaining))
                allocation[intent] += extra

        # Adjust rounding errors
        total_allocated = sum(allocation.values())
        while total_allocated < n_select:
            # Give extra slots to the most underrepresented intent
            for intent in intents:
                if total_allocated >= n_select:
                    break
                if allocation[intent] < intent_counts[intent]:
                    allocation[intent] += 1
                    total_allocated += 1
        while total_allocated > n_select:
            # Remove from the least important (smallest intent)
            for intent in reversed(intents):
                if total_allocated <= n_select:
                    break
                if allocation[intent] > 1:
                    allocation[intent] -= 1
                    total_allocated -= 1

    # Select from each intent stratum with escalation balance
    selected_rows = []
    for intent in intents:
        n_to_pick = min(allocation.get(intent, 0), intent_counts.get(intent, 0))
        intent_df = dev_df[dev_df["primary_intent"] == intent].copy()

        # Try to pick a balanced mix of escalation labels
        escalate_df = intent_df[intent_df["escalation_needed"] == "escalate"]
        no_escalate_df = intent_df[intent_df["escalation_needed"] == "do_not_escalate"]

        if len(escalate_df) > 0 and len(no_escalate_df) > 0 and n_to_pick >= 2:
            n_esc = min(n_to_pick // 2, len(escalate_df))
            n_no_esc = min(n_to_pick - n_esc, len(no_escalate_df))
            # If we still have room, top up from whichever has more
            leftover = n_to_pick - n_esc - n_no_esc
            if leftover > 0:
                if len(escalate_df) > n_esc:
                    n_esc += min(leftover, len(escalate_df) - n_esc)
                    leftover = n_to_pick - n_esc - n_no_esc
                if leftover > 0 and len(no_escalate_df) > n_no_esc:
                    n_no_esc += min(leftover, len(no_escalate_df) - n_no_esc)

            selected_rows.extend(escalate_df.head(n_esc).index.tolist())
            selected_rows.extend(no_escalate_df.head(n_no_esc).index.tolist())
        else:
            selected_rows.extend(intent_df.head(n_to_pick).index.tolist())

    result = dev_df.loc[selected_rows].copy()

    # If still short due to rounding, pick remaining by difficulty
    if len(result) < n_select:
        remaining_df = dev_df[~dev_df.index.isin(result.index)]
        remaining_df = remaining_df.sort_values(
            ["difficulty_score", "conversation_id"], ascending=[False, True]
        )
        need = n_select - len(result)
        result = pd.concat([result, remaining_df.head(need)])

    result = result.head(n_select)
    result = result.drop(columns=["difficulty_score", "conversation_id_str"], errors="ignore")
    return result


def select_evaluation_subset(
    golden_df: pd.DataFrame,
    dev_ids: List[Any],
    test_ids: List[Any],
    phase6_sentiment_conflict_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select the 60-conversation evaluation subset.

    Returns:
        (full_60_df, dev_20_df, test_40_df)
        Each has a 'split' column ('dev' or 'test').
    """
    golden_df = golden_df.copy()
    golden_df["conversation_id"] = golden_df["conversation_id"].astype(str)
    dev_id_set = set(str(x) for x in dev_ids)
    test_id_set = set(str(x) for x in test_ids)

    # All 40 Test conversations
    test_df = golden_df[golden_df["conversation_id"].isin(test_id_set)].copy()
    test_df["split"] = "test"

    # Stratified 20 from Dev
    dev_full = golden_df[golden_df["conversation_id"].isin(dev_id_set)].copy()
    dev_selected = _stratified_dev_selection(
        dev_full, n_select=20, phase6_sentiment_conflict_ids=phase6_sentiment_conflict_ids
    )
    dev_selected["split"] = "dev"

    full_60 = pd.concat([test_df, dev_selected], ignore_index=True)

    return full_60, dev_selected, test_df


def validate_evaluation_subset(full_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate the evaluation subset meets coverage requirements."""
    intents = full_df["primary_intent"].unique().tolist()
    escalation_labels = full_df["escalation_needed"].unique().tolist()

    validation = {
        "total_conversations": len(full_df),
        "dev_count": int((full_df["split"] == "dev").sum()),
        "test_count": int((full_df["split"] == "test").sum()),
        "intent_categories_covered": len(intents),
        "all_8_intents_covered": len(intents) >= 8,
        "intents": sorted(intents),
        "escalation_labels_present": sorted(escalation_labels),
        "both_escalation_labels": "escalate" in escalation_labels and "do_not_escalate" in escalation_labels,
        "long_conversations_gte5": int((full_df["message_count"] >= 5).sum()),
    }
    return validation
