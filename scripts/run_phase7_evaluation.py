"""Phase 7 — Reply evaluation, judge validation, and human comparison.

Three-stage CLI:
  --stage dev      : Finalize live prompt/model setup on Dev subset (iterative)
  --stage test     : One-shot locked Test run (final reported result)
  --stage finalize : Read saved artifacts + human labels, produce final report (no API calls)

Usage:
  python scripts/run_phase7_evaluation.py --stage dev --provider openai --model gpt-4o-mini
  python scripts/run_phase7_evaluation.py --stage test --provider openai --model gpt-4o-mini
  python scripts/run_phase7_evaluation.py --stage finalize
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np

from hiver_tesco.baselines.intent_baselines import TfidfLogRegIntentBaseline
from hiver_tesco.baselines.query_extractor import extract_leakage_safe_query
from hiver_tesco.evaluation.automated_checks import run_all_automated_checks
from hiver_tesco.evaluation.disagreement_analyzer import (
    build_disagreement_summary,
    classify_disagreements,
)
from hiver_tesco.evaluation.escalation_evaluation import (
    build_escalation_review_table,
    evaluate_escalation_routing,
)
from hiver_tesco.evaluation.human_comparison import (
    compute_judge_human_agreement,
    compute_preference_stats,
    generate_blinded_annotation_csv,
)
from hiver_tesco.evaluation.pairwise_judge import (
    BlindedPairwiseJudge,
    MockPairwiseJudge,
    OpenAIPairwiseJudge,
    compute_position_bias_metrics,
    summarize_judge_preferences,
)
from hiver_tesco.evaluation.sample_selection import (
    select_evaluation_subset,
    validate_evaluation_subset,
)
from hiver_tesco.generation.cache import ResponseCache
from hiver_tesco.generation.generator import GroundedReplyGenerator
from hiver_tesco.generation.models import EvidenceMatch, GenerationConfig, GenerationStatus
from hiver_tesco.generation.providers import get_provider
from hiver_tesco.policy.engine import DeterministicPolicyEngine
from hiver_tesco.policy.models import PolicyAction, PolicyContext
from hiver_tesco.retrieval.corpus import load_corpus_jsonl
from hiver_tesco.retrieval.evidence_filter import EvidenceFilter
from hiver_tesco.retrieval.retrievers import SemanticHistoricalRetriever


def count_pre_resolution_inbound_turns(messages) -> int:
    """Count customer inbound messages arriving before any Tesco outbound reply."""
    first_outbound_idx = None
    for idx, m in enumerate(messages):
        if m.get("direction") == "outbound":
            first_outbound_idx = idx
            break
    cutoff = first_outbound_idx if first_outbound_idx is not None else len(messages)
    return sum(1 for i in range(cutoff) if messages[i].get("direction") == "inbound")


def run_generation_for_conversations(
    conv_ids, conversations_map, golden_df, train_ids,
    provider_name, model_name, output_dir, retriever, evidence_filter, intent_model,
):
    """Run Phase 6 generation pipeline for a set of conversation IDs.

    Returns list of audit record dicts.
    """
    golden_df = golden_df.copy()
    golden_df["conversation_id"] = golden_df["conversation_id"].astype(str)

    config = GenerationConfig(
        provider=provider_name,
        model=model_name or ("mock-pipeline-v1" if provider_name == "mock" else "gpt-4o-mini"),
        temperature=0.0,
        max_tokens=150,
    )
    if provider_name == "ollama":
        config.base_url = "http://localhost:11434/v1/chat/completions"
        config.api_key_env_var = "OLLAMA_API_KEY"

    cache_dir = output_dir / "cache"
    audit_path = output_dir / "generation_audit.jsonl"

    generator = GroundedReplyGenerator(
        config=config,
        cache=ResponseCache(cache_dir=cache_dir),
        audit_log_path=audit_path,
    )

    policy_engine = DeterministicPolicyEngine(retrieval_threshold=0.55, confidence_threshold=0.18)
    audit_records = []

    for idx, conv_id in enumerate(conv_ids, 1):
        conv_id_str = str(conv_id)
        conv_data = conversations_map.get(conv_id_str)
        if conv_data is None:
            print(f"  [WARN] Conversation {conv_id_str} not found in conversations.jsonl")
            continue

        messages = conv_data.get("messages", [])
        query_text = extract_leakage_safe_query(messages)
        if not query_text:
            continue

        # Intent prediction & confidence
        probs = intent_model.predict_proba([query_text])[0]
        model_confidence = float(np.max(probs))
        predicted_intent = str(intent_model.classes_[np.argmax(probs)])

        # Retrieval with per-query historical cutoff & self-exclusion
        conv_started_at = conv_data.get("started_at")
        matches = retriever.retrieve(
            query_text=query_text,
            query_started_at=conv_started_at,
            query_conversation_id=conv_id_str,
            top_k=1,
        )
        top_match = matches[0] if matches else None
        filter_res = evidence_filter.evaluate(top_match) if top_match else None
        evidence_accepted = filter_res.is_accepted if filter_res else False

        evidence = EvidenceMatch.from_candidate_match(top_match) if top_match else None

        # Policy
        inbound_turns = count_pre_resolution_inbound_turns(messages)
        policy_ctx = PolicyContext(
            query_text=query_text,
            model_confidence=model_confidence,
            predicted_intent=predicted_intent,
            retrieval_score=top_match.score if top_match else 0.0,
            retrieval_accepted=evidence_accepted,
            inbound_turn_count=inbound_turns,
        )
        policy_decision = policy_engine.evaluate(policy_ctx)

        # Generation
        audit = generator.generate(
            conversation_id=conv_id_str,
            query_text=query_text,
            policy_decision=policy_decision,
            predicted_intent=predicted_intent,
            evidence=evidence,
            evidence_accepted=evidence_accepted,
        )

        policy_label = "escalate" if policy_decision.action == PolicyAction.ESCALATE else "respond"
        status = audit.generation_status
        print(f"  [{idx:02d}/{len(conv_ids):02d}] Conv {conv_id_str:<10} | Policy: {policy_label:<8} | Status: [{status}]")

        audit_records.append(audit.to_dict())

    return audit_records


def run_pairwise_judging(audit_records, judge, split_label):
    """Run blinded pairwise judging on RESPOND-only cases."""
    respond_records = [r for r in audit_records if r.get("policy_action") == "respond" and r.get("final_draft")]

    if not respond_records:
        print(f"  No RESPOND cases with drafts in {split_label} split — skipping pairwise judging.")
        return [], {}

    blinded = BlindedPairwiseJudge(judge)
    results = []

    for idx, rec in enumerate(respond_records, 1):
        conv_id = str(rec["conversation_id"])
        query = ""
        prompt = rec.get("sanitized_prompt", "")
        if "CUSTOMER INQUIRY (Sanitized):" in prompt:
            parts = prompt.split("CUSTOMER INQUIRY (Sanitized):")
            if len(parts) > 1:
                query = parts[1].split("POLICY GUIDANCE:")[0].strip().strip('"')

        generated = rec["final_draft"]
        template = rec.get("template_baseline_reply", "")

        print(f"  [{idx:02d}/{len(respond_records):02d}] Judging Conv {conv_id}...")
        result = blinded.evaluate_pair(conv_id, query, generated, template)
        results.append(result)

    bias_metrics = compute_position_bias_metrics(results)
    pref_summary = summarize_judge_preferences(results)

    print(f"\n  Pairwise Judge Summary ({split_label}):")
    print(f"    Total pairs judged: {pref_summary['total']}")
    print(f"    Generated wins: {pref_summary['generated_wins']}")
    print(f"    Template wins: {pref_summary['template_wins']}")
    print(f"    Ties: {pref_summary['ties']} + {pref_summary['inconsistent_ties']} inconsistent")
    print(f"    Position-A win rate: {bias_metrics['position_a_win_rate']:.2%}")
    print(f"    Swap consistency: {bias_metrics['swap_consistency_rate']:.2%}")
    print(f"    Flip rate: {bias_metrics['flip_rate']:.2%}")

    return results, {"bias_metrics": bias_metrics, "preference_summary": pref_summary}


def stage_dev(args, repo_root):
    """Stage 1: Dev validation — finalize live prompt/model setup."""
    print("=" * 70)
    print("PHASE 7 — STAGE: DEV VALIDATION")
    print("=" * 70)
    print(f"Provider: {args.provider}, Model: {args.model or 'gpt-4o-mini'}")
    print()

    output_dir = repo_root / "outputs" / "evaluation" / "dev"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    golden_df = pd.read_csv(repo_root / "data" / "golden" / "golden_set_v1.0.csv")
    with open(repo_root / "data" / "golden" / "splits" / "dev_ids.json") as f:
        dev_ids = json.load(f)
    with open(repo_root / "data" / "golden" / "splits" / "test_ids.json") as f:
        test_ids = json.load(f)
    with open(repo_root / "data" / "golden" / "splits" / "train_ids.json") as f:
        train_ids = json.load(f)

    # Load conversations
    conversations_map = {}
    conv_path = repo_root / "outputs" / "conversations.jsonl"
    with open(conv_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            conversations_map[str(c["conversation_id"])] = c

    # Select evaluation subset
    full_60, dev_20, test_40 = select_evaluation_subset(golden_df, dev_ids, test_ids)
    full_60.to_csv(repo_root / "outputs" / "evaluation" / "evaluation_subset_60.csv", index=False)
    validation = validate_evaluation_subset(full_60)
    print(f"Evaluation subset: {validation['total_conversations']} conversations")
    print(f"  Dev subset: {validation['dev_count']}, Test: {validation['test_count']}")
    print(f"  All 8 intents covered: {validation['all_8_intents_covered']}")
    print()

    dev_conv_ids = dev_20["conversation_id"].tolist()
    print(f"Running generation on {len(dev_conv_ids)} Dev conversations...")

    # Load retriever
    print("Loading retrieval engine...")
    corpus = load_corpus_jsonl(repo_root / "outputs" / "retrieval" / "historical_corpus.jsonl")
    retriever = SemanticHistoricalRetriever(model_name_or_path="all-MiniLM-L6-v2", batch_size=128)
    retriever.index(corpus, embeddings_cache_path=repo_root / "outputs" / "retrieval" / "corpus_embeddings.npy")
    evidence_filter = EvidenceFilter(semantic_threshold=0.55)

    # Load intent model
    golden_map = {str(r["conversation_id"]): r for _, r in golden_df.iterrows()}
    X_train = [extract_leakage_safe_query(conversations_map[str(cid)]["messages"]) for cid in train_ids if str(cid) in conversations_map]
    y_train = [golden_map[str(cid)]["primary_intent"] for cid in train_ids if str(cid) in conversations_map]
    intent_model = TfidfLogRegIntentBaseline()
    intent_model.fit(X_train, y_train)

    audit_records = run_generation_for_conversations(
        dev_conv_ids, conversations_map, golden_df, train_ids,
        args.provider, args.model, output_dir, retriever, evidence_filter, intent_model,
    )

    # Save audit records
    audit_path = output_dir / "generation_audit.jsonl"
    with open(audit_path, "w", encoding="utf-8") as f:
        for rec in audit_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Automated checks
    checks = run_all_automated_checks(audit_records)
    pd.DataFrame(checks).to_csv(output_dir / "automated_checks.csv", index=False)
    print(f"\nAutomated checks exported: {output_dir / 'automated_checks.csv'}")

    # Pairwise judging
    respond_count = sum(1 for r in audit_records if r.get("policy_action") == "respond" and r.get("final_draft"))
    print(f"\nRespond cases with drafts (pairwise-comparable): {respond_count}/{len(audit_records)}")

    if args.provider == "mock":
        judge = MockPairwiseJudge()
    else:
        judge = OpenAIPairwiseJudge(
            model=args.model or "gpt-4o-mini",
            base_url=("http://localhost:11434/v1/chat/completions" if args.provider == "ollama"
                       else "https://api.openai.com/v1/chat/completions"),
        )

    results, judge_stats = run_pairwise_judging(audit_records, judge, "dev")

    # Save judge results
    judge_rows = [r.to_dict() for r in results]
    with open(output_dir / "pairwise_judge.json", "w", encoding="utf-8") as f:
        json.dump(judge_rows, f, indent=2, default=str)
    print(f"Judge results exported: {output_dir / 'pairwise_judge.json'}")

    print("\n" + "=" * 70)
    print("DEV STAGE COMPLETE")
    print("Review Dev results before proceeding to --stage test.")
    print("=" * 70)


def stage_test(args, repo_root):
    """Stage 2: Test final run — one-shot, locked."""
    print("=" * 70)
    print("PHASE 7 — STAGE: TEST (ONE-SHOT, LOCKED)")
    print("=" * 70)
    print(f"Provider: {args.provider}, Model: {args.model or 'gpt-4o-mini'}")
    print()

    output_dir = repo_root / "outputs" / "evaluation" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    golden_df = pd.read_csv(repo_root / "data" / "golden" / "golden_set_v1.0.csv")
    with open(repo_root / "data" / "golden" / "splits" / "test_ids.json") as f:
        test_ids = json.load(f)
    with open(repo_root / "data" / "golden" / "splits" / "train_ids.json") as f:
        train_ids = json.load(f)
    with open(repo_root / "data" / "golden" / "splits" / "dev_ids.json") as f:
        dev_ids = json.load(f)

    conversations_map = {}
    with open(repo_root / "outputs" / "conversations.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            conversations_map[str(c["conversation_id"])] = c

    # Load full evaluation subset for blinded CSV
    full_60, dev_20, test_40 = select_evaluation_subset(golden_df, dev_ids, test_ids)
    test_conv_ids = test_40["conversation_id"].tolist()

    print(f"Running generation on {len(test_conv_ids)} Test conversations (one-shot)...")

    # Load retriever
    print("Loading retrieval engine...")
    corpus = load_corpus_jsonl(repo_root / "outputs" / "retrieval" / "historical_corpus.jsonl")
    retriever = SemanticHistoricalRetriever(model_name_or_path="all-MiniLM-L6-v2", batch_size=128)
    retriever.index(corpus, embeddings_cache_path=repo_root / "outputs" / "retrieval" / "corpus_embeddings.npy")
    evidence_filter = EvidenceFilter(semantic_threshold=0.55)

    golden_map = {str(r["conversation_id"]): r for _, r in golden_df.iterrows()}
    X_train = [extract_leakage_safe_query(conversations_map[str(cid)]["messages"]) for cid in train_ids if str(cid) in conversations_map]
    y_train = [golden_map[str(cid)]["primary_intent"] for cid in train_ids if str(cid) in conversations_map]
    intent_model = TfidfLogRegIntentBaseline()
    intent_model.fit(X_train, y_train)

    audit_records = run_generation_for_conversations(
        test_conv_ids, conversations_map, golden_df, train_ids,
        args.provider, args.model, output_dir, retriever, evidence_filter, intent_model,
    )

    # Save audit records
    with open(output_dir / "generation_audit.jsonl", "w", encoding="utf-8") as f:
        for rec in audit_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Automated checks
    checks = run_all_automated_checks(audit_records)
    pd.DataFrame(checks).to_csv(output_dir / "automated_checks.csv", index=False)

    # Escalation routing evaluation (separate from pairwise)
    esc_metrics = evaluate_escalation_routing(audit_records, golden_df)
    esc_table = build_escalation_review_table(audit_records, golden_df)
    esc_table.to_csv(output_dir / "escalation_routing.csv", index=False)
    with open(output_dir / "escalation_metrics.json", "w") as f:
        json.dump(esc_metrics, f, indent=2)

    print(f"\nEscalation routing — Accuracy: {esc_metrics['accuracy']:.2%}, "
          f"Precision: {esc_metrics['precision']:.2%}, Recall: {esc_metrics['recall']:.2%}")

    # Pairwise judging
    respond_count = sum(1 for r in audit_records if r.get("policy_action") == "respond" and r.get("final_draft"))
    print(f"\nRespond cases with drafts (pairwise-comparable): {respond_count}/{len(audit_records)}")

    if args.provider == "mock":
        judge = MockPairwiseJudge()
    else:
        judge = OpenAIPairwiseJudge(
            model=args.model or "gpt-4o-mini",
            base_url=("http://localhost:11434/v1/chat/completions" if args.provider == "ollama"
                       else "https://api.openai.com/v1/chat/completions"),
        )

    results, judge_stats = run_pairwise_judging(audit_records, judge, "test")

    judge_rows = [r.to_dict() for r in results]
    with open(output_dir / "pairwise_judge.json", "w", encoding="utf-8") as f:
        json.dump(judge_rows, f, indent=2, default=str)

    # Generate blinded human annotation CSV (RESPOND-only from full 60)
    # Combine dev + test audit records if dev exists
    all_audit = list(audit_records)
    dev_audit_path = repo_root / "outputs" / "evaluation" / "dev" / "generation_audit.jsonl"
    if dev_audit_path.exists():
        with open(dev_audit_path, encoding="utf-8") as f:
            for line in f:
                all_audit.append(json.loads(line))

    blinded_df, mapping = generate_blinded_annotation_csv(full_60, all_audit)
    annotation_dir = repo_root / "data" / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    blinded_df.to_csv(annotation_dir / "human_pairwise_comparison.csv", index=False)
    with open(output_dir / "human_comparison_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nBlinded human annotation CSV exported: {annotation_dir / 'human_pairwise_comparison.csv'}")
    print(f"  Total pairwise pairs: {len(blinded_df)} (RESPOND-only from {len(full_60)} eval conversations)")
    print(f"Internal mapping exported: {output_dir / 'human_comparison_mapping.json'}")

    print("\n" + "=" * 70)
    print("TEST STAGE COMPLETE")
    print("Now complete human annotation in data/annotation/human_pairwise_comparison.csv")
    print("Then run: python scripts/run_phase7_evaluation.py --stage finalize")
    print("=" * 70)


def stage_finalize(args, repo_root):
    """Stage 3: Read saved artifacts + human labels, produce final report. No API calls."""
    print("=" * 70)
    print("PHASE 7 — STAGE: FINALIZE (OFFLINE REPORT ASSEMBLY)")
    print("No LLM or judge API calls will be made.")
    print("=" * 70)
    print()

    eval_dir = repo_root / "outputs" / "evaluation"
    test_dir = eval_dir / "test"
    dev_dir = eval_dir / "dev"

    # Load mapping
    mapping_path = test_dir / "human_comparison_mapping.json"
    if not mapping_path.exists():
        print("ERROR: Run --stage test first to generate the mapping file.")
        sys.exit(1)

    with open(mapping_path) as f:
        mapping = json.load(f)

    # Load human annotations
    human_path = repo_root / "data" / "annotation" / "human_pairwise_comparison.csv"
    if not human_path.exists():
        print(f"ERROR: Human annotation file not found at {human_path}")
        sys.exit(1)

    human_df = pd.read_csv(human_path)
    annotated_count = human_df["human_preference"].notna().sum()
    total_pairs = len(human_df)
    print(f"Loaded {annotated_count}/{total_pairs} annotated human comparisons.")

    if annotated_count == 0:
        print("WARNING: No human annotations found. Complete the annotation file first.")
        print("Generating report with judge-only results...")

    # Load judge results
    judge_results = []
    for split_dir in [dev_dir, test_dir]:
        judge_path = split_dir / "pairwise_judge.json"
        if judge_path.exists():
            with open(judge_path) as f:
                judge_results.extend(json.load(f))

    # Load audit records for both splits
    all_audit = []
    for split_dir in [dev_dir, test_dir]:
        audit_path = split_dir / "generation_audit.jsonl"
        if audit_path.exists():
            with open(audit_path, encoding="utf-8") as f:
                for line in f:
                    all_audit.append(json.loads(line))

    # Compute preference stats (Dev and Test SEPARATELY)
    pref_stats = compute_preference_stats(human_df, mapping)

    # Compute human–judge agreement (raw rate only, no Kappa)
    agreement = compute_judge_human_agreement(human_df, judge_results, mapping)

    # Disagreement analysis
    disagreements = classify_disagreements(human_df, judge_results, mapping, all_audit)
    disagreement_summary = build_disagreement_summary(disagreements)

    # Load automated checks
    auto_checks = {}
    for split_name, split_dir in [("dev", dev_dir), ("test", test_dir)]:
        check_path = split_dir / "automated_checks.csv"
        if check_path.exists():
            auto_checks[split_name] = pd.read_csv(check_path)

    # Load escalation metrics
    esc_metrics = {}
    esc_path = test_dir / "escalation_metrics.json"
    if esc_path.exists():
        with open(esc_path) as f:
            esc_metrics = json.load(f)

    # Build final report
    report = _build_final_report(pref_stats, agreement, disagreement_summary,
                                  judge_results, auto_checks, esc_metrics, mapping)

    report_path = eval_dir / "phase7_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Save computed stats
    with open(eval_dir / "preference_stats.json", "w") as f:
        json.dump(pref_stats, f, indent=2)
    with open(eval_dir / "agreement_stats.json", "w") as f:
        json.dump({k: v for k, v in agreement.items() if k != "disagreements"}, f, indent=2)
    if not disagreements.empty:
        disagreements.to_csv(eval_dir / "disagreements.csv", index=False)
    human_df.to_csv(eval_dir / "human_comparison_results.csv", index=False)

    print(f"\nFinal evaluation report: {report_path}")
    print(f"Preference stats: {eval_dir / 'preference_stats.json'}")
    print(f"Agreement stats: {eval_dir / 'agreement_stats.json'}")

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE — FINAL REPORT PUBLISHED")
    print("=" * 70)


def _build_final_report(pref_stats, agreement, disagreement_summary,
                         judge_results, auto_checks, esc_metrics, mapping):
    """Build the comprehensive Phase 7 evaluation report."""
    md = """# Phase 7 — Reply Evaluation Report

> **Single-Reviewer Disclosure**: All human annotations were performed by a single reviewer.
> No inter-annotator agreement metrics are reported. This is an acknowledged limitation.

---

## 1. Human Preference Results (Primary Ground Truth)

Dev and Test results are reported separately.

"""
    for split in ("test", "dev"):
        s = pref_stats.get(split, {})
        if not s or s.get("total_pairs", 0) == 0:
            md += f"### {split.title()} Split\nNo pairwise pairs available.\n\n"
            continue

        md += f"""### {split.title()} Split ({s['total_pairs']} pairwise pairs)

| Metric | Count | Rate | 95% Wilson CI |
|---|:---:|:---:|:---:|
| Generated wins | {s['generated_wins']} | {s['generated_win_rate']:.1%} | [{s['generated_win_ci_95'][0]:.1%}, {s['generated_win_ci_95'][1]:.1%}] |
| Template wins | {s['template_wins']} | {s['template_win_rate']:.1%} | [{s['template_win_ci_95'][0]:.1%}, {s['template_win_ci_95'][1]:.1%}] |
| Ties | {s['ties']} | {s['tie_rate']:.1%} | [{s['tie_ci_95'][0]:.1%}, {s['tie_ci_95'][1]:.1%}] |

"""
        by_intent = s.get("by_intent", {})
        if by_intent:
            md += "#### By Intent\n\n| Intent | Pairs | Gen Wins | Template Wins | Ties |\n|---|:---:|:---:|:---:|:---:|\n"
            for intent, counts in sorted(by_intent.items()):
                md += f"| {intent} | {counts['pairs']} | {counts['generated_wins']} | {counts['template_wins']} | {counts['ties']} |\n"
            md += "\n"

    md += """---

## 2. LLM Judge Preference Results

"""
    # Aggregate judge results by split
    for split in ("test", "dev"):
        split_judge = [jr for jr in judge_results if mapping.get(f"pair_{jr.get('conversation_id', '')}", {}).get("split") == split]
        gen_w = sum(1 for j in split_judge if j.get("mitigated_winner") == "generated")
        tmpl_w = sum(1 for j in split_judge if j.get("mitigated_winner") == "template")
        ties = sum(1 for j in split_judge if j.get("mitigated_winner") in ("tie", "tie_inconsistent"))

        if split_judge:
            md += f"### {split.title()} Split ({len(split_judge)} pairs)\n"
            md += f"- Generated wins: {gen_w}\n- Template wins: {tmpl_w}\n- Ties/Inconsistent: {ties}\n\n"

    md += f"""---

## 3. Human–Judge Agreement

- **Total compared**: {agreement.get('total_compared', 0)}
- **Raw agreement rate**: {agreement.get('raw_agreement_rate', 0):.1%}
- **Disagreements**: {agreement.get('disagreements_count', 0)}

> No Cohen's Kappa or inter-rater reliability metrics are reported (single reviewer).

---

## 4. Escalation Routing Evaluation (Separate from Pairwise)

Escalated conversations are evaluated as routing/safety decisions, not for reply quality.

"""
    if esc_metrics:
        md += f"""| Metric | Value |
|---|:---:|
| Accuracy | {esc_metrics.get('accuracy', 0):.1%} |
| Precision | {esc_metrics.get('precision', 0):.1%} |
| Recall | {esc_metrics.get('recall', 0):.1%} |
| F1 | {esc_metrics.get('f1', 0):.1%} |
| True Positives | {esc_metrics.get('true_positives', 0)} |
| False Positives | {esc_metrics.get('false_positives', 0)} |
| False Negatives | {esc_metrics.get('false_negatives', 0)} |
| True Negatives | {esc_metrics.get('true_negatives', 0)} |

"""

    md += "---\n\n## 5. Automated Quality Checks\n\n"
    for split, checks_df in auto_checks.items():
        unsafe_total = int(checks_df["unsafe_claim_count"].sum())
        near_copy = int(checks_df["near_copy_flag"].sum())
        policy_violations = int((~checks_df["policy_compliant"]).sum())
        md += f"""### {split.title()} Split
- Policy violations: {policy_violations}
- Unsafe claims flagged: {unsafe_total}
- Near-copy flags (Jaccard > 0.85): {near_copy}

"""

    md += f"""---

## 6. Disagreement Deep-Dive

{disagreement_summary}
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Reply Evaluation")
    parser.add_argument("--stage", required=True, choices=["dev", "test", "finalize"],
                        help="dev: validate on Dev subset; test: one-shot locked Test; finalize: offline report")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai", "ollama"],
                        help="LLM provider (mock for offline tests, openai or ollama for live)")
    parser.add_argument("--model", default=None, help="Model identifier (e.g., gpt-4o-mini, llama3.1:8b)")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    if args.stage == "dev":
        stage_dev(args, repo_root)
    elif args.stage == "test":
        stage_test(args, repo_root)
    elif args.stage == "finalize":
        stage_finalize(args, repo_root)


if __name__ == "__main__":
    main()
