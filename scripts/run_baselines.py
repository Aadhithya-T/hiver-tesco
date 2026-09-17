#!/usr/bin/env python3
"""Run and evaluate leakage-safe intent, escalation, and template reply baselines on golden set splits."""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure src/ and root are on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from hiver_tesco.baselines.escalation_baselines import (
    AlwaysDoNotEscalateBaseline,
    MajorityClassEscalateBaseline,
    RiskKeywordEscalationBaseline,
)
from hiver_tesco.baselines.evaluation import (
    evaluate_escalation_predictions,
    evaluate_intent_predictions,
    extract_escalation_false_negatives,
)
from hiver_tesco.baselines.intent_baselines import (
    MajorityClassIntentBaseline,
    TfidfLogRegIntentBaseline,
)
from hiver_tesco.baselines.query_extractor import extract_leakage_safe_query
from hiver_tesco.baselines.reply_templates import IntentTemplateReplyBaseline


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s")


def load_dataset_with_queries(
    golden_csv_path: Path,
    conversations_jsonl_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load golden CSV and enrich with leakage-safe customer query_text from conversations.jsonl."""
    logger = logging.getLogger("run_baselines")

    # Load full conversation message threads to extract pre-resolution query_text
    logger.info(f"Loading full conversation message threads from {conversations_jsonl_path}...")
    full_convs = {}
    with open(conversations_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                full_convs[data["conversation_id"]] = data

    # Load golden records
    logger.info(f"Loading golden annotations from {golden_csv_path}...")
    golden_records = {}
    with open(golden_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cid = r["conversation_id"]
            conv_data = full_convs.get(cid, {})
            messages = conv_data.get("messages", [])

            # Strictly leakage-safe customer query text (prior to any Tesco response)
            query_text = extract_leakage_safe_query(messages)
            if not query_text:
                # Fallback to dialogue_text customer turn if messages missing
                query_text = r.get("dialogue_text", "").split("\n\n")[0]

            r_enriched = dict(r)
            r_enriched["query_text"] = query_text
            golden_records[cid] = r_enriched

    return golden_records


def evaluate_split(
    split_name: str,
    records: List[Dict[str, Any]],
    intent_majority_model: MajorityClassIntentBaseline,
    intent_tfidf_model: TfidfLogRegIntentBaseline,
    esc_always_no_model: AlwaysDoNotEscalateBaseline,
    esc_majority_model: MajorityClassEscalateBaseline,
    esc_risk_model: RiskKeywordEscalationBaseline,
    reply_model: IntentTemplateReplyBaseline,
    all_intent_labels: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate all baselines on a single split and generate predictions and metrics."""
    X_queries = [r["query_text"] for r in records]
    y_intent_true = [r["primary_intent"] for r in records]
    y_esc_true = [r["escalation_needed"] for r in records]

    # 1. Intent Predictions
    pred_intent_maj = intent_majority_model.predict(X_queries)
    pred_intent_tfidf = intent_tfidf_model.predict(X_queries)

    # 2. Escalation Predictions
    pred_esc_always_no = esc_always_no_model.predict(X_queries)
    pred_esc_maj = esc_majority_model.predict(X_queries)
    pred_esc_risk = esc_risk_model.predict(X_queries)

    # 3. Intent Evaluation Metrics
    metrics_intent_maj = evaluate_intent_predictions(y_intent_true, pred_intent_maj, labels=all_intent_labels)
    metrics_intent_tfidf = evaluate_intent_predictions(y_intent_true, pred_intent_tfidf, labels=all_intent_labels)

    # 4. Escalation Evaluation Metrics
    metrics_esc_always_no = evaluate_escalation_predictions(y_esc_true, pred_esc_always_no)
    metrics_esc_maj = evaluate_escalation_predictions(y_esc_true, pred_esc_maj)
    metrics_esc_risk = evaluate_escalation_predictions(y_esc_true, pred_esc_risk)

    # 5. False Negative Analysis for Risk Keyword Baseline
    false_negatives = extract_escalation_false_negatives(records, y_esc_true, pred_esc_risk)

    # 6. Assemble prediction records with template replies conditioned on predicted intent
    prediction_records = []
    for idx, r in enumerate(records):
        p_intent = pred_intent_tfidf[idx]
        p_esc = pred_esc_risk[idx]

        # Template reply conditioned STRICTLY on predicted intent and predicted policy
        template_reply = reply_model.generate_reply(p_intent, p_esc)

        # Oracle diagnostic reply (diagnostic only, separate from evaluation)
        oracle_info = reply_model.generate_oracle_reply(r["primary_intent"], r["escalation_needed"])

        prediction_records.append({
            "split": split_name,
            "conversation_id": r["conversation_id"],
            "query_text": r["query_text"],
            "ground_truth": {
                "primary_intent": r["primary_intent"],
                "escalation_needed": r["escalation_needed"],
                "escalation_rationale": r.get("escalation_rationale"),
            },
            "predictions": {
                "intent_majority": pred_intent_maj[idx],
                "intent_tfidf_logreg": p_intent,
                "escalation_always_do_not_escalate": pred_esc_always_no[idx],
                "escalation_majority_class": pred_esc_maj[idx],
                "escalation_risk_keyword": p_esc,
            },
            "generated_replies": {
                "baseline_template_reply": template_reply,
                "conditioning": {
                    "conditioned_on_predicted_intent": p_intent,
                    "conditioned_on_predicted_escalation": p_esc,
                },
                "oracle_diagnostic_only": oracle_info,
            },
        })

    split_metrics = {
        "split": split_name,
        "sample_size": len(records),
        "intent_baselines": {
            "majority_class": metrics_intent_maj,
            "tfidf_logistic_regression": metrics_intent_tfidf,
        },
        "escalation_baselines": {
            "always_do_not_escalate": metrics_esc_always_no,
            "majority_class_escalate": metrics_esc_maj,
            "risk_keyword_regex": metrics_esc_risk,
        },
    }

    return split_metrics, prediction_records, false_negatives


def main():
    setup_logging()
    logger = logging.getLogger("run_baselines")

    parser = argparse.ArgumentParser(description="Run simple intent, escalation, and reply baselines on golden set.")
    parser.add_argument(
        "--golden-csv",
        type=Path,
        default=Path("data/golden/golden_set_v1.0.csv"),
        help="Path to frozen golden set CSV (default: data/golden/golden_set_v1.0.csv)",
    )
    parser.add_argument(
        "--conversations-jsonl",
        type=Path,
        default=Path("outputs/conversations.jsonl"),
        help="Path to full conversations JSONL (default: outputs/conversations.jsonl)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/golden/splits"),
        help="Directory containing split ID files (default: data/golden/splits)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baselines"),
        help="Directory to save baseline outputs and metrics (default: outputs/baselines)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test", "all"],
        default="dev",
        help="Split to evaluate (default: dev). Use 'all' or 'test' once development choices are frozen.",
    )

    args = parser.parse_args()

    # Load enriched golden dataset
    records_by_id = load_dataset_with_queries(args.golden_csv, args.conversations_jsonl)

    # Load split IDs
    with open(args.splits_dir / "train_ids.json", "r", encoding="utf-8") as f:
        train_ids = json.load(f)
    with open(args.splits_dir / "dev_ids.json", "r", encoding="utf-8") as f:
        dev_ids = json.load(f)
    with open(args.splits_dir / "test_ids.json", "r", encoding="utf-8") as f:
        test_ids = json.load(f)

    train_records = [records_by_id[cid] for cid in train_ids if cid in records_by_id]
    dev_records = [records_by_id[cid] for cid in dev_ids if cid in records_by_id]
    test_records = [records_by_id[cid] for cid in test_ids if cid in records_by_id]

    logger.info(f"Loaded splits: Train={len(train_records)}, Dev={len(dev_records)}, Test={len(test_records)}")

    # Extract training data
    train_queries = [r["query_text"] for r in train_records]
    train_intents = [r["primary_intent"] for r in train_records]
    train_escalations = [r["escalation_needed"] for r in train_records]

    all_intents = sorted(list(set(r["primary_intent"] for r in records_by_id.values())))

    # 1. Fit baselines STRICTLY on Train split query_text
    logger.info("Fitting Intent Baselines on Train split query_text...")
    intent_maj = MajorityClassIntentBaseline().fit(train_queries, train_intents)
    intent_tfidf = TfidfLogRegIntentBaseline().fit(train_queries, train_intents)

    logger.info("Fitting Escalation Baselines on Train split...")
    esc_always_no = AlwaysDoNotEscalateBaseline().fit(train_queries, train_escalations)
    esc_maj = MajorityClassEscalateBaseline().fit(train_queries, train_escalations)
    esc_risk = RiskKeywordEscalationBaseline().fit(train_queries, train_escalations)

    reply_model = IntentTemplateReplyBaseline()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits_to_evaluate = []
    if args.split in ("dev", "all"):
        splits_to_evaluate.append(("dev", dev_records))
    if args.split in ("test", "all"):
        splits_to_evaluate.append(("test", test_records))

    all_metrics_summary = {}
    csv_rows = [
        ["split", "task", "model", "accuracy", "macro_f1", "precision", "recall", "f1_score", "fn_rate"]
    ]

    for split_name, split_records in splits_to_evaluate:
        logger.info(f"Evaluating baselines on {split_name.upper()} split (N={len(split_records)})...")
        metrics, preds, fn_analysis = evaluate_split(
            split_name=split_name,
            records=split_records,
            intent_majority_model=intent_maj,
            intent_tfidf_model=intent_tfidf,
            esc_always_no_model=esc_always_no,
            esc_majority_model=esc_maj,
            esc_risk_model=esc_risk,
            reply_model=reply_model,
            all_intent_labels=all_intents,
        )

        all_metrics_summary[split_name] = metrics

        # Save predictions JSONL
        preds_path = args.output_dir / f"predictions_{split_name}.jsonl"
        logger.info(f"Writing predictions to {preds_path}...")
        with open(preds_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        # Save false negative analysis
        if fn_analysis:
            fn_path = args.output_dir / f"escalation_false_negatives_{split_name}.json"
            with open(fn_path, "w", encoding="utf-8") as f:
                json.dump(fn_analysis, f, indent=2)

        # Collect summary rows
        im = metrics["intent_baselines"]["majority_class"]
        csv_rows.append([split_name, "intent", "majority_class", im["accuracy"], im["macro_f1"], "", "", "", ""])
        it = metrics["intent_baselines"]["tfidf_logistic_regression"]
        csv_rows.append([split_name, "intent", "tfidf_logreg", it["accuracy"], it["macro_f1"], "", "", "", ""])

        en = metrics["escalation_baselines"]["always_do_not_escalate"]
        csv_rows.append([split_name, "escalation", "always_do_not_escalate", en["accuracy"], "", en["precision"], en["recall"], en["f1_score"], en["false_negative_rate"]])
        em = metrics["escalation_baselines"]["majority_class_escalate"]
        csv_rows.append([split_name, "escalation", "majority_class_escalate", em["accuracy"], "", em["precision"], em["recall"], em["f1_score"], em["false_negative_rate"]])
        er = metrics["escalation_baselines"]["risk_keyword_regex"]
        csv_rows.append([split_name, "escalation", "risk_keyword_regex", er["accuracy"], "", er["precision"], er["recall"], er["f1_score"], er["false_negative_rate"]])

    # Save metrics JSON & CSV
    metrics_json_path = args.output_dir / "metrics_summary.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics_summary, f, indent=2)

    metrics_csv_path = args.output_dir / "metrics_summary.csv"
    with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    # Save confusion matrices
    cms = {}
    for s_name in all_metrics_summary:
        s_data = all_metrics_summary[s_name]
        cms[s_name] = {
            "intent_majority": s_data["intent_baselines"]["majority_class"]["confusion_matrix"],
            "intent_tfidf": s_data["intent_baselines"]["tfidf_logistic_regression"]["confusion_matrix"],
            "escalation_risk_regex": s_data["escalation_baselines"]["risk_keyword_regex"]["confusion_matrix"],
        }
    with open(args.output_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(cms, f, indent=2)

    # Save sample template replies
    sample_replies = {}
    for intent in all_intents:
        sample_replies[intent] = {
            "predicted_intent": intent,
            "reply_when_escalated": reply_model.generate_reply(intent, "escalate"),
            "reply_when_not_escalated": reply_model.generate_reply(intent, "do_not_escalate"),
        }
    with open(args.output_dir / "template_replies_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_replies, f, indent=2)

    # Print clean formatted summary report to stdout
    print("\n" + "=" * 80)
    print("PHASE 3 BASELINES EVALUATION REPORT")
    print("=" * 80)
    print(f"Golden Set File: {args.golden_csv}")
    print("Single Reviewer Disclosure: All 200 labels manually annotated by 1 reviewer (no agreement measured).")
    print("Leakage Protection: All models evaluated strictly on customer query_text (pre-Tesco resolution).")
    print("-" * 80)

    for split_name, split_data in all_metrics_summary.items():
        print(f"\n[{split_name.upper()} SPLIT - N={split_data['sample_size']}]")
        print("\n1. INTENT CLASSIFICATION BASELINES:")
        print(f"  {'Model':<25} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12}")
        print("  " + "-" * 55)
        for m_name in ("majority_class", "tfidf_logistic_regression"):
            m = split_data["intent_baselines"][m_name]
            print(f"  {m_name:<25} {m['accuracy']:<10.4f} {m['macro_f1']:<10.4f} {m['weighted_f1']:<12.4f}")

        print("\n2. ESCALATION DETECTION BASELINES (Positive = 'escalate'):")
        print(f"  {'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FN Rate':<10}")
        print("  " + "-" * 75)
        for m_name in ("always_do_not_escalate", "majority_class_escalate", "risk_keyword_regex"):
            m = split_data["escalation_baselines"][m_name]
            print(f"  {m_name:<25} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1_score']:<10.4f} {m['false_negative_rate']:<10.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
