"""Script to generate Development-split retrieval review sample for human relevance annotation.

Selects representative Development queries, retrieves top-3 candidates using BM25
and SentenceTransformers (with strict per-query historical cutoff and self-exclusion),
evaluates deterministic evidence filter rules, and produces both an annotation CSV
template and a markdown review booklet.
"""

import json
from pathlib import Path
import sys
import time

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from hiver_tesco.baselines.query_extractor import extract_leakage_safe_query
from hiver_tesco.retrieval.corpus import load_corpus_jsonl
from hiver_tesco.retrieval.evidence_filter import EvidenceFilter
from hiver_tesco.retrieval.retrievers import BM25HistoricalRetriever, SemanticHistoricalRetriever


def main():
    repo_root = Path(__file__).resolve().parent.parent
    conversations_path = repo_root / "outputs" / "conversations.jsonl"
    corpus_path = repo_root / "outputs" / "retrieval" / "historical_corpus.jsonl"
    embeddings_cache_path = repo_root / "outputs" / "retrieval" / "corpus_embeddings.npy"
    dev_ids_path = repo_root / "data" / "golden" / "splits" / "dev_ids.json"
    golden_csv_path = repo_root / "data" / "golden" / "golden_set_v1.0.csv"

    output_csv_template = repo_root / "data" / "annotation" / "retrieval_relevance_dev_sample.csv"
    output_md_booklet = repo_root / "outputs" / "retrieval" / "retrieval_review_sample.md"

    print("=" * 60)
    print("PHASE 4: DEVELOPMENT RETRIEVAL REVIEW SAMPLE GENERATOR")
    print("=" * 60)

    # 1. Load Dev IDs and Golden Metadata
    with open(dev_ids_path, "r", encoding="utf-8") as f:
        dev_ids = set(str(x) for x in json.load(f))
    print(f"Loaded {len(dev_ids)} Development query IDs.")

    golden_df = pd.read_csv(golden_csv_path)
    golden_intent_map = dict(zip(golden_df["conversation_id"].astype(str), golden_df["primary_intent"]))

    # 2. Load Full Dev Conversations for Queries
    dev_convs = []
    with open(conversations_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cid = str(c["conversation_id"])
            if cid in dev_ids:
                q_text = extract_leakage_safe_query(c.get("messages", []))
                dev_convs.append({
                    "conversation_id": cid,
                    "started_at": c.get("started_at", ""),
                    "query_text": q_text,
                    "primary_intent": golden_intent_map.get(cid, "unknown"),
                })

    print(f"Loaded {len(dev_convs)} Development query conversations.")

    # 3. Stratified Selection of 16 Representative Dev Queries across intents
    df_dev = pd.DataFrame(dev_convs)
    sample_queries = []
    intents = df_dev["primary_intent"].unique()

    # Pick up to 2 representative queries per intent
    for intent in sorted(intents):
        sub = df_dev[df_dev["primary_intent"] == intent].sort_values("conversation_id")
        picked = sub.head(2)
        sample_queries.extend(picked.to_dict("records"))

    print(f"Selected {len(sample_queries)} stratified development review queries.")

    # 4. Load Historical Resolution Corpus and Initialize Retrievers
    print("Loading historical corpus and initializing BM25 retriever...")
    corpus = load_corpus_jsonl(corpus_path)
    bm25 = BM25HistoricalRetriever()
    bm25.index(corpus)
    print(f"Indexed {len(corpus):,} documents in BM25.")

    print("Initializing Semantic dense retriever...")
    semantic = SemanticHistoricalRetriever(model_name_or_path="all-MiniLM-L6-v2", batch_size=128)
    semantic.index(corpus, embeddings_cache_path=embeddings_cache_path)
    print("Semantic retriever ready.")

    evidence_filter = EvidenceFilter(semantic_threshold=0.45)

    # 5. Run Retrieval for Each Sample Query
    rows = []
    md_sections = []
    sample_id = 1

    md_sections.append("# Historical Resolution Retrieval: Development Review Sample\n")
    md_sections.append(
        "> **Methodology Note**:\n"
        "> - Retrieval is constrained by per-query historical cutoff (`candidate.ended_at <= query.started_at`) "
        "> and self-exclusion (`candidate.conversation_id != query.conversation_id`).\n"
        "> - Ranking is performed strictly on `sanitized_customer_issue`.\n"
        "> - Retrieved evidence text is `sanitized_tesco_resolution`.\n"
        "> - Intent labels are strictly context; relevance must be independently judged.\n\n"
    )

    for q_idx, q in enumerate(sample_queries, start=1):
        qid = q["conversation_id"]
        q_time = q["started_at"]
        q_text = q["query_text"]
        q_intent = q["primary_intent"]

        md_sections.append(f"## Query {q_idx}: ID {qid} ({q_intent})\n")
        md_sections.append(f"**Timestamp**: `{q_time}`  \n")
        md_sections.append(f"**Customer Inquiry**: > {q_text}\n\n")

        # Lexical Retrieval
        bm25_matches = bm25.retrieve(
            query_text=q_text,
            query_started_at=q_time,
            query_conversation_id=qid,
            top_k=3,
        )
        # Semantic Retrieval
        semantic_matches = semantic.retrieve(
            query_text=q_text,
            query_started_at=q_time,
            query_conversation_id=qid,
            top_k=3,
        )

        all_matches = [("lexical_bm25", m) for m in bm25_matches] + [("semantic", m) for m in semantic_matches]

        md_sections.append("### Retrieved Candidate Resolutions\n\n")
        md_sections.append("| Retriever | Rank | Cand ID | Score | Filter Status | Customer Issue | Tesco Resolution |\n")
        md_sections.append("|---|---|---|---|---|---|---|\n")

        for rtype, match in all_matches:
            ev = evidence_filter.evaluate(match)
            doc = match.document
            filter_status = "accepted" if ev.is_accepted else "rejected"
            rejection_str = "; ".join(ev.rejection_reasons)

            row = {
                "sample_id": sample_id,
                "query_id": qid,
                "query_started_at": q_time,
                "query_text": q_text,
                "retriever_type": rtype,
                "rank": match.rank,
                "candidate_id": doc.conversation_id,
                "score": round(match.score, 4),
                "filter_status": filter_status,
                "rejection_reasons": rejection_str,
                "candidate_customer_issue": doc.sanitized_customer_issue,
                "candidate_tesco_resolution": doc.sanitized_tesco_resolution,
                "human_relevance": "",  # To be filled by reviewer
                "relevance_notes": "",
            }
            rows.append(row)
            sample_id += 1

            clean_issue = doc.sanitized_customer_issue[:75] + ("..." if len(doc.sanitized_customer_issue) > 75 else "")
            clean_res = doc.sanitized_tesco_resolution[:85] + ("..." if len(doc.sanitized_tesco_resolution) > 85 else "")
            clean_issue = clean_issue.replace("|", "/")
            clean_res = clean_res.replace("|", "/")

            md_sections.append(
                f"| `{rtype}` | {match.rank} | `{doc.conversation_id}` | {match.score:.3f} | `{filter_status}` | {clean_issue} | {clean_res} |\n"
            )

        md_sections.append("\n---\n\n")

    # 6. Save CSV Template and Markdown Booklet
    output_csv_template.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv_template, index=False)
    print(f"Exported {len(df_out)} candidate rows to {output_csv_template}")

    output_md_booklet.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_booklet, "w", encoding="utf-8") as f:
        f.writelines(md_sections)
    print(f"Exported review booklet to {output_md_booklet}")
    print("=" * 60)


if __name__ == "__main__":
    main()
