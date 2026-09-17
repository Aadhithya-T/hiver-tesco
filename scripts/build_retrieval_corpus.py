"""Script to extract historical Tesco resolution corpus and cache embeddings.

Quarantines the complete 200-conversation golden evaluation set, validates
clean conversation quality flags, sanitizes customer PII, and precomputes
Sentence-Transformer dense embeddings for the ~16,000 historical documents.
"""

from pathlib import Path
import sys
import time

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from hiver_tesco.retrieval.corpus import (
    build_historical_corpus,
    save_corpus_jsonl,
)
from hiver_tesco.retrieval.retrievers import SemanticHistoricalRetriever


def main():
    repo_root = Path(__file__).resolve().parent.parent
    conversations_path = repo_root / "outputs" / "conversations.jsonl"
    golden_csv_path = repo_root / "data" / "golden" / "golden_set_v1.0.csv"
    output_corpus_path = repo_root / "outputs" / "retrieval" / "historical_corpus.jsonl"
    embeddings_cache_path = repo_root / "outputs" / "retrieval" / "corpus_embeddings.npy"

    print("=" * 60)
    print("PHASE 4: HISTORICAL RESOLUTION CORPUS BUILDER")
    print("=" * 60)

    # 1. Load Golden Set IDs
    golden_df = pd.read_csv(golden_csv_path)
    golden_ids = set(golden_df["conversation_id"].astype(str))
    print(f"Quarantined golden evaluation IDs: {len(golden_ids)}")

    # 2. Extract Eligible Historical Documents
    start_time = time.time()
    print("Extracting eligible historical resolution documents from outputs/conversations.jsonl...")
    corpus = build_historical_corpus(
        conversations_path=conversations_path,
        excluded_golden_ids=golden_ids,
        require_clean_flags=True,
    )
    extract_duration = time.time() - start_time
    print(f"Extracted {len(corpus):,} eligible resolution documents in {extract_duration:.2f}s.")

    # 3. Verify Golden Set Quarantine
    corpus_ids = set(doc.conversation_id for doc in corpus)
    overlap = corpus_ids.intersection(golden_ids)
    if overlap:
        raise RuntimeError(f"FATAL: Golden set leakage detected! Overlapping IDs: {overlap}")
    print("Verified golden set quarantine: 0 golden IDs present in historical corpus.")

    # 4. Save Historical Corpus JSONL
    print(f"Saving corpus to {output_corpus_path}...")
    save_corpus_jsonl(corpus, output_corpus_path)
    print(f"Saved {len(corpus):,} documents.")

    # 5. Precompute Sentence-Transformer Embeddings
    print("Precomputing dense embeddings using all-MiniLM-L6-v2...")
    emb_start = time.time()
    retriever = SemanticHistoricalRetriever(model_name_or_path="all-MiniLM-L6-v2", batch_size=128)
    retriever.index(corpus, embeddings_cache_path=embeddings_cache_path)
    emb_duration = time.time() - emb_start
    print(
        f"Encoded and cached {retriever.embeddings.shape[0]:,} embeddings "
        f"to {embeddings_cache_path} in {emb_duration:.2f}s "
        f"(device: {retriever.device})."
    )

    # 6. Corpus Statistics Summary
    earliest = min(doc.started_at for doc in corpus)
    latest = max(doc.ended_at for doc in corpus)
    print("-" * 60)
    print(f"Corpus size: {len(corpus):,} documents")
    print(f"Date range: {earliest} to {latest}")
    print(f"Embedding matrix shape: {retriever.embeddings.shape}")
    print("Corpus build complete successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
