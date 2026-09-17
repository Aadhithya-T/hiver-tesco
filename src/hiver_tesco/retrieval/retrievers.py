"""Lexical (BM25) and Semantic (dense embedding) retrievers with historical cutoff.

Enforces:
1. Indexing and ranking strictly on sanitized_customer_issue.
2. Returning sanitized_tesco_resolution as evidence payload.
3. Per-query historical cutoff (doc.ended_at <= query.started_at).
4. Strict self-exclusion (doc.conversation_id != query.conversation_id).
"""

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hiver_tesco.retrieval.corpus import HistoricalResolutionDoc


@dataclass
class CandidateMatch:
    """A retrieved historical candidate match."""

    document: HistoricalResolutionDoc
    score: float
    retriever_type: str
    rank: int


def _tokenize(text: str) -> List[str]:
    """Lightweight whitespace and punctuation tokenizer."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class BM25HistoricalRetriever:
    """BM25Okapi lexical retriever over historical customer issues."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[HistoricalResolutionDoc] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: np.ndarray = np.array([], dtype=np.float32)
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}
        self.doc_freqs: List[Dict[str, int]] = []
        self.ended_at_array: List[str] = []
        self.conv_ids: List[str] = []

    def index(self, documents: List[HistoricalResolutionDoc]) -> None:
        """Index documents strictly on sanitized_customer_issue."""
        self.documents = list(documents)
        self.conv_ids = [doc.conversation_id for doc in self.documents]
        self.ended_at_array = [doc.ended_at for doc in self.documents]

        num_docs = len(self.documents)
        self.doc_tokens = [_tokenize(doc.sanitized_customer_issue) for doc in self.documents]
        lens = [len(tokens) for tokens in self.doc_tokens]
        self.doc_lens = np.array(lens, dtype=np.float32)
        self.avg_doc_len = float(np.mean(self.doc_lens)) if num_docs > 0 else 1.0

        # Term document frequencies
        df: Dict[str, int] = {}
        self.doc_freqs = []
        for tokens in self.doc_tokens:
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.doc_freqs.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1

        # BM25 standard Robertson-Spärck Jones IDF
        self.idf = {}
        for t, freq in df.items():
            self.idf[t] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def retrieve(
        self,
        query_text: str,
        query_started_at: Optional[str] = None,
        query_conversation_id: Optional[str] = None,
        top_k: int = 3,
    ) -> List[CandidateMatch]:
        """Retrieve top-k historical resolutions for a query.

        Applies historical cutoff (doc.ended_at <= query_started_at) and self-exclusion.
        """
        if not self.documents:
            return []

        q_tokens = _tokenize(query_text)
        if not q_tokens:
            return []

        scores = np.zeros(len(self.documents), dtype=np.float32)

        for term in q_tokens:
            idf_val = self.idf.get(term, 0.0)
            if idf_val <= 0.0:
                continue

            for idx, tf_dict in enumerate(self.doc_freqs):
                tf = tf_dict.get(term, 0)
                if tf > 0:
                    doc_len = self.doc_lens[idx]
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    scores[idx] += idf_val * ((tf * (self.k1 + 1.0)) / denominator)

        # Apply leakage-safe temporal and self-exclusion masks
        for idx in range(len(self.documents)):
            if query_conversation_id and self.conv_ids[idx] == query_conversation_id:
                scores[idx] = -1.0
            elif query_started_at and self.ended_at_array[idx] > query_started_at:
                scores[idx] = -1.0

        valid_indices = np.where(scores > 0.0)[0]
        if len(valid_indices) == 0:
            return []

        sorted_valid = valid_indices[np.argsort(-scores[valid_indices])]
        top_indices = sorted_valid[:top_k]

        matches = []
        for rank, idx in enumerate(top_indices, start=1):
            matches.append(
                CandidateMatch(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    retriever_type="lexical_bm25",
                    rank=rank,
                )
            )

        return matches


class SemanticHistoricalRetriever:
    """Dense embedding retriever over historical customer issues using SentenceTransformers."""

    def __init__(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name_or_path, device=device)

        self.documents: List[HistoricalResolutionDoc] = []
        self.embeddings: np.ndarray = np.array([], dtype=np.float32)
        self.conv_ids: List[str] = []
        self.ended_at_array: List[str] = []

    def index(
        self,
        documents: List[HistoricalResolutionDoc],
        embeddings_cache_path: Optional[Path] = None,
    ) -> None:
        """Index documents strictly on sanitized_customer_issue, optionally loading/saving cached embeddings."""
        self.documents = list(documents)
        self.conv_ids = [doc.conversation_id for doc in self.documents]
        self.ended_at_array = [doc.ended_at for doc in self.documents]

        if embeddings_cache_path:
            embeddings_cache_path = Path(embeddings_cache_path)
            if embeddings_cache_path.exists():
                # Load precomputed embeddings
                self.embeddings = np.load(embeddings_cache_path)
                if self.embeddings.shape[0] != len(self.documents):
                    raise ValueError(
                        f"Cached embeddings count ({self.embeddings.shape[0]}) does not match "
                        f"documents count ({len(self.documents)})"
                    )
            else:
                self.embeddings = np.array([], dtype=np.float32)

        if self.embeddings.size == 0:
            # Encode strictly sanitized_customer_issue
            texts = [doc.sanitized_customer_issue for doc in self.documents]
            embs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            self.embeddings = embs.astype(np.float32)

            if embeddings_cache_path:
                embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(embeddings_cache_path, self.embeddings)

    def retrieve(
        self,
        query_text: str,
        query_started_at: Optional[str] = None,
        query_conversation_id: Optional[str] = None,
        top_k: int = 3,
    ) -> List[CandidateMatch]:
        """Retrieve top-k semantic candidate resolutions.

        Applies historical cutoff (doc.ended_at <= query_started_at) and self-exclusion.
        """
        if len(self.documents) == 0 or self.embeddings.size == 0:
            return []

        # Encode query
        q_emb = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        # Dot product with normalized document embeddings equals cosine similarity
        sims = np.dot(self.embeddings, q_emb)

        # Apply leakage-safe temporal and self-exclusion masks
        for idx in range(len(self.documents)):
            if query_conversation_id and self.conv_ids[idx] == query_conversation_id:
                sims[idx] = -2.0
            elif query_started_at and self.ended_at_array[idx] > query_started_at:
                sims[idx] = -2.0

        valid_indices = np.where(sims > -1.5)[0]
        if len(valid_indices) == 0:
            return []

        sorted_valid = valid_indices[np.argsort(-sims[valid_indices])]
        top_indices = sorted_valid[:top_k]

        matches = []
        for rank, idx in enumerate(top_indices, start=1):
            matches.append(
                CandidateMatch(
                    document=self.documents[idx],
                    score=float(sims[idx]),
                    retriever_type="semantic",
                    rank=rank,
                )
            )

        return matches
