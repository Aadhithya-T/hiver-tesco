"""Unit tests for Phase 4 historical retrieval and evidence filtering."""

import json
from pathlib import Path
import pytest
import pandas as pd

from hiver_tesco.retrieval.pii import detect_competitor_mentions, sanitize_evidence_text
from hiver_tesco.retrieval.corpus import (
    HistoricalResolutionDoc,
    build_historical_corpus,
)
from hiver_tesco.retrieval.retrievers import (
    BM25HistoricalRetriever,
    CandidateMatch,
)
from hiver_tesco.retrieval.evidence_filter import EvidenceFilter
from hiver_tesco.retrieval.relevance_validator import (
    RelevanceValidationError,
    validate_relevance_file,
)
from hiver_tesco.retrieval.evaluation import (
    compute_retrieval_metrics,
    compute_evidence_filter_stats,
    calibrate_semantic_threshold,
)


def test_competitor_detection_before_redaction():
    text = "Hey @Tesco, why is @Sainsburys selling bread cheaper? Even Asda is better."
    competitors = detect_competitor_mentions(text)
    assert "@sainsburys" in competitors
    assert "asda" in competitors

    # Redaction should preserve @Tesco and mask non-Tesco handles
    sanitized = sanitize_evidence_text(text)
    assert "@Tesco" in sanitized
    assert "[CUSTOMER]" in sanitized
    assert "@Sainsburys" not in sanitized


def test_pii_sanitization():
    raw_text = (
        "Hi @Tesco my order 12345678 failed. Delivery postcode was SW1A 1AA. "
        "Contact me at user@test.com or 07123456789. Thanks @customer_handle"
    )
    clean = sanitize_evidence_text(raw_text)
    assert "12345678" not in clean
    assert "[ORDER_REF]" in clean
    assert "SW1A 1AA" not in clean
    assert "[POSTCODE]" in clean
    assert "user@test.com" not in clean
    assert "[EMAIL]" in clean
    assert "07123456789" not in clean
    assert "[PHONE]" in clean
    assert "@customer_handle" not in clean
    assert "[CUSTOMER]" in clean
    assert "@Tesco" in clean


def test_golden_set_exclusion(tmp_path):
    conv_file = tmp_path / "convs.jsonl"
    data = [
        {
            "conversation_id": "100",
            "source_root_tweet_id": "100",
            "started_at": "2017-10-01T10:00:00+00:00",
            "ended_at": "2017-10-01T11:00:00+00:00",
            "outbound_count": 1,
            "quality_flags": {},
            "messages": [
                {"tweet_id": "100", "direction": "inbound", "text": "Customer issue 100"},
                {"tweet_id": "101", "direction": "outbound", "text": "Tesco reply 101"},
            ],
        },
        {
            "conversation_id": "200",  # Golden ID
            "source_root_tweet_id": "200",
            "started_at": "2017-10-02T10:00:00+00:00",
            "ended_at": "2017-10-02T11:00:00+00:00",
            "outbound_count": 1,
            "quality_flags": {},
            "messages": [
                {"tweet_id": "200", "direction": "inbound", "text": "Golden query"},
                {"tweet_id": "201", "direction": "outbound", "text": "Golden reply"},
            ],
        },
    ]
    with open(conv_file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    corpus = build_historical_corpus(conv_file, excluded_golden_ids={"200"})
    c_ids = [doc.conversation_id for doc in corpus]
    assert "100" in c_ids
    assert "200" not in c_ids


def test_per_query_cutoff_and_self_exclusion():
    docs = [
        HistoricalResolutionDoc(
            conversation_id="past_1",
            source_root_tweet_id="past_1",
            started_at="2017-10-01T10:00:00+00:00",
            ended_at="2017-10-01T11:00:00+00:00",
            customer_issue="Refund for milk",
            tesco_resolution="Refund issued",
            sanitized_customer_issue="refund for milk",
            sanitized_tesco_resolution="refund issued",
            outbound_tweet_ids=["101"],
            outbound_timestamps=["2017-10-01T11:00:00+00:00"],
            quality_flags={},
            competitor_mentions=[],
        ),
        HistoricalResolutionDoc(
            conversation_id="future_1",
            source_root_tweet_id="future_1",
            started_at="2017-10-15T10:00:00+00:00",
            ended_at="2017-10-15T11:00:00+00:00",
            customer_issue="Refund for milk again",
            tesco_resolution="Refund issued again",
            sanitized_customer_issue="refund for milk again",
            sanitized_tesco_resolution="refund issued again",
            outbound_tweet_ids=["102"],
            outbound_timestamps=["2017-10-15T11:00:00+00:00"],
            quality_flags={},
            competitor_mentions=[],
        ),
        HistoricalResolutionDoc(
            conversation_id="query_self",
            source_root_tweet_id="query_self",
            started_at="2017-10-01T09:00:00+00:00",
            ended_at="2017-10-01T09:30:00+00:00",
            customer_issue="Refund for milk self",
            tesco_resolution="Refund self",
            sanitized_customer_issue="refund for milk self",
            sanitized_tesco_resolution="refund self",
            outbound_tweet_ids=["103"],
            outbound_timestamps=["2017-10-01T09:30:00+00:00"],
            quality_flags={},
            competitor_mentions=[],
        ),
    ]

    retriever = BM25HistoricalRetriever()
    retriever.index(docs)

    # Query timestamp: 2017-10-10 (after past_1, before future_1)
    # Own conversation ID: query_self
    matches = retriever.retrieve(
        query_text="milk refund",
        query_started_at="2017-10-10T00:00:00+00:00",
        query_conversation_id="query_self",
        top_k=5,
    )

    retrieved_ids = [m.document.conversation_id for m in matches]
    assert "past_1" in retrieved_ids
    assert "future_1" not in retrieved_ids, "Future document should be cut off!"
    assert "query_self" not in retrieved_ids, "Self document should be excluded!"


def test_evidence_filter_rejections():
    doc_cross_brand = HistoricalResolutionDoc(
        conversation_id="1",
        source_root_tweet_id="1",
        started_at="2017-10-01T00:00:00+00:00",
        ended_at="2017-10-01T01:00:00+00:00",
        customer_issue="cross brand issue",
        tesco_resolution="reply",
        sanitized_customer_issue="cross brand issue",
        sanitized_tesco_resolution="reply",
        outbound_tweet_ids=["10"],
        outbound_timestamps=["2017-10-01T01:00:00+00:00"],
        quality_flags={"has_cross_brand_mention": True},
        competitor_mentions=["@sainsburys"],
    )

    match = CandidateMatch(
        document=doc_cross_brand,
        score=0.85,
        retriever_type="semantic",
        rank=1,
    )

    filt = EvidenceFilter(semantic_threshold=0.50)
    ev = filt.evaluate(match)
    assert not ev.is_accepted
    assert any("cross_brand" in r or "competitor" in r for r in ev.rejection_reasons)

    # Test weak semantic match
    doc_clean = HistoricalResolutionDoc(
        conversation_id="2",
        source_root_tweet_id="2",
        started_at="2017-10-01T00:00:00+00:00",
        ended_at="2017-10-01T01:00:00+00:00",
        customer_issue="clean issue",
        tesco_resolution="clean reply",
        sanitized_customer_issue="clean issue",
        sanitized_tesco_resolution="clean reply",
        outbound_tweet_ids=["20"],
        outbound_timestamps=["2017-10-01T01:00:00+00:00"],
        quality_flags={},
        competitor_mentions=[],
    )
    match_weak = CandidateMatch(
        document=doc_clean,
        score=0.32,  # Below threshold 0.50
        retriever_type="semantic",
        rank=1,
    )
    ev_weak = filt.evaluate(match_weak)
    assert not ev_weak.is_accepted
    assert any("weak_semantic_match" in r for r in ev_weak.rejection_reasons)


def test_relevance_validator_and_metrics(tmp_path):
    csv_file = tmp_path / "test_relevance.csv"

    # Incomplete file missing columns
    df_bad = pd.DataFrame({"query_id": ["q1"], "human_relevance": ["relevant"]})
    df_bad.to_csv(csv_file, index=False)
    with pytest.raises(RelevanceValidationError):
        validate_relevance_file(csv_file)

    # Complete valid dataframe
    df_valid = pd.DataFrame(
        [
            {
                "sample_id": 1,
                "query_id": "q1",
                "query_started_at": "2017-10-10T10:00:00+00:00",
                "query_text": "Missing bread",
                "retriever_type": "lexical_bm25",
                "rank": 1,
                "candidate_id": "c1",
                "score": 4.5,
                "filter_status": "accepted",
                "candidate_customer_issue": "Missing bread from order",
                "candidate_tesco_resolution": "Refunded bread",
                "human_relevance": "relevant",
                "relevance_notes": "Identical missing bread issue",
            },
            {
                "sample_id": 2,
                "query_id": "q1",
                "query_started_at": "2017-10-10T10:00:00+00:00",
                "query_text": "Missing bread",
                "retriever_type": "lexical_bm25",
                "rank": 2,
                "candidate_id": "c2",
                "score": 2.1,
                "filter_status": "accepted",
                "candidate_customer_issue": "Missing butter",
                "candidate_tesco_resolution": "Refunded butter",
                "human_relevance": "partially_relevant",
                "relevance_notes": "Missing grocery item",
            },
            {
                "sample_id": 3,
                "query_id": "q1",
                "query_started_at": "2017-10-10T10:00:00+00:00",
                "query_text": "Missing bread",
                "retriever_type": "lexical_bm25",
                "rank": 3,
                "candidate_id": "c3",
                "score": 1.0,
                "filter_status": "accepted",
                "candidate_customer_issue": "Store car park pothole",
                "candidate_tesco_resolution": "Reported to manager",
                "human_relevance": "irrelevant",
                "relevance_notes": "Car park issue",
            },
        ]
    )
    df_valid.to_csv(csv_file, index=False)
    validated = validate_relevance_file(csv_file)
    assert len(validated) == 3

    # Strict: only rank 1 is positive
    strict_m = compute_retrieval_metrics(validated, "lexical_bm25", strict=True, max_k=3)
    assert strict_m.hit_at_1 == 1.0
    assert strict_m.hit_at_3 == 1.0
    assert strict_m.precision_at_1 == 1.0
    assert pytest.approx(strict_m.precision_at_3, 0.01) == 1.0 / 3.0
    assert strict_m.mrr == 1.0

    # Relaxed: rank 1 and 2 are positive
    relaxed_m = compute_retrieval_metrics(validated, "lexical_bm25", strict=False, max_k=3)
    assert relaxed_m.hit_at_1 == 1.0
    assert relaxed_m.hit_at_3 == 1.0
    assert pytest.approx(relaxed_m.precision_at_3, 0.01) == 2.0 / 3.0
    assert relaxed_m.mrr == 1.0


def test_corpus_serialization_and_deserialization(tmp_path):
    corpus_file = tmp_path / "corpus.jsonl"
    doc = HistoricalResolutionDoc(
        conversation_id="c1",
        source_root_tweet_id="r1",
        started_at="2017-10-01T10:00:00+00:00",
        ended_at="2017-10-01T11:00:00+00:00",
        customer_issue="issue",
        tesco_resolution="resolution",
        sanitized_customer_issue="clean issue",
        sanitized_tesco_resolution="clean resolution",
        outbound_tweet_ids=["t1"],
        outbound_timestamps=["2017-10-01T11:00:00+00:00"],
        quality_flags={"flag": True},
        competitor_mentions=[],
    )

    from hiver_tesco.retrieval.corpus import save_corpus_jsonl, load_corpus_jsonl
    save_corpus_jsonl([doc], corpus_file)
    loaded = load_corpus_jsonl(corpus_file)
    assert len(loaded) == 1
    assert loaded[0].conversation_id == "c1"
    assert loaded[0].sanitized_customer_issue == "clean issue"


def test_validation_errors(tmp_path):
    csv_file = tmp_path / "bad.csv"
    # Empty file
    df_empty = pd.DataFrame(columns=[
        "sample_id", "query_id", "query_started_at", "query_text", "retriever_type",
        "rank", "candidate_id", "score", "filter_status", "candidate_customer_issue",
        "candidate_tesco_resolution", "human_relevance", "relevance_notes"
    ])
    df_empty.to_csv(csv_file, index=False)
    with pytest.raises(RelevanceValidationError, match="Annotation file is empty"):
        validate_relevance_file(csv_file)

    # Invalid label
    df_invalid = pd.DataFrame([{
        "sample_id": 1, "query_id": "q1", "query_started_at": "2017-10-01",
        "query_text": "text", "retriever_type": "bm25", "rank": 1, "candidate_id": "c1",
        "score": 1.0, "filter_status": "accepted", "candidate_customer_issue": "issue",
        "candidate_tesco_resolution": "res", "human_relevance": "invalid_label",
        "relevance_notes": "notes"
    }])
    df_invalid.to_csv(csv_file, index=False)
    with pytest.raises(RelevanceValidationError, match="Invalid label"):
        validate_relevance_file(csv_file)


def test_full_evaluation_pipeline(tmp_path):
    csv_file = tmp_path / "valid_sample.csv"
    df = pd.DataFrame([
        {
            "sample_id": 1, "query_id": "q1", "query_started_at": "2017-10-01",
            "query_text": "issue 1", "retriever_type": "lexical_bm25", "rank": 1, "candidate_id": "c1",
            "score": 10.0, "filter_status": "accepted", "rejection_reasons": "",
            "candidate_customer_issue": "issue 1", "candidate_tesco_resolution": "res 1",
            "human_relevance": "relevant", "relevance_notes": "match"
        },
        {
            "sample_id": 2, "query_id": "q1", "query_started_at": "2017-10-01",
            "query_text": "issue 1", "retriever_type": "semantic", "rank": 1, "candidate_id": "c1",
            "score": 0.85, "filter_status": "accepted", "rejection_reasons": "",
            "candidate_customer_issue": "issue 1", "candidate_tesco_resolution": "res 1",
            "human_relevance": "relevant", "relevance_notes": "match"
        },
    ])
    df.to_csv(csv_file, index=False)
    from hiver_tesco.retrieval.evaluation import evaluate_retrieval_from_file
    report = evaluate_retrieval_from_file(csv_file)
    assert report.strict_metrics["semantic"].hit_at_1 == 1.0
    assert report.relaxed_metrics["lexical_bm25"].hit_at_1 == 1.0
    assert report.total_evaluated_candidates == 2
