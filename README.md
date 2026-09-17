# Hiver SDE Take-Home: Phase 1 -- Tesco Conversation Extraction & Reconstruction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-23%20passed-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deterministic, idempotent conversation extraction, thread reconstruction, and taxonomy-sampling pipeline built on the Tesco subset of the Twitter Customer Support (TWCS) dataset.

This repository implements **Phase 1** of a customer support AI assistant for Tesco: preparing clean, structured, anomaly-flagged conversation threads suitable for human taxonomy discovery, without inventing schemas or using premature LLMs/classifiers.

---

## Table of Contents
- [Project Architecture](#project-architecture)
- [Quickstart & Setup](#quickstart--setup)
- [Running the Pipeline](#running-the-pipeline)
  - [1. Dataset Inspection](#1-dataset-inspection)
  - [2. Conversation Extraction & Reconstruction](#2-conversation-extraction--reconstruction)
  - [3. Human Taxonomy Discovery Sampling](#3-human-taxonomy-discovery-sampling)
- [Validated Dataset Assumptions](#validated-dataset-assumptions)
- [Known Dataset Limitations & Anomalies](#known-dataset-limitations--anomalies)
- [Output Artifacts & Schemas](#output-artifacts--schemas)
  - [`conversations.jsonl`](#conversationsjsonl)
  - [`audit_report.json` & `audit_report.csv`](#audit_reportjson--audit_reportcsv)
  - [`conversation_sample_200.md`](#conversation_sample_200md)
- [Observed Real-Data Statistics](#observed-real-data-statistics)
- [Testing & Verification](#testing--verification)
- [Roadmap: Next Phases (Not Yet Implemented)](#roadmap-next-phases-not-yet-implemented)

---

## Project Architecture

```
hiver-tesco/
├── .gitignore               # Excludes raw datasets (*.csv), outputs/, caches, and venvs
├── pyproject.toml           # PEP 517/621 build configuration
├── requirements.txt         # Core dependencies (pandas, pytest, pytest-cov)
├── README.md                # Engineering documentation
├── data/
│   └── README.md            # Raw data placement and schema guidelines
├── outputs/                 # Generated artifacts (ignored from git)
│   ├── conversations.jsonl  # 16,565 reconstructed conversation records
│   ├── audit_report.json    # Full funnel & anomaly audit metrics
│   ├── audit_report.csv     # Flat metrics table
│   └── samples/
│       ├── conversation_sample_200.md    # Human review booklet
│       └── conversation_sample_200.jsonl # Sampled JSONL records
├── src/
│   └── hiver_tesco/
│       ├── __init__.py      # Package entry point
│       ├── config.py        # Constants, handle regex, PipelineConfig dataclass
│       ├── models.py        # Strongly-typed dataclasses (RawTweet, ThreadMessage, etc.)
│       ├── loader.py        # Safe CSV streaming, validation, redaction, schema reporting
│       ├── direction.py     # Explicit inbound vs outbound role classification
│       ├── anomalies.py     # Cycles, missing parents, branching, competitor mentions
│       ├── graph.py         # Tree building, topological sort, depth computation
│       ├── sampler.py       # Deterministic length-stratified sampling & markdown review booklet
│       ├── audit.py         # Stage funnel tracking and JSON/CSV reporting
│       └── pipeline.py      # End-to-end extraction orchestrator
├── scripts/
│   ├── inspect_dataset.py       # Standalone inspection CLI
│   ├── run_pipeline.py          # Main pipeline runner CLI
│   └── sample_conversations.py  # Standalone human taxonomy sampler CLI
└── tests/
    ├── conftest.py              # Synthetic thread fixtures
    ├── test_loader.py           # Ingestion, validation, and error tests
    ├── test_direction.py        # Role and ambiguous ownership tests
    ├── test_anomalies.py        # Cycle detection, cross-brand regex, anomaly flags
    ├── test_graph.py            # Graph reconstruction, ordering, branching tests
    ├── test_sampler.py          # Determinism and stratification tests
    └── test_pipeline_e2e.py     # Full integration & idempotency tests
```

---

## Quickstart & Setup

### Requirements
- Python 3.9+ (tested on Python 3.14)
- Pip / virtualenv

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Aadhithya-T/hiver-tesco.git
cd hiver-tesco

python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or editable install:
pip install -e .
```

---

## Running the Pipeline

### 1. Dataset Inspection
To inspect the local CSV schema, non-null counts, data types, and a redacted preview without running the extraction pipeline:
```bash
python scripts/inspect_dataset.py --input-path tesco_tweets.csv
```

### 2. Conversation Extraction & Reconstruction
Run the complete pipeline to load raw tweets, reconstruct threads, label message direction, flag anomalies, export `conversations.jsonl`, write audit reports, and generate human inspection samples:
```bash
python scripts/run_pipeline.py \
  --input-path tesco_tweets.csv \
  --output-dir outputs \
  --brand-id Tesco \
  --seed 42 \
  --sample-size 200
```

CLI Parameters:
- `--input-path`: Path to input dataset CSV (default: `tesco_tweets.csv`).
- `--output-dir`: Destination folder for output artifacts (default: `outputs`).
- `--brand-id`: Account identifier for brand (default: `Tesco`).
- `--seed`: Integer seed for deterministic sampling and tie-breaking (default: `42`).
- `--sample-size`: Number of diverse conversations to sample for manual review (default: `200`).

### 3. Human Taxonomy Discovery Sampling
To re-sample conversations with different seeds or sample sizes from an already-generated `conversations.jsonl`:
```bash
python scripts/sample_conversations.py \
  --input-jsonl outputs/conversations.jsonl \
  --output-dir outputs/samples \
  --sample-size 200 \
  --seed 42
```

---

## Validated Dataset Assumptions

The following assumptions were **empirically validated** against the actual `tesco_tweets.csv` file before writing pipeline logic:

1. **File Format & Delimiter**: Valid comma-separated values (CSV) with double-quoted multiline text, UTF-8 encoded with emojis (e.g. 😡, 😂) and currency symbols (£).
2. **Schema & Columns**: Exactly 8 columns present: `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`, `conversation_id`.
3. **Primary Key**: `tweet_id` is unique across all 70,536 rows; 0 duplicates exist in the raw dataset.
4. **Author Identification**: `author_id` is `'Tesco'` for 37,974 brand tweets, and numeric IDs (e.g. `'115881'`) for 32,562 customer tweets across 20,446 distinct customers.
5. **Inbound Flag Consistency**: `inbound == 'FALSE'` matches `author_id == 'Tesco'` 1:1; `inbound == 'TRUE'` matches customer authors 1:1.
6. **Timestamps**: All 70,536 rows match Twitter timestamp format `%a %b %d %H:%M:%S %z %Y` cleanly without nulls or parsing errors, spanning `2014-09-07` to `2017-12-03`.
7. **Parent Link Integrity**: In 100% of the 53,970 parent-child links, the child timestamp is strictly greater than the parent timestamp (`created_at_child > created_at_parent`). There are 0 timestamp inversions.
8. **Conversation Participation**: 100% of the 16,565 conversations contain both Tesco and at least one customer. There are zero singletons (minimum length is 2 tweets).

---

## Known Dataset Limitations & Anomalies

1. **`response_tweet_id` Float Precision Corruption**:
   - In upstream processing, whenever a tweet received multiple responses, the comma-separated IDs were coerced into floating-point numbers or formatted with commas (e.g., `2,50,52,50,62,50,72,50,00,00,00,00...000`), truncating significant digits beyond 15 digits.
   - **Resolution**: The pipeline relies solely on `in_response_to_tweet_id` (which is clean and uncorrupted across all 70,536 rows) to construct the thread graph.
2. **Missing Parent Tweets (Orphaned Sub-trees)**:
   - 67 tweets reference parent IDs that are not present in `tesco_tweets.csv`.
   - In all 67 cases, the missing parent ID equals the `conversation_id` (i.e. the original root tweet was not ingested, likely because the root was deleted or did not mention Tesco explicitly).
   - **Resolution**: The pipeline detects orphaned sub-trees, selects the earliest available tweet as canonical root, sets `source_root_tweet_id = None`, and flags `has_missing_parent = True`.
3. **Branching & Multi-part Replies**:
   - 8,048 conversations (48.58%) have branching replies. In customer support, Tesco agents frequently break long messages into multi-part replies (e.g., `1/2` and `2/2`), both replying to the same parent tweet.
   - **Resolution**: The pipeline computes explicit `child_tweet_ids` and tree `depth` on every message while maintaining deterministic chronological ordering.
4. **Broadcast / Marketing Root Tweets**:
   - 100 conversations start with a tweet authored by Tesco. 69 of these are marketing promotions (e.g. Christmas food advertisements, baking competitions) that attracted customer comments.
   - **Resolution**: The pipeline flags these with `is_outbound_root = True` so downstream intent modeling can separate marketing chatter from genuine customer support inquiries.
5. **Cross-Brand Mentions**:
   - 232 conversations mention competitor supermarket handles (`@sainsburys`, `@asda`, `@morrisons`, etc.).
   - **Resolution**: Flagged with `has_cross_brand_mention = True` and handle list attached for competitor comparison analysis.

---

## Output Artifacts & Schemas

### `conversations.jsonl`
Each line is a complete, self-contained JSON record for a reconstructed conversation.

```json
{
  "conversation_id": "789",
  "source_root_tweet_id": "789",
  "message_count": 5,
  "inbound_count": 2,
  "outbound_count": 3,
  "customer_author_ids": ["115881"],
  "started_at": "2017-10-31T18:02:11+00:00",
  "ended_at": "2017-10-31T22:18:59+00:00",
  "duration_seconds": 15408.0,
  "quality_flags": {
    "has_missing_parent": false,
    "has_cycle": false,
    "has_duplicate_tweet_id": false,
    "has_cross_brand_mention": false,
    "is_outbound_root": false,
    "is_multi_customer": false,
    "is_branching": true,
    "has_empty_text": false,
    "is_ambiguous_ownership": false,
    "missing_parent_ids": [],
    "cross_brand_handles": []
  },
  "messages": [
    {
      "tweet_id": "789",
      "parent_tweet_id": null,
      "child_tweet_ids": ["788"],
      "author_id": "115881",
      "direction": "inbound",
      "created_at": "2017-10-31T18:02:11+00:00",
      "text": "@Tesco 3/3 on the wrong which I really don’t appreciate...",
      "turn_index": 0,
      "depth": 0,
      "is_root": true
    },
    {
      "tweet_id": "788",
      "parent_tweet_id": "789",
      "child_tweet_ids": ["786"],
      "author_id": "Tesco",
      "direction": "outbound",
      "created_at": "2017-10-31T19:47:00+00:00",
      "text": "@115881 Hi Cade, what was my colleagues name?...",
      "turn_index": 1,
      "depth": 1,
      "is_root": false
    }
  ],
  "provenance": {
    "pipeline_phase": "Phase 1 - Extraction & Reconstruction",
    "sorting_rule": "chronological_by_created_at_then_tweet_id",
    "brand_id": "Tesco"
  }
}
```

### `audit_report.json` & `audit_report.csv`
Captures funnel metrics from input rows to clean conversations:
- Total input rows, unique tweet IDs, duplicate IDs.
- Total conversations discovered, brand-filtered, and reconstructed.
- Clean vs flagged conversation counts.
- Frequency table of anomalies (missing parents, cross-brand, broadcast roots, etc.).
- Conversation length distribution buckets (lengths 2, 3, 4, 5, 6, 7-10, 11+).

### `conversation_sample_200.md`
Human review booklet formatted with:
- Conversation summary metadata and quality tags.
- Blockquoted turn-by-turn dialogue with speaker labels (`[CUSTOMER (Inbound)]` vs `[TESCO (Outbound)]`), timestamps, and parent reply indicators.
- Interactive annotation checkboxes for intent discovery (`Customer Primary Intent`, `Secondary Topic`, `Resolution Status`, `Notes`).

---

## Observed Real-Data Statistics

Below are the exact metrics produced by running Phase 1 on the 14.3 MB `tesco_tweets.csv`:

| Metric | Observed Count | Percentage |
| :--- | :--- | :--- |
| **Total Ingested Rows** | 70,536 | 100.0% |
| **Unique Tweet IDs** | 70,536 | 100.0% |
| **Duplicate Tweet IDs** | 0 | 0.0% |
| **Total Conversations Reconstructed** | 16,565 | 100.0% |
| **Clean Conversations** | 16,200 | 97.80% |
| **Flagged Conversations** | 365 | 2.20% |
| - *Cross-Brand Mentions* | 232 | 1.40% |
| - *Outbound Roots (Broadcast / Proactive)* | 100 | 0.60% |
| - *Missing Parents* | 66 | 0.40% |
| - *Reply Graph Cycles* | 0 | 0.0% |
| - *Empty Tweet Content* | 0 | 0.0% |
| - *Ambiguous Ownership* | 0 | 0.0% |
| **Structural Characteristics** | | |
| - *Branching Threads (multi-part / multi-reply)* | 8,048 | 48.58% |
| - *Multi-Customer Conversations* | 1,003 | 6.05% |
| **Length Distribution** | | |
| - *Length 2* | 5,092 | 30.74% |
| - *Length 3* | 3,042 | 18.36% |
| - *Length 4* | 2,973 | 17.95% |
| - *Length 5* | 1,941 | 11.72% |
| - *Length 6* | 1,222 | 7.38% |
| - *Length 7 to 10* | 1,754 | 10.59% |
| - *Length 11+* | 541 | 3.27% |

---

## Testing & Verification

The project includes unit, anomaly, graph, and end-to-end integration tests with synthetic fixtures.

```bash
# Run test suite
python -m pytest

# Run test suite with code coverage
python -m pytest --cov=hiver_tesco
```

Test coverage: **90%** across all modules with 23 passing tests covering:
- Safe CSV loading, invalid schema rejection, and bad timestamp handling.
- Linear, branching, missing-parent, and cyclic graph reconstruction.
- Inbound vs outbound role labeling and conflicting ownership detection.
- Competitor supermarket handle matching regex.
- Stratified sampling reproducibility with seeds.
- Bit-for-bit idempotency across multiple pipeline executions.

---

## Roadmap: Next Phases (Not Yet Implemented)

Phase 1 provides the clean, reproducible data foundation. Subsequent phases will build upon this without modifying raw data assumptions:

1. **Human-Derived Taxonomy Discovery**:
   - Manually review the 200 stratified conversations in `outputs/samples/conversation_sample_200.md`.
   - Inductively derive 8–15 mutually exclusive, collectively exhaustive intent categories (e.g. *Delivery Delay*, *In-Store Experience / Colleague Conduct*, *Damaged / Spoiled Produce*, *Clubcard & Promotions*, *Stock Availability*, *Refund & Overcharge*).
2. **Golden Test Set (150–250 hand-labelled conversations)**:
   - Hand-annotate 150–250 golden conversations with consensus intent labels, key customer entities (postcode, order number), and expected resolution routes.
3. **Trivial & Heuristic Baselines**:
   - Implement keyword and regex intent routers and measure baseline accuracy against the golden set before introducing machine learning.
4. **Semantic Retrieval using Historical Tesco Resolutions**:
   - Build vector retrieval over historical agent responses to surface precedent resolutions for incoming customer queries.
5. **Evidence Filtering & Hallucination Prevention**:
   - Filter customer tweets for concrete evidence (timestamps, product barcodes, store locations) and refuse or escalate if critical info is absent.
6. **Deterministic Escalation Rules**:
   - Implement deterministic safeguards for immediate escalation to human agents (e.g. profanity, severe hygiene issues, legal threats, repeated failures).
7. **LLM Reply Generation with Grounded Context**:
   - Ground response generation in retrieved Tesco policies and historical agent tones.
8. **Automated Evaluation & LLM Judge Validated Against Humans**:
   - Build automated metrics (intent accuracy, policy compliance, tone) and validate an LLM-as-a-judge against human grading on the golden test set.
9. **Failure Analysis**:
   - Conduct systematic error categorization across ambiguity, retrieval misses, and out-of-domain conversations.
