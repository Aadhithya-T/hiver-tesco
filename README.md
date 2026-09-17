# Tesco Customer Support AI Assistant — Evidence-First Engineering Submission

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Test Suite](https://img.shields.io/badge/pytest-99%20passed-green.svg)]()
[![Zero-Network Tests](https://img.shields.io/badge/tests-100%25%20hermetic-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, evidence-grounded customer support pipeline built on real-world Tesco Twitter Customer Support (TWCS) data.

This project implements a production-grade, risk-managed support assistant that pairs **strict deterministic policy routing** with **evidence-grounded drafting**, ensuring zero unauthorized claims (refunds, account changes, promises) and total auditability.

---

## Table of Contents
1. [Problem Framing & Non-Goals](#1-problem-framing--non-goals)
2. [High-Level Architecture & Data Flow](#2-high-level-architecture--data-flow)
3. [Dataset Provenance & Real-World Limitations](#3-dataset-provenance--real-world-limitations)
4. [Human Taxonomy & Golden Set Methodology](#4-human-taxonomy--golden-set-methodology)
5. [Baseline vs. Final-System Results](#5-baseline-vs-final-system-results)
   - [Intent Classification](#intent-classification)
   - [Escalation Routing & Safety Trade-Off](#escalation-routing--safety-trade-off)
   - [Historical Retrieval & Evidence Grounding](#historical-retrieval--evidence-grounding)
6. [Evidence-Grounded Reply Generation & Guardrails](#6-evidence-grounded-reply-generation--guardrails)
7. [Evaluation, LLM Judge Validation & Human Comparison](#7-evaluation-llm-judge-validation--human-comparison)
8. [Failure Analysis, Risks & Unresolved Boundaries](#8-failure-analysis-risks--unresolved-boundaries)
9. [Reproducibility Instructions (One-Command Run)](#9-reproducibility-instructions-one-command-run)
10. [Test Suite & Verification](#10-test-suite--verification)
11. [Project Directory & Artifact Map](#11-project-directory--artifact-map)

---

## 1. Problem Framing & Non-Goals

### The Core Problem
Public social-media customer care for a major grocery retailer (Tesco) involves high volume, high legal/safety stakes (food contamination, allergy risks, abusive interactions), and frequent attempts by customers to seek financial restitution or status updates.

Deploying an unconstrained LLM directly to customer tweets introduces severe liability:
- **Hallucinated promises**: Inventing refund authorizations or delivery arrival times.
- **Unauthorized commitments**: Claiming account access or order modification capabilities that a Twitter bot does not possess.
- **Tone & safety failures**: Apologizing for praise, or mishandling harassment and food-poisoning reports.

### The Solution: Policy First, Grounding Second
1. **Deterministic Safety Gate**: Before spending any LLM tokens or retrieving evidence, an auditable deterministic policy evaluates pre-resolution signals. High-risk issues (harassment, food safety, payment disputes, account security) are immediately routed to human queues (`ESCALATE`), halting reply drafting with **0 LLM tokens spent**.
2. **Strict Evidence Grounding**: For safe, routine queries (`RESPOND`), the model is strictly constrained to historical Tesco resolutions retrieved via a leak-safe semantic search. The LLM acts as an editorial summarizer, not an autonomous agent.
3. **Multi-Tier Safety Guardrails**: Pre-prompt PII and name redaction, strict structured JSON output schema, and heuristic sentiment-conflict checks to prevent apologizing on compliments.

### Non-Goals (What This System Deliberately Does NOT Do)
- **Autonomous Financial Actions**: The system will **never** issue refunds, waive fees, or promise compensation.
- **Direct Database / Account Mutability**: The bot does not authenticate users, alter loyalty accounts, or check internal logistics systems.
- **Unconstrained Conversational Banter**: The bot will not engage in open-ended creative dialogue outside verified Tesco customer support topics.
- **Hiding Uncertainty Behind LLM Judges**: System quality is proven by human review, automated rule compliance, and position-bias audits, never by LLM judging alone.

---

## 2. High-Level Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Ingestion["Phase 1: Ingestion & Anomaly Auditing"]
        CSV["Raw Tesco CSV<br/>(65,308 tweets)"] --> Parser["Safe CSV Parser<br/>& Direction Classifier"]
        Parser --> Graph["Graph Thread Builder<br/>Topological Sort"]
        Graph --> Anomaly["Cycle & Cross-Brand<br/>Anomaly Scanner"]
        Anomaly --> ConvPath["outputs/conversations.jsonl<br/>(16,565 reconstructed threads)"]
    end

    subgraph Golden["Phase 2: Golden Dataset & Splitting"]
        ConvPath --> StratSample["Deterministic 3-Tier Sampler<br/>(N=200, seed=42)"]
        StratSample --> HumanTax["Human Taxonomy Labeling<br/>(8 mutually exclusive classes)"]
        HumanTax --> Split["Frozen Splitting<br/>Train: 120 | Dev: 40 | Test: 40"]
    end

    subgraph Retrieval["Phase 4: Historical Evidence Retrieval"]
        ConvPath -.->|"Exclude 200 Golden Sets"| Corpus["outputs/retrieval/historical_corpus.jsonl<br/>(16,365 documents)"]
        Corpus --> Embeds["SentenceTransformers (all-MiniLM-L6-v2)<br/>Sanitized Customer Issue Index"]
    end

    subgraph PolicyLayer["Phase 5: Deterministic Policy Engine"]
        Split --> DevQuery["Dev / Test Query Text<br/>(Turn 1 Inbound Only)"]
        DevQuery --> IntentModel["TF-IDF LogReg Intent Baseline<br/>(Trained on 120 Train)"]
        DevQuery --> Embeds
        Embeds --> RetMatch["Evidence Match & Filter<br/>(Historical Cutoff enforced)"]
        
        IntentModel --> PolicyEngine["Deterministic Policy Engine<br/>Priority Rules 1-5"]
        RetMatch --> PolicyEngine
        
        PolicyEngine -->|ESCALATE (19/40 Dev)| HumanQueue["Halt Drafting (0 LLM Tokens)<br/>Route to Suggested Routing Category*"]
        PolicyEngine -->|RESPOND (21/40 Dev)| GenPipeline["Eligible for Reply Drafting"]
    end

    subgraph Generation["Phase 6: Grounded Reply Generation"]
        GenPipeline --> PIIScrub["PII & Name Sanitization<br/>- Redact handles, emails, phones, order refs<br/>- Generic greeting 'Hi,' / 'Hello,'<br/>- Redact evidence names to [CUSTOMER]"]
        PIIScrub --> LLMCall{"Configurable LLM Provider<br/>(Mock / Gemini / OpenAI / Ollama)"}
        LLMCall --> StrictJSON["Strict Structured Output<br/>{'status': 'draft' | 'escalate'}"]
        StrictJSON --> SentimentCheck{"Sentiment Safety Check<br/>Heuristic Conflict Interceptor"}
        SentimentCheck -->|Conflict: Apology on Praise| TemplateFallback["Fallback to Template Reply"]
        SentimentCheck -->|Aligned| FinalReply["Delivered Grounded Reply"]
    end

    subgraph Evaluation["Phase 7: Multi-Tier Evaluation"]
        FinalReply --> AutoChecks["Automated Quality Audits<br/>(Length, Unsafe Claims, Near-Copy)"]
        FinalReply --> BlindedJudge["Blinded Pairwise Judge<br/>Forward + Reversed Passes"]
        FinalReply --> HumanComp["Truly Blinded Human Review<br/>(No intent, escalation, or split shown)"]
        AutoChecks --> FinalReport["Phase 7 Evaluation Report"]
        BlindedJudge --> FinalReport
        HumanComp --> FinalReport
    end
```

*\*Note: Suggested routing categories are project routing labels, not verified Tesco corporate team names.*

---

## 3. Dataset Provenance & Real-World Limitations

### Source Dataset
The source data is the Tesco slice of the public **Twitter Customer Support (TWCS)** Kaggle dataset:
- **Total records processed**: 65,308 tweets.
- **Date range**: October 2017 to November 2017.
- **Brand identifier**: `Tesco` (inbound customers `@Tesco`, outbound agents `Tesco`).

### Observed Anomalies & Real-World Dirty Data
Real-world social data contains significant structural noise that naive processors miss. The Phase 1 pipeline explicitly handles:

| Anomaly Type | Count | Rate | Impact & Mitigation |
|---|:---:|:---:|---|
| **Missing Parents (`in_response_to_tweet_id` missing)** | 10,578 | 16.20% | Handled via synthetic root nodes (`missing_parent_root`). Child turns preserved. |
| **Branching Threads** | 761 | 4.59% | Tree traversal maintains branching structures without dropping responses. |
| **Cross-Brand / Competitor Mentions** | 1,842 | 2.82% | Tweets mentioning `@Morrisons`, `@AldiUK`, `@Sainsburys` flagged to prevent brand confusion. |
| **Truncated / Displaced Inbounds** | 2,110 | 3.23% | Conversations starting mid-thread detected and isolated. |
| **Cycles in Tweet References** | 0 | 0.00% | Cycle detection algorithm ensures DAG integrity before topological sorting. |

---

## 4. Human Taxonomy & Golden Set Methodology

### 8-Class Mutually Exclusive Human Taxonomy
Derived through exploratory sampling of 200 candidate threads across complexity tiers:

| Intent Category | Primary Description | Escalation Default |
|---|---|:---:|
| `order_delivery_delay_and_tracking` | Grocery van late, delivery tracking slot inquiries | `do_not_escalate` |
| `order_issue_damaged_wrong_missing` | Crushed eggs, missing items, spoiled produce | `do_not_escalate` |
| `product_availability_and_stock` | Item out of stock in store, stock check requests | `do_not_escalate` |
| `refund_payment_and_billing` | Charged twice, voucher failure, refund inquiries | `escalate` |
| `clubcard_account_and_app_support` | Password resets, loyalty points, app crash | `escalate` |
| `in_store_experience_and_facilities` | Dirty trolley, parking ticket, staff praise | `do_not_escalate` |
| `general_feedback_and_chitchat` | Compliments, funny packaging banter, jokes | `do_not_escalate` |
| `critical_risk_and_safety_alert` | Glass in food, severe allergy, harassment, legal threat | `escalate` |

### Frozen Three-Way Split ($N=200$)
To prevent test leakage and guarantee scientific evaluation:
- **Train (120 conversations, 60%)**: Used to fit the baseline intent classifier and build lexical term indices.
- **Dev (40 conversations, 20%)**: Used for threshold tuning, risk keyword development, and Phase 6 prompt validation.
- **Test (40 conversations, 20%)**: Held out in strict quarantine. Tested **once** on the final locked system.

---

## 5. Baseline vs. Final-System Results

### Intent Classification
Evaluated on frozen held-out splits (macro-averaged):

| Model | Dev Accuracy | Dev Macro F1 | Test Accuracy | Test Macro F1 | Notes |
|---|:---:|:---:|:---:|:---:|---|
| **Majority Class** | 0.225 | 0.046 | 0.200 | 0.042 | Predicts `order_delivery_delay_and_tracking` |
| **TF-IDF + Logistic Regression** | **0.550** | **0.449** | **0.600** | **0.505** | Character & word n-grams (1,2), balanced class weights |

### Escalation Routing & Safety Trade-Off
Evaluated on Dev Split ($N=40$):

| Escalation Model / Layer | Accuracy | Precision | Recall | F1 Score | False Negatives (Safety Risk) | False Positives (Operator Cost) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always Do Not Escalate** | 0.525 | 0.000 | 0.000 | 0.000 | 19 | 0 |
| **Majority Class Escalate** | 0.475 | 0.475 | 1.000 | 0.644 | 0 | 21 |
| **Phase 3 Keyword Regex** | **0.675** | **0.714** | 0.526 | 0.606 | 9 | **4** |
| **Phase 5 Deterministic Policy** | 0.650 | 0.632 | **0.632** | **0.632** | **7** | 7 |

#### Honest Assessment of Policy Results:
- **Caught 2 Extra True Escalations**: Missed escalations dropped from 9 to 7 (Recall increased from 52.6% to 63.2%).
- **Created 3 Extra False Alarms**: Unnecessary escalations increased from 4 to 7.
- **The Trade-Off**: Overall accuracy dropped slightly from 67.5% to 65.0%. In high-stakes grocery customer service (food safety, payment disputes, legal claims), **catching 2 additional liability events at the cost of 3 extra agent reviews is a deliberate, cautious safety-first trade-off, not an unalloyed accuracy win**.

### Historical Retrieval & Evidence Grounding
Evaluated over 16,365 non-golden historical documents with strict per-query temporal cutoff ($t_{doc} \le t_{query}$) and self-exclusion:

| Retriever | Top-1 Accuracy (MRR) | Accepted Evidence Rate | Primary Rejection Reason |
|---|:---:|:---:|---|
| **Lexical (BM25Okapi)** | 0.412 | 37.5% | Vocabulary mismatch on colloquial complaints |
| **Semantic (`all-MiniLM-L6-v2`)** | **0.683** | **52.5%** | Weak semantic match ($< 0.55$) |

---

## 6. Evidence-Grounded Reply Generation & Guardrails

### Safeguards Implemented

1. **Pre-Prompt PII & Name Scrubbing**:
   - Customer handles redacted to `[CUSTOMER]`, order references to `[ORDER_REF]`, emails, phones, postcodes redacted.
   - Named greetings (*"Hi Ellie,"*, *"Hello David,"*) normalized to generic *"Hi,"* or *"Hello,"*.
   - Colleague signatures (*"- Callum"*, *"TY Mike"*) normalized to `"- Team"`.
   - *Honest Disclosure*: Regex-based redaction removed all names detected by automated checks; unusual names may still require human review.

2. **Sentiment Conflict Interception (Heuristic Tone Guard)**:
   - When semantic retrieval matches a negative/apologetic resolution for a positive tweet (compliment or humor), the system automatically detects the conflict and falls back to an aligned template reply.
   - Example caught: CID `1742132` (9 brownies in a pack joke) retrieved a Caesar salad defect refund; intercepted and fell back to standard product feedback template.

3. **Strict Structured Output Schema**:
   - Model must output strict JSON: `{"status": "draft", "reply": "...", "grounded_evidence_id": "..."}` or `{"status": "escalate", ...}`.
   - Any unstructured text or markdown error fails safe to human escalation.

---

## 7. Evaluation, LLM Judge Validation & Human Comparison

### Evaluation Funnel (Dev Subset, $N=20$)
- **Total queries evaluated**: 20.
- **Policy Escalated (bypassed drafting)**: 9 (45.0%) — **0 LLM tokens consumed**.
- **Eligible for reply generation**: 11 (55.0%).
- **Pairwise comparable cases**: Strictly the 11 `RESPOND` queries where both systems produced replies.

### Blinded Pairwise Judging & Position Bias
- Evaluates Candidate A vs Candidate B with model identities stripped.
- Two-pass execution: Pass 1 (Forward: Gen vs Tmpl) and Pass 2 (Reversed: Tmpl vs Gen).
- **Position-Bias Metrics (Dev run)**:
  - Position-A win rate: 45.45% (healthy balance around 50%).
  - Swap consistency: 100.00% (judge maintained consistent preference regardless of presentation order).
  - Flip rate: 0.00% (zero order-dependent flip anomalies).

### Truly Blinded Human Review Protocol
- Human review CSV ([`data/annotation/human_pairwise_comparison.csv`](file:///c:/Users/aadhi/OneDrive/Desktop/hiver-tesco/data/annotation/human_pairwise_comparison.csv)) presents only `sanitized_query`, `candidate_a`, `candidate_b`, with candidate order randomized.
- `primary_intent`, `escalation_needed`, and `split` labels are strictly withheld in a separate internal mapping file to prevent reviewer anchoring bias.
- **Single-Reviewer Disclosure**: All human labels were provided by a single evaluator. Only raw percentage agreement is reported; no Cohen's Kappa or inter-annotator reliability is claimed.

---

## 8. Failure Analysis, Risks & Unresolved Boundaries

### 1. Semantic Retrieval Tone Mismatch
Even with high cosine similarity ($\ge 0.65$), retrieval can match on shared keywords (*"salad"*, *"brownies"*) while pulling an apology for a humorous comment. The sentiment-conflict check handles high-discrepancy praise/complaint conflicts, but subtle sarcasm remains a risk for future exploration.

### 2. Lexical vs Semantic Intent Ambiguity
Short queries like *"Tesco flitwick"* or *"delivery saver"* lack grammatical context, leading to lower classifier confidence ($< 0.18$). The policy intentionally catches these under Rule 4.2 (`low_model_confidence`) and routes them to human operators.

### 3. Operator Overhead vs. Customer Safety
By routing payment disputes and delivery tracking with repetition to human queues, the system incurs higher operational costs (7 false alarms per 40 Dev queries). In enterprise grocery retail, this overhead is preferable to hallucinating refund confirmations or incorrect order dispatch statuses.

### 4. What Remains Unsafe / Unproven
- **Unusual PII Formats**: Custom foreign postcodes or non-standard order numbers may bypass automated regex filters.
- **Real-Time Dynamic State**: The system cannot verify whether a customer's specific delivery van is running late today.

---

## 9. Reproducibility Instructions (One-Command Run)

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Aadhithya-T/hiver-tesco.git
cd hiver-tesco

python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Fast 100% Offline Reproduction (Recommended)
Runs the entire pipeline end-to-end using mock providers and cached artifacts with zero API costs, zero network calls, and hermetic guarantees:
```powershell
python scripts/run_all_phases.py --offline --skip-extraction --run-tests
```

### 3. Live Model Execution (Optional)
To run with live models, copy `.env.example` to `.env` and set your key:
```bash
cp .env.example .env
# Edit .env with GEMINI_API_KEY or OPENAI_API_KEY
```
Then run:
```powershell
# Using Google Gemini (gemini-3.5-flash):
python scripts/run_all_phases.py --provider gemini --skip-extraction

# Or step-by-step Phase 7 CLI:
python scripts/run_phase7_evaluation.py --stage dev --provider gemini
python scripts/run_phase7_evaluation.py --stage test --provider gemini
python scripts/run_phase7_evaluation.py --stage finalize
```

---

## 10. Test Suite & Verification

The repository contains **99 hermetic unit and integration tests** executing with zero network calls and zero external API dependencies:

```powershell
python -m pytest tests/ -v
```

### Test Coverage Highlights:
- `test_loader.py` & `test_direction.py`: CSV ingestion, edge-case headers, inbound/outbound ownership.
- `test_anomalies.py` & `test_graph.py`: Cycle detection, cross-brand mentions, missing parent tree assembly.
- `test_golden_freeze.py` & `test_annotation_validation.py`: Manifest checksums, 8-class taxonomy validation, PII leakage detection.
- `test_baselines.py`: Leakage-safe query extractor, majority class, TF-IDF + LogReg classifier.
- `test_retrieval.py`: Historical cutoff ($t_{doc} \le t_{query}$), golden-set exclusion, evidence filtering.
- `test_policy.py`: Priority hierarchy (Priority 1–5), risk rules, safe respond conditions.
- `test_generation.py`: PII & customer name scrubbing, greeting normalization, sentiment conflict fallbacks, structured output parsing, caching.
- `test_evaluation.py`: Automated checks, stratified sampling, blinded pairwise judge swap mechanics, position-bias metrics, Wilson score CIs, escalation review tables.

---

## 11. Project Directory & Artifact Map

```
hiver-tesco/
├── .env.example               # Template for optional API credentials
├── .gitignore                 # Excludes raw CSVs, .env, outputs/, and caches
├── pyproject.toml             # Build configuration & dependencies
├── requirements.txt           # Pinned dependencies
├── README.md                  # Comprehensive engineering report (this file)
├── tesco_tweets.csv           # Raw TWCS Tesco dataset (65,308 tweets)
│
├── data/
│   ├── annotation/
│   │   ├── golden_candidates_200.csv       # 200 candidate sampling sheet
│   │   └── human_pairwise_comparison.csv   # Truly blinded reviewer CSV
│   └── golden/
│       ├── golden_set_v1.0.csv             # 200 labeled golden conversations
│       ├── golden_set_manifest.json        # Integrity checksums and distribution
│       └── splits/                         # train_ids, dev_ids, test_ids
│
├── outputs/                                # (Generated, ignored from git)
│   ├── conversations.jsonl                 # 16,565 reconstructed threads
│   ├── audit_report.json / .csv            # Threading and anomaly metrics
│   ├── retrieval/
│   │   ├── historical_corpus.jsonl         # 16,365 non-golden resolution docs
│   │   └── corpus_embeddings.npy           # Precomputed dense embeddings
│   ├── policy/                             # Policy evaluation decisions & tables
│   ├── generation/                         # Grounded generation audit logs & cache
│   └── evaluation/                         # Automated checks, judge JSONs, reports
│
├── src/hiver_tesco/
│   ├── config.py, models.py                # Core constants and data types
│   ├── loader.py, direction.py             # CSV streaming and role tagging
│   ├── graph.py, anomalies.py, audit.py    # Thread reconstruction & anomaly scanner
│   ├── baselines/                          # Intent, escalation & template models
│   ├── retrieval/                          # BM25, semantic retriever, PII cleaner
│   ├── policy/                             # Deterministic policy engine & risk rules
│   ├── generation/                         # Grounded generator, providers, cache
│   └── evaluation/                         # Automated audits, judge, human metrics
│
├── scripts/
│   ├── run_all_phases.py                   # Unified end-to-end pipeline driver
│   ├── run_pipeline.py                     # Phase 1 extraction
│   ├── freeze_golden_set.py                # Phase 2 golden manifest freeze
│   ├── run_baselines.py                    # Phase 3 baselines
│   ├── build_retrieval_corpus.py           # Phase 4 corpus builder
│   ├── evaluate_dev_retrieval.py           # Phase 4 retrieval evaluator
│   ├── evaluate_policy.py                  # Phase 5 policy evaluator
│   ├── generate_replies.py                 # Phase 6 generation runner
│   └── run_phase7_evaluation.py            # Phase 7 three-stage evaluation CLI
│
└── tests/                                  # 99 hermetic pytest unit/integration tests
```
