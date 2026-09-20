# Unified Risk Intelligence – FinServ

**An intelligent complaint-management agent for financial services.**
Customers describe a problem in a chat window; the system anonymises PII, understands intent and sentiment, classifies the complaint, scores its risk, retrieves a resolution from a knowledge base (RAG), decides whether to escalate to a human, and stores everything for an admin analytics dashboard, including churn-risk analytics and model-evaluation views.

Built as an MTech dissertation prototype. Everything runs locally (SQLite + ChromaDB); an LLM (Anthropic Claude) is **optional**.

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Using the app](#using-the-app)
- [REST API](#rest-api)
- [Models, training and evaluation](#models-training-and-evaluation)
- [Configuration](#configuration)
- [What is *not* in the repository](#what-is-not-in-the-repository)
- [Prototype caveats](#prototype-caveats)

---

## Features

**Customer chat assistant**
- Register, track, escalate and close complaints in natural language
- Answers general "how do I…" questions straight from the knowledge base, without opening a complaint
- Cites additional related KB articles in expandable panels

**Analytics for admins**
- KPIs, category / priority / sentiment distributions, risk by tier and category, complaint volume over time
- Customer **churn-risk** scoring (10 features) with model comparison and feature-importance vs. hand-set weights
- **Explainability**: top predictive terms per category, per-complaint term contributions, per-complaint risk-factor breakdown
- Filterable complaint log and a per-complaint detail inspector

**System evaluation page**
- Accuracy / precision / recall / F1 / confusion matrices for intent, category (3 model variants), sentiment, RAG retrieval and risk scoring

**Security & privacy**
- PII masked (Microsoft Presidio, with a regex fallback) before the NLP, classification, risk-scoring and retrieval stages (see [caveats](#prototype-caveats) for what is still stored raw)
- bcrypt password hashing, JWT auth for the API, role-based page access (`customer` / `admin`)

---

## How it works

```
                    ┌────────────────────────────────────────────────────────┐
 customer message → │ 1. PII anonymisation   Presidio → regex fallback       │
                    │ 2. NLP                 intent · entities · sentiment   │
                    │ 3. Classification      9 complaint categories          │
                    │ 4. Risk scoring        6-factor score → priority       │
                    │ 5. RAG                 ChromaDB semantic KB retrieval  │
                    │ 6. Escalation check    rules → human agent?            │
                    │ 7. Response            KB template  (or Claude)        │
                    │ 8. Persist             SQLite                          │
                    └────────────────────────────────────────────────────────┘
```

| Stage | Implementation (`src/…`) | Method |
|---|---|---|
| PII | `security/pii_handler.py` | Presidio analyzer/anonymizer; regex patterns (cards, phone, e-mail, PAN, Aadhaar…) if Presidio fails |
| Intent | `nlp/intent_detector.py` | Regex rules first (7 intents), zero-shot BART-MNLI fallback |
| Entities | `nlp/entity_extractor.py` | Regex (amount, txn ID, dates, card/account type, complaint ID) + spaCy NER |
| Sentiment | `nlp/sentiment_analyzer.py` | `ProsusAI/finbert` → polarity in [-1, 1] |
| Category | `classification/complaint_classifier.py` | Pattern rules → ensembled with trained **TF-IDF + Logistic Regression** → zero-shot fallback |
| Risk | `classification/risk_scorer.py` | Weighted 6-factor score (see below); priority from rules, or from a trained GB/RF model when `models/risk_scorer.pkl` exists |
| RAG | `rag/` | `all-MiniLM-L6-v2` embeddings in ChromaDB (cosine), category-filtered top-k, 30-article KB |
| Escalation | `chatbot/escalation_engine.py` | Mandatory categories (Fraud), risk ≥ 0.75, max auto-attempts, prior escalation, customer request |
| Response | `rag/response_generator.py` | Formats the top KB article; optionally Claude when `USE_LLM=true` |
| Churn | `analytics/churn_analyzer.py` | 10 complaint-history features → weighted score + RF/GB tier classifier |

**Risk score** = `0.30·severity + 0.20·sentiment + 0.15·escalations + 0.15·unresolved + 0.10·frequency + 0.10·customer-value`
Rule thresholds: **Critical ≥ 0.75 · High ≥ 0.50 · Medium ≥ 0.25 · Low** below that (all configurable in `config.yaml`).

**Complaint categories:** Transaction Dispute · Account Access Issue · Billing Concern · Card Services · Fraud & Unauthorized Charges · Reward & Points Issue · Customer Service Complaint · Loan & Credit Issue · General Inquiry

---

## Tech stack

Python · Streamlit · FastAPI · Hugging Face Transformers (FinBERT, BART-MNLI, DistilBERT) · scikit-learn · spaCy · Presidio · ChromaDB + sentence-transformers · SQLite · Plotly · bcrypt + python-jose · Anthropic SDK (optional)

---

## Project structure

```
.
├── app.py                     # Streamlit entry point: login/register + role-based navigation
├── pages/
│   ├── chat.py                # Customer chat assistant
│   ├── analytics.py           # Admin analytics, churn & explainability dashboard
│   └── evaluation.py          # Admin model-evaluation dashboard
├── src/
│   ├── config.py              # Loads config.yaml + .env
│   ├── chatbot/               # agent (orchestrator), conversation manager, escalation engine
│   ├── nlp/                   # intent, entities, sentiment, pipeline
│   ├── classification/        # category classifier, risk scorer
│   ├── rag/                   # KB ingestion, retriever, response generator
│   ├── analytics/             # churn analyzer
│   ├── security/              # auth (bcrypt/JWT), PII handler
│   ├── data/                  # Pydantic models + SQLite access layer
│   └── api/main.py            # FastAPI REST interface
├── scripts/
│   ├── simulate_complaints.py     # ┐ seed the DB with synthetic complaints
│   ├── simulate_complaints_2.py   # │ (hard-coded demo customers)
│   ├── run_prompts.py             # │
│   ├── run_high_risk.py           # ┘ pushes two demo customers into the High churn tier
│   ├── train_category_classifier.py   # TF-IDF + LR   → models/category_classifier.pkl
│   ├── train_risk_scorer.py           # RF vs GB      → models/risk_scorer.pkl
│   ├── train_churn_advanced.py        # RF vs GB LOOCV → models/churn_classifier.pkl
│   ├── train_distilbert_category.py   # fine-tune DistilBERT (evaluation comparison only)
│   ├── evaluate_system.py             # writes data/eval/results.json
│   └── generate_arch_pptx.py          # builds an editable architecture slide
├── data/
│   ├── knowledge_base/financial_kb.json     # 30 KB articles
│   ├── eval/                                # hand-labelled test sets + results.json
│   └── supplementary_train_categories.json  # 135 extra labelled training samples
├── models/                    # trained artefacts (*.pkl committed; distilbert_category/ is git-ignored)
├── tests/                     # pytest: database, NLP components, risk scorer
├── demo.ipynb                 # step-by-step walkthrough of the whole pipeline
├── config.yaml                # thresholds, weights, model names, LLM settings
├── setup.py                   # ONE-TIME init script (DB + KB ingest) – not a packaging file
├── requirements.txt           # runtime dependencies
└── requirements-dev.txt       # + pytest, python-pptx, jupyter, matplotlib
```

---

## Getting started

### Prerequisites
- Python 3.10+ recommended (developed on 3.11)
- ~5 GB free disk and internet on first run (model downloads, see below)
- A GPU is **not** required; all inference runs on CPU

### 1. Install

```bash
git clone <your-repo-url>
cd "Unified Risk Intelligence - FinServ"

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # add requirements-dev.txt for tests / notebook
python -m spacy download en_core_web_sm
```

### 2. Configure secrets

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `JWT_SECRET_KEY` | Signs API tokens. **Set a long random value**, e.g. `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ANTHROPIC_API_KEY` | Only needed if `USE_LLM=true` |
| `USE_LLM` | `false` (default) → replies come from the top KB article. `true` → Claude writes the reply; any API error silently falls back to the template |

### 3. Initialise

```bash
python setup.py      # creates data/complaints.db and ingests the KB into ChromaDB
```

### 4. Run

```bash
streamlit run app.py                    # web UI  → http://localhost:8501
uvicorn src.api.main:app --reload       # optional REST API → http://localhost:8000/docs
```

> Run everything **from the repository root**: several paths (`models/`, `data/eval/`) are relative.

### First-run downloads

| Asset | Size (approx.) | When |
|---|---|---|
| `ProsusAI/finbert` | ~440 MB | first chat-page load |
| `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | `setup.py` / first retrieval |
| spaCy `en_core_web_sm` | ~15 MB | installed above (also auto-downloaded on demand) |
| `facebook/bart-large-mnli` | ~1.6 GB | only when rules and the trained model can't decide (fallback) |

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

---

## Using the app

The Streamlit app has a **Login / Register / Forgot Password** screen. After login the sidebar depends on the role:

| Role | Pages |
|---|---|
| `customer` | 💬 Chat Assistant |
| `admin` | 📊 Analytics Dashboard · 🧪 System Evaluation |

A fresh database has no users, so **register at least one `customer` and one `admin`** on the Register tab.

Example chat messages:

```
"I was charged Rs.2,500 twice for the same purchase"       → registers a complaint, returns a reference (CMP-XXXXXXXX)
"There is an unauthorized transaction of Rs.45,000 on my card" → Fraud → auto-escalated to a senior agent
"What is the status of CMP-1A2B3C4D?"                      → tracks a complaint
"How do I redeem my reward points for a voucher?"          → answered from the KB, no complaint created
"I want to speak to a manager"                             → escalates the last complaint
"Thanks, the issue is resolved"                            → closes the last complaint
```

---

## REST API

Interactive docs at `/docs` once the API is running. All `/complaints*` routes need `Authorization: Bearer <token>`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Create a customer account |
| `POST` | `/auth/login` | Returns a JWT (60 min) and the customer profile |
| `POST` | `/complaints/register` | Submit a complaint → category, priority, risk score, response |
| `GET` | `/complaints` | List the caller's complaints |
| `GET` | `/complaints/{id}` | Track one complaint |
| `POST` | `/complaints/{id}/escalate` | Escalate to a human agent |
| `POST` | `/complaints/{id}/resolve` | Mark as resolved |
| `GET` | `/health` | Liveness check |

---

## Models, training and evaluation

### Rebuilding the local data and models from scratch

The database, vector store and DistilBERT weights are not in git (see [below](#what-is-not-in-the-repository)). To reproduce them, in this order:

```bash
python setup.py                                  # 1. DB + vector store

# 2. Seed synthetic complaints (also register users in the UI first if you want names in the churn tables;
#    the seed scripts use hard-coded customer IDs – edit them to match your users)
python scripts/simulate_complaints.py
python scripts/simulate_complaints_2.py
python scripts/run_prompts.py
python scripts/run_high_risk.py

# 3. Train models (they learn from the seeded DB)
python scripts/train_category_classifier.py      # → models/category_classifier.pkl
python scripts/train_risk_scorer.py              # → models/risk_scorer.pkl
python scripts/train_churn_advanced.py           # → models/churn_classifier.pkl
python scripts/train_distilbert_category.py      # optional, ~20–30 min on CPU

# 4. Evaluate
python scripts/evaluate_system.py                # → data/eval/results.json (shown in the admin UI)
```

### Latest results (`data/eval/results.json`)

| Component | Method | Test set | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Intent detection | Rules + zero-shot fallback | 50 | 88.0% | 88.1% |
| Category | Zero-shot BART-MNLI (baseline) | 45 | 60.0% | 51.8% |
| Category | **TF-IDF + Logistic Regression** (live) | 45 | **84.4%** | 84.9% |
| Category | Fine-tuned DistilBERT (comparison only) | 45 | 82.2% | 83.8% |
| Sentiment | FinBERT | 30 | 100% | 100% |
| RAG | Precision@1 / @3 (keyword relevance) | 10 queries | 100% / 100% | – |
| Risk priority | Rule-based scorer (sanity cases) | 8 | 62.5% | – |
| Risk priority | Trained GB model (same sanity cases) | 8 | 25.0% | – |

These test sets are small and hand-labelled; treat the numbers as indicative rather than definitive.

### Category classifier tiers
1. **Pattern rules** decide when several patterns match.
2. If only one pattern hits, the **trained TF-IDF + LR** model is consulted and wins if it is more confident.
3. With no pattern hit, the trained model decides; if it is unavailable, **zero-shot BART-MNLI** does.
4. Final fallback: `General Inquiry`.

The fine-tuned DistilBERT model is trained and evaluated for the dissertation comparison but is **not** loaded by the live pipeline.

---

## Configuration

`config.yaml` holds the tunable settings. Values that are **actively read** by the code:

| Section | Keys |
|---|---|
| `rag` | `collection_name`, `embedding_model`, `top_k_retrieval` |
| `nlp` | `intent_confidence_threshold` |
| `risk_scoring` | `weights`, `thresholds` |
| `escalation` | `risk_score_threshold`, `max_auto_resolution_attempts`, `escalation_categories` |
| `security` | `jwt_algorithm`, `access_token_expire_minutes`, `pii_entities` |
| `llm` | `model`, `max_tokens`, `temperature` |

Other keys in the file (`app.*`, `database.url`, `rag.chroma_db_path`, `rag.chunk_size`, `rag.chunk_overlap`, `nlp.sentiment_model`, `nlp.spacy_model`, `classification.*`, `security.jwt_secret_env`, `llm.provider`) are currently **not read**. Paths and model names for those are set in code (`data/complaints.db`, `./chroma_db`, `ProsusAI/finbert`, `en_core_web_sm`), so editing them has no effect yet.

---

## What is *not* in the repository

Ignored via `.gitignore` because they are generated, sensitive, or too large:

| Path | Why | How to get it back |
|---|---|---|
| `.env` | secrets | copy `.env.example` |
| `data/complaints.db` | user accounts (password hashes) + complaint text | `python setup.py` + seed scripts |
| `chroma_db/` | generated vector store | `python setup.py` |
| `models/distilbert_category/` | ~270 MB, over GitHub's 100 MB file limit | `python scripts/train_distilbert_category.py` |

If you want the demo data to be reproducible from a fresh clone, consider exporting an anonymised CSV of the training texts/labels, or tracking the large weights with [Git LFS](https://git-lfs.com).

---

## Prototype caveats

This is a research/demo system and **not production-ready**. Known points to address before real use:

- **Raw text is stored**: the `complaints` table keeps the original `raw_text` next to `anonymized_text`, and intent detection runs on the un-masked message. Drop `raw_text` (or encrypt it) if PII must never be persisted.
- **Open admin registration**: the Register tab lets anyone choose the `admin` role. Restrict or remove that selector.
- **Weak recovery flow**: password reset is verified only by account number (customers) or full name (admins).
- **API ownership check**: `POST /complaints/{id}/resolve` does not verify that the complaint belongs to the caller.
- **JWT fallback secret**: if `JWT_SECRET_KEY` is unset, a hard-coded default is used.
- **Priority vs. score**: when `models/risk_scorer.pkl` exists, the ML model chooses the *priority* while the displayed *risk score* (and escalation threshold) still come from the weighted formula, so the two can disagree. The model is trained on rule-generated pseudo-labels, and scored lower than the rules on the hand-written sanity cases above.
- **Churn model**: labels are derived from the same rule-based score over a small number of customers, so it demonstrates the approach rather than predictive validity.
- **Model files**: `.pkl` files are loaded with `joblib`; only load ones you trust, and keep `scikit-learn` compatible with the version that trained them.
- **Setup script name**: `setup.py` is a one-off initialiser, not a `setuptools` file. Don't run `pip install .` on this repo.
- No `LICENSE` file yet; add one before publishing.
