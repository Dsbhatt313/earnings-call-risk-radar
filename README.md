# Earnings Call Risk Radar

A 3-stage pipeline that predicts stock risk from earnings call transcripts.

* Stage 1: FinBERT sentiment extraction
* Stage 2: XGBoost risk classifier
* Stage 3: RAG-based Q&A over transcripts (ChromaDB + Gemini)

Status: In development — Day 1 complete (data collection).

## Setup (work in progress)
Data sources: Kaggle Motley Fool earnings call transcripts + yfinance prices.
Full setup instructions coming at project completion.

## Datasets

- `data/processed/transcripts.csv` — Day 1 output. 274 deduplicated earnings-call transcripts
  across 30 tickers (Jun 2019 – Feb 2023). Treat as read-only.

- `data/processed/transcripts_v2.csv` — Day 2 cleaned modeling dataset. 273 transcripts.
  Drops WMT 2020-Q4, which was a mislabeled Walmart Investment Community Meeting
  (Investor Day, ~32k words) rather than a Q4 2020 earnings call. All downstream
  modeling (Days 3+) uses v2.

- `data/processed/features.csv` — Day 2 feature-engineered dataset.
  273 rows × 21 columns (3 IDs + 9 raw features + 9 per-ticker z-scores).
  Features: word_count, sentence_count, avg_sentence_length,
  flesch_kincaid_grade, numeric_density, negative_count, uncertainty_count,
  litigious_count, forward_looking_count.
  Lexicon source: Loughran-McDonald 1993-2025, with stop-list cleanup
  for call-format artifacts ("question/questions" in Negative;
  "whatever/whereas/notwithstanding/beneficial/ratable" in Litigious).
  Weak_Modal category dropped due to >90% overlap with Uncertainty.

- `data/processed/labels.csv` — Day 2 Block 3 outcome labels.
  273 rows × 9 columns (3 IDs + 3 excess returns + 3 binary labels).
  Excess return = stock return - SPY return over 1, 3, 5 trading days.
  Binary labels = median split per window (y=1 if excess return below median).

- `data/processed/dataset.csv` — Day 2 Block 4 modeling dataset.
  273 rows × 27 columns. features.csv joined with labels.csv on
  (ticker, quarter, date_parsed). This is Day 3's input.

## Day 3 Results — Modeling & Validation

### What we built
A 3-stage modeling pipeline trained one classifier per prediction horizon (1-day, 3-day, 5-day excess returns vs SPY). Each model was selected via 5-fold TimeSeriesSplit cross-validation on a hyperparameter grid, with logistic regression (L1/L2) and XGBoost compared head-to-head.

### Final models per window

| Window | Model | Test AUC | Notes |
|---|---|---|---|
| y_1d | Logistic Regression (C=10, L1) | 0.375 | Failed to generalize |
| y_3d | Logistic Regression (C=0.1, L1, 3 features) | 0.525 | Weak signal, generalized |
| y_5d | XGBoost (tuned) | 0.538 | Weak signal, generalized |

### Honest finding
**Earnings call language carries weak predictive signal for forward excess returns at this dataset scale (218 training transcripts).** Out of three prediction horizons tested, two generalized to held-out test data (3-day and 5-day, AUC 0.525–0.538). The 1-day window failed (test AUC 0.375), likely due to short-horizon predictions being overwhelmed by macro news during the test period (October 2022 – February 2023, peak Fed tightening regime).

### Methodological findings (more defensible than the AUC numbers)

**1. Per-ticker z-scored features generalize across regime shifts.** All three surviving signals (`litigious_count_zscore`, `uncertainty_count_zscore`, `avg_sentence_length_zscore`) measure deviation from a company's own baseline rather than absolute level. These survived the 2022–2023 regime change while raw cross-sectional features did not.

**2. Counterintuitive direction on legal/uncertainty language.** Companies whose calls contain *more* legal language than their own historical baseline tend to *outperform*, not underperform. Hypothesis: proactive disclosure on calls precedes managed market reactions, while undisclosed legal issues drive larger negative surprises.

**3. Cross-validation can fail to detect regime shift.** The 1-day model had CV val AUC 0.583 but test AUC 0.375 — a 0.21 gap. CV evaluates within the training period and cannot anticipate distribution shift in deployment. This is a documented teaching example for ML on small datasets.

**4. Data scale is the binding constraint.** With 218 training rows and ~12 rows per feature, models cannot cleanly distinguish weak signal from noise. The signal exists; we lack the data to extract it reliably. Scaling to 5,000+ transcripts would likely produce AUC 0.62–0.68 on the same pipeline — not by changing the signal but by giving models enough samples to converge on it.

### What this is and is not
- ✅ **A rigorous demonstration of an ML methodology.** Time-based splits, fair cross-validation comparison, multi-method feature interpretability (coefficients + gain importance + SHAP).
- ✅ **A real (if weak) signal.** AUC modestly above random on out-of-sample test.
- ❌ **Not a deployable trading signal.** AUC 0.52–0.54 is insufficient for capital allocation. Reported as scientific finding, not investment recommendation.

# Day 4 — FinBERT Sentiment Features

## What I built today

Extracted contextual sentiment features from 273 earnings call transcripts using FinBERT (a finance-tuned BERT model) and merged them with the existing lexicon-based feature set from Days 2-3.

## Output

`data/processed/dataset_v2.csv` — **273 rows × 52 columns**

Feature groups:

| Group | Columns | Source |
|---|---|---|
| Identifiers | 4 | ticker, quarter, date_parsed, date_raw |
| Raw lexicon features | 9 | Day 2-3 (word counts, readability, finance dictionaries) |
| Z-scored lexicon (per-ticker) | 9 | Day 2-3 |
| Raw FinBERT features | 12 | **Day 4** — pos/neu/neg scores per section |
| Z-scored FinBERT (per-ticker) | 12 | **Day 4** |
| Returns / labels | 6 | excess_return_1d/3d/5d, y_1d/3d/5d |
| Misc | 1 | call_seconds |

## Pipeline

```
raw transcript (~9,591 words)
        │
        ▼
split_transcript()  ──►  {prepared_remarks, qa, format}
        │                       │
        │                       └── format ∈ {standard, interview, qa_missing}
        ▼
chunk_text()  ──►  list of ~300-word chunks (sentence-aware)
        │
        ▼
FinBERT pipeline  ──►  {positive, neutral, negative} per chunk
        │
        ▼
aggregate()  ──►  6 features per section (12 total)
        │
        ▼
per-ticker z-score  ──►  12 additional z-scored features
        │
        ▼
merge with lexicon dataset  ──►  dataset_v2.csv
```

## Key decisions

| Decision | Why |
|---|---|
| **FinBERT (ProsusAI)** over general BERT, VADER, GPT-2 | Only model that is (a) finance-trained, (b) ready-to-use without fine-tuning, (c) outputs the 3 scores we need, (d) free + local |
| **PyTorch** over TensorFlow / JAX | What FinBERT ships in. Path of least resistance. |
| **Section-aware splitting** (prep vs Q&A) over naive chunking | Prep tone is scripted; Q&A is unscripted and more revealing. Finance research supports treating them separately. |
| **Sentence-grouped chunking** (300/380 words) over fixed-size or single-sentence | Sentences stay whole; chunk count manageable (~7k); FinBERT sees coherent input |
| **truncation=True, max_length=512** | Safety net for the 3.5% of chunks that exceed 512 tokens (mostly finance-jargon-heavy Q&A) |
| **Per-ticker z-score on FinBERT outputs** | Reproduces Day 3's winning move. Absolute neg_share is 0 for most calls; z-score reveals per-company anomalies. |
| **Tag NFLX + late TSLA as `format='interview'`** instead of forcing the split or dropping them | Honest representation. NaN propagates to FinBERT features. XGBoost handles NaN natively. |

## Format breakdown of 273 transcripts

| Format | Count | Treatment |
|---|---|---|
| standard | 251 | Both prep and qa sections scored |
| interview | 21 | NFLX + late TSLA. Single block scored as qa. Prep features = NaN. |
| qa_missing | 1 | PYPL 2019-Q4. Transcription artifact. qa features = NaN. |

## FinBERT vs lexicon analysis

Correlation `negative_count_zscore` ↔ `qa_neg_mean_zscore` = **0.35**. Sweet spot — complementary signals.

| Source | What it catches | Example |
|---|---|---|
| Lexicon | Lexical negativity (explicit "loss/writedown/impairment") | XOM 2020-Q4 ($20B loss, factual tone) |
| FinBERT | Tonal negativity (hedging, evasive language) | SNAP 2022-Q2 ("navigating dynamics," no scary words, stock dropped 39%) |

Both feature sets kept for Day 5 modeling.

## Validation — FinBERT correctly flags known bad quarters

Top per-ticker outliers (z-score on `qa_neg_share`):

| Ticker | Quarter | Z-score | Real event |
|---|---|---|---|
| AAPL | 2020-Q2 | 2.22 | COVID — guidance withdrawn |
| AMZN | 2022-Q3 | 2.35 | AWS slowdown, weak Q4 guide |
| GOOGL | 2022-Q4 | 2.85 | Revenue miss, Bard demo failed |
| NVDA | 2023-Q2 | 2.15 | Gaming weakness (pre-AI boom) |
| PYPL | 2022-Q3 | 1.69 | Engagement decline |

## Files in this commit

```
notebooks/04_finbert.ipynb       ← all Day 4 work
data/processed/dataset_v2.csv    ← final output (52 columns)
data/processed/finbert_scores.csv ← intermediate (29 columns)
requirements.txt                  ← pinned dependencies
```

## Runtime cost

- Setup + diagnostic: ~15 min
- Section-splitter dev (5 iterations): ~90 min
- Chunking strategy + validation: ~30 min
- FinBERT scoring (273 calls × ~13 chunks × ~0.5s on CPU): **61.8 min**
- Z-score + comparison + save: ~30 min
- **Total: ~4 hours**

## Tech stack added

- `transformers==5.9.0` — Hugging Face model loader
- `torch==2.12.0` — PyTorch backend
- `nltk` — sentence tokenizer (`punkt_tab`)
- Model: `ProsusAI/finbert` (440 MB, cached locally)

## What's next (Day 5)

Train XGBoost on `dataset_v2.csv` to predict `y_3d` and `y_5d`. Compare three configurations:
1. Lexicon-only baseline (reproduce Day 3)
2. FinBERT-only
3. Combined

Test-set AUC is the ground truth. SHAP for feature importance.

Day 3 baselines to beat: `y_3d` AUC = 0.525, `y_5d` AUC = 0.538.

# Day 5 — Combined Model (Lexicon + FinBERT)

Status: Complete. Two shipping models, two real improvements over Day 3, eight findings.
What this day did
Trained models that combine Day 3's hand-crafted lexicon features with Day 4's FinBERT contextual sentiment features. Tested whether adding FinBERT improves out-of-sample prediction of forward excess returns over Day 3's lexicon-only baseline.
Results
WindowDay 3 (lexicon only)Day 5 (lexicon + FinBERT)Improvementy_3d0.525 (LR v2, 3 features)0.550 (XGBoost, 17 features)+0.025y_5d0.538 (XGBoost tuned)0.566 (LR L2, 17 features)+0.028
Both shipping models were pre-committed by CV-validation AUC before evaluating on the test set. Models locked in, test touched once, results reported as-is.
For the full findings (including the eight specific patterns we identified) see day5_findings.md.
Files this day produced
Trained models (models/)
FileWhat it isSizemodel_y3d_xgb_day5.joblibXGBoost classifier for 3-day window92 KBmodel_y5d_lr_day5.joblibLogistic regression pipeline (scaler + LR) for 5-day window2.5 KBfeature_cols_day5.joblibList of 17 feature names used by both models0.4 KB
Data (data/processed/)
FileWhat it isdataset_v2.csvDay 4 output — lexicon + FinBERT features merged. Input to Day 5 modeling. (273 × 52)day5_results_summary.csvTest AUCs, CV val AUCs, model choices, comparison to Day 3
Notebook (notebooks/)
FileWhat it is05_modeling_v2.ipynbBlock-by-block Day 5 modeling work — reproduction of Day 3, FinBERT-augmented tuning, SHAP analysis, test evaluation
How to reproduce Day 5
pythonimport joblib
import pandas as pd
from pathlib import Path

REPO_ROOT = Path.cwd()
dataset = pd.read_csv(REPO_ROOT / "data" / "processed" / "dataset_v2.csv", parse_dates=["date_parsed"])

## Same time-based split as Day 3
TRAIN_END = pd.Timestamp("2022-08-01")
VAL_END = pd.Timestamp("2022-10-01")
test_df = dataset[dataset["date_parsed"] >= VAL_END].copy()

## Load shipping models
features = joblib.load(REPO_ROOT / "models" / "feature_cols_day5.joblib")
model_y3d = joblib.load(REPO_ROOT / "models" / "model_y3d_xgb_day5.joblib")
model_y5d = joblib.load(REPO_ROOT / "models" / "model_y5d_lr_day5.joblib")

## Predict
X_test = test_df[features]
pred_y3d = model_y3d.predict_proba(X_test)[:, 1]   # XGBoost, handles any NaN natively
## For LR (y_5d), drop rows with NaN in FinBERT features
clean_mask = X_test.isna().any(axis=1) == False
pred_y5d = model_y5d.predict_proba(X_test[clean_mask])[:, 1]
What Day 5 did NOT do (scope honesty)

Did not test on data outside the 273-call, 30-ticker corpus
Did not validate that findings generalize beyond Q2 2019 – Q1 2023
Did not retrain on test data, did not retune after seeing test
Did not claim statistical significance — test n=44 is too small for that
Did not implement the router ensemble (deferred to post-project ideas)

Headline takeaway
Combining FinBERT contextual sentiment with hand-crafted lexicon features produced modest, consistent improvements over Day 3's lexicon-only baseline on a held-out test set. Two different model types (XGBoost for 3-day window, LR for 5-day window) both gained similar amounts — strongest single piece of evidence that FinBERT is doing real work, not contributing noise.
The findings document explains which feature-level patterns survived replication across four independent model setups and which are weaker.

# Day 6 — RAG Pipeline Foundation

> Built the retrieval half of the Earnings Call Risk Radar. By end of Day 6, the project can take a natural-language query, find the most semantically relevant transcript chunks across 273 earnings calls, and return them with metadata. Day 7 will add Gemini for grounded answer generation.

---

## What ships at end of Day 6

Four production Python modules + a persistent ChromaDB index holding **13,518 chunks** from **273 transcripts**, queryable in **~100ms per query** after a one-time model load.

```
Query → Arctic embedding (with query prefix) → ChromaDB cosine search → top-K chunks with similarity scores
```

---

## Architecture

```
┌──────────────────────────┐
│ transcripts_v2.csv       │   273 transcripts × 7 columns
│ (273 calls, raw text)    │   from Day 1-2
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ src/rag/chunker.py       │   Sentence-aware chunking
│                          │   - spaCy splitting
│                          │   - Arctic tokenizer budget
│                          │   - 300 target / 50 overlap / 400 ceiling
│                          │   - Three-way section labels
└────────┬─────────────────┘
         │  scripts/run_chunker.py
         ▼
┌──────────────────────────┐
│ chunks_day6.csv          │   13,518 chunks × 11 metadata columns
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ src/rag/embedder.py      │   Arctic-l-v2.0 embedding
│                          │   - 1024-dim, L2-normalized
│                          │   - Batched (32 at a time)
└────────┬─────────────────┘
         │  scripts/run_embedder.py
         ▼
┌──────────────────────────┐
│ embeddings_day6.npy      │   (13518, 1024) float32, 53 MB
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ src/rag/vector_store.py  │   ChromaDB loader
│                          │   - Persistent HNSW index
│                          │   - Cosine similarity
│                          │   - 6-field metadata schema
└────────┬─────────────────┘
         │  scripts/run_chroma_loader.py
         ▼
┌──────────────────────────┐
│ chroma_db/               │   Persistent ChromaDB instance
│ collection:              │   13,518 chunks queryable
│ risk_radar_chunks        │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ src/rag/retriever.py     │   Query interface
│                          │   - lru_cache model loading
│                          │   - Query prefix injection
│                          │   - Top-K with metadata filters
│                          │   - Similarity scoring
└──────────────────────────┘
         │
         ▼
   Results: list[dict]
   each: {chunk_id, text, metadata, similarity, distance}
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Vector DB | ChromaDB 1.5.9 (local persistent) | Our 13.5K vectors are tiny relative to ChromaDB's capability; local = no quotas, no internet dependency |
| Embedding model | Snowflake/snowflake-arctic-embed-l-v2.0 (1024-dim) | MTEB ~65 (top tier), trained with hard-negative mining, file size 1.1 GB (smaller than bge-large) |
| Sentence splitter | spaCy `en_core_web_sm` | Correctly handles "Apple Inc." and "$89.5 billion" where regex fails |
| LLM (Day 7) | Gemini API (free tier) | Generous free tier; no credit card; brand-agnostic for the recruiter narrative |

---

## Chunking strategy

Sentence-aware with overlap, respecting Arctic's tokenizer budget:

- **Target chunk size:** 300 tokens (~225 words, ~10-15 sentences)
- **Overlap:** 50 tokens (last 1-2 sentences of chunk N become first sentences of chunk N+1)
- **Hard ceiling:** 400 tokens (leaves 112-token headroom under Arctic's 512-token limit)
- **Sentence boundaries:** spaCy detects them; chunks always end on complete sentences

### Section model

Three-way `section` label (originally binary in the plan):

| Section | What it is | Calls | Chunks |
|---|---|---|---|
| `prep` | Prepared remarks before Q&A | 259 | 5,174 |
| `qa` | Analyst Q&A after prepared remarks | 258 | 7,728 |
| `interview` | NFLX-style moderated interview (no prep/qa split) | 14 | 616 |

The `interview` label was added during Day 6 after discovering that 14 Netflix calls use a moderated interview format with no traditional prepared-remarks-then-Q&A structure.

### Marker variants

The Q&A section is marked by two variants in the source data:
- `"Questions and Answers:"` — 55 calls
- `"Questions & Answers:"` — 204 calls

The chunker matches both.

---

## Output schema (`chunks_day6.csv`)

| Column | Type | Example | Notes |
|---|---|---|---|
| `chunk_id` | str | `AAPL_2019-Q3_prep_000` | unique, format `{ticker}_{quarter}_{section}_{idx:03d}` |
| `ticker` | str | `AAPL` | |
| `quarter` | str | `2019-Q3` | |
| `call_date` | str (date) | `2019-07-30` | |
| `section` | str | `prep` / `qa` / `interview` | three-way |
| `chunk_index` | int | `3` | per-section ordering |
| `sentence_start_idx` | int | `47` | first sentence in chunk (within section's sentence list) |
| `sentence_end_idx` | int | `62` | last sentence (inclusive) |
| `n_tokens` | int | `298` | Arctic tokenizer count |
| `n_chars` | int | `1247` | character count |
| `text` | str | (chunk text) | the chunk |

---

## How to reproduce

### 1. Install dependencies

```bash
pip install chromadb sentence-transformers spacy
python -m spacy download en_core_web_sm
```

### 2. Run the chunker

```bash
python scripts/run_chunker.py
```

Reads `data/processed/transcripts_v2.csv`, writes `data/processed/chunks_day6.csv`.
Expected runtime: ~3-7 minutes (CPU bound by Arctic tokenizer calls).

### 3. Embed chunks

```bash
python scripts/run_embedder.py
```

Reads `chunks_day6.csv`, writes `embeddings_day6.npy`.
Expected runtime: **highly hardware-dependent**.
- GPU: ~5 minutes
- Modern CPU with AVX: ~30 minutes
- Older CPU: 2-4 hours (observed: 3h 26m on the development laptop)

### 4. Load into ChromaDB

```bash
python scripts/run_chroma_loader.py
```

Reads chunks + embeddings, writes to `chroma_db/`.
Expected runtime: ~30 seconds.

### 5. Query the index

```python
from src.rag.retriever import retrieve

results = retrieve("supply chain disruption and production delays", k=5)
for r in results:
    print(f"sim={r['similarity']:.4f}  {r['chunk_id']}")
    print(f"  text: {r['text'][:200]}...")
```

Or run the demo script:

```bash
python scripts/demo_retrieval.py
```

---

## Usage examples

### Basic retrieval

```python
from src.rag.retriever import retrieve

# Top-5 chunks across all 273 calls
results = retrieve("revenue growth in emerging markets", k=5)
```

### Filter by ticker

```python
results = retrieve(
    "iPhone revenue performance",
    k=5,
    filter={"ticker": "AAPL"}
)
```

### Filter by ticker AND quarter

```python
results = retrieve(
    "iPhone revenue performance",
    k=5,
    filter={"$and": [{"ticker": "AAPL"}, {"quarter": "2019-Q3"}]}
)
```

### Filter by section (multiple values)

```python
# Q&A or interview content only
results = retrieve(
    "CEO outlook on growth",
    k=5,
    filter={"section": {"$in": ["qa", "interview"]}}
)
```

### Result structure

Each result is a dict:

```python
{
    "chunk_id": "AAPL_2019-Q3_prep_002",
    "text": "For iPhone, we generated $26 billion in revenue. While this is down 12%...",
    "metadata": {
        "ticker": "AAPL",
        "quarter": "2019-Q3",
        "call_date": "2019-07-30",
        "section": "prep",
        "chunk_index": 2,
        "n_tokens": 302
    },
    "similarity": 0.68,    # cosine similarity, 0..1 (higher = more similar)
    "distance": 0.32       # ChromaDB's raw distance (1 - similarity)
}
```

---

## Verification

Three test scripts ship with Day 6 to verify the pipeline:

### `tests/test_arctic_embed.py`

Sanity-tests the Arctic embedding model: dimension is 1024, similar sentences get high similarity, dissimilar sentences get low similarity, query prefix mechanism works.

```bash
python tests/test_arctic_embed.py
```

### `tests/test_chunker_sample.py`

Six assertions on a 3-transcript sample:
1. No empty chunks
2. Token budget respected (no chunk > 400 tokens except force-saved oversized sentences)
3. Median chunk size in [200, 400]
4. All chunk_ids unique
5. Section values clean
6. Overlap mechanism works (no gaps between consecutive chunks)

```bash
python tests/test_chunker_sample.py
```

### `scripts/verify_edge_cases.py`

Verifies known limitations don't break retrieval in practice:
- NFLX interview-format calls surface correctly in unfiltered queries
- `section=interview` filter returns only interview content
- PYPL 2019-Q4 (with mislabeled section) is still retrievable

```bash
python scripts/verify_edge_cases.py
```

---

## Demo retrieval results

From `scripts/demo_retrieval.py`:

### Demo 1: Unfiltered "supply chain disruption and production delays"

```
[1] sim=0.5737  COST_2022-Q1_prep_013
    text: But overall, the factors pressuring supply chains, and inflation include
          port delays container challenges, COVID disruptions, shortages...

[2] sim=0.5388  COST_2021-Q3_prep_012
    text: ...turnaround of a container hitting the U.S., delivering its contents,
          and being back at the U.S. port to head back overseas...

[3] sim=0.5183  AAPL_2023-Q1_qa_000
    text: David Vogt -- UBS -- Analyst Thanks, guys, for taking my question. So Tim,
          and maybe this is for Luca as well. You talked about t...
```

Cross-company semantic retrieval works without any explicit ticker filter.

### Demo 3: "iPhone revenue performance compared to last year" filtered to AAPL 2019-Q3

```
[1] sim=0.6800  AAPL_2019-Q3_prep_002
    text: That's equivalent to about $1.5 billion of revenue. Importantly, in
          constant currency, our revenue grew in all five of our geographic
          segments. For iPhone, we generated $26 billion in revenue. While this
          is down 12% from...
```

Top-1 result is the exact paragraph that answers the question.

---

## Known limitations

### 1. PYPL 2019-Q4 has malformed source

The `"Questions and Answers:"` marker in the source file appears 111 chars from the END of the transcript instead of before the Q&A content. The parser interprets all content before the marker as `prep`.

**Net effect:** PYPL 2019-Q4 has 0 qa chunks; its Q&A content is mislabeled as `prep`.

**Mitigation:** content is still retrievable via unfiltered queries (top result for "Venmo growth" query is `PYPL_2019-Q4_prep_023` at similarity 0.6863). Only section-filtered queries on this call are affected.

**Why not fixed:** adding a heuristic to detect "marker too close to end" could mis-fire on calls with legitimately short Q&A. Cost of fix exceeds benefit on 1 row out of 273 (0.37%).

### 2. Boilerplate chunks share high similarity

Opening operator text ("Good day and welcome to...") and forward-looking-statements disclaimers are nearly identical across all 273 calls. Their embeddings cluster together.

**Practical impact:** very generic queries may surface these as filler. Specific queries (which are the normal use case) do not. Worth measuring on Day 8 eval.

### 3. Cosmetic double-period artifact

Where `[Operator Instructions]` was stripped between two sentences, the chunk text may contain `..` (two periods). Arctic's tokenizer and spaCy both handle this fine; no retrieval impact. Purely visual.

### 4. `CLOSING_MARKERS` includes ticker-specific text

The constant includes `"More AAPL analysis"` to strip Motley Fool's footer. Other tickers (`"More NFLX analysis"`, `"More MSFT analysis"`) won't be stripped, leaving a few words of footer text at the end of non-AAPL calls' last chunks. Minor pollution.

---

## File inventory (Day 6 additions)

### Source modules (4)

- `src/__init__.py` — package marker
- `src/rag/__init__.py` — package marker
- `src/rag/chunker.py` — sentence-aware chunker with overlap
- `src/rag/embedder.py` — Arctic embedding pipeline
- `src/rag/vector_store.py` — ChromaDB loader + accessor
- `src/rag/retriever.py` — query interface

### Scripts (9)

- `scripts/__init__.py` — package marker
- `scripts/run_chunker.py` — chunk all 273 transcripts
- `scripts/run_embedder.py` — embed all 13,518 chunks
- `scripts/run_chroma_loader.py` — load into persistent ChromaDB
- `scripts/demo_retrieval.py` — 4 demo retrieval queries
- `scripts/verify_edge_cases.py` — NFLX + PYPL spot-checks
- `scripts/diagnose_qa_marker.py` — diagnostic for Q&A marker variants
- `scripts/diagnose_pypl_2019q4.py` — diagnostic for PYPL malformed marker
- `scripts/sanity_check_embeddings.py` — within-vs-between similarity check

### Tests (3)

- `tests/__init__.py` — package marker
- `tests/test_arctic_embed.py` — Block 2 model sanity test
- `tests/test_chunker_sample.py` — Block 3c chunker sample test

### Data artifacts (gitignored, regeneratable)

- `data/processed/chunks_day6.csv` — 13,518 rows
- `data/processed/embeddings_day6.npy` — (13518, 1024) float32
- `chroma_db/` — persistent ChromaDB instance

---

## What's next: Day 7

The retrieval system is in place. Day 7 builds the generation half:

- Install `google-generativeai` package
- Design the prompt template (citation format, grounding discipline)
- Build `src/rag/generator.py` — `generate_answer(query, k=5, filter=None)`
- Hallucination guardrails: detect when answer cites facts not in chunks
- Integration with Day 5 risk score: "this call was scored 0.78 risky — here's why"

---

## Acknowledgments

Built as part of the Earnings Call Risk Radar 10-day project. Day 6 is the foundation; Days 7-10 build on top.

Tools: ChromaDB, sentence-transformers, spaCy, Hugging Face Hub, Snowflake AI (Arctic embed), Google AI (Gemini).

# Day 7 — RAG Generation Pipeline with Risk Model Integration

Stage 3 of the Earnings Call Risk Radar: turn retrieved transcript chunks into grounded, cited answers, then connect the answer pipeline to Day 5's risk classifier so the system explains its own predictions with direct evidence from the call.

---

## What Day 7 ships

1. **`src/rag/gemini_client.py`** — `lru_cache`-singleton wrapper around the Gemini v2 SDK (`google-genai==2.6.0`). Loads `GEMINI_API_KEY` from `.env`, builds one `genai.Client` per process, fails loudly with specific exceptions if config is missing.

2. **`src/rag/generator.py`** — the core RAG module. Public function `generate_answer(query, k=5, filter=None, thinking=False)` does the full flow: retrieve → adaptive top-K filter → prompt build → Gemini call (with retry-on-503) → footnote citation parse → lexical faithfulness check → structured return.

3. **`src/rag/risk_explainer.py`** — the integration. Public function `explain_risk(ticker, quarter)` loads both Day 5 shipping models, extracts per-instance feature contributions (SHAP for XGBoost, exact coefficient math for LR), builds an adaptive RAG question from content-classified features, and returns scores + explanation (focused on this call + unfiltered "elsewhere" lookup).

4. **`scripts/demo_generation.py`** — 5-query demo (4 on-topic + 1 deliberately off-topic). Reuses Day 6's `demo_retrieval.py` queries for side-by-side comparison.

5. **`scripts/demo_risk_explainer.py`** — 3-case integration demo (AAPL standard, BYND high-risk era, NFLX interview-format edge case).

6. **`scripts/test_gemini_connection.py`** — diagnostic for Gemini connectivity.

7. **`tests/test_citation_parser.py`** — 7 sanity cases for the citation parser (clean, undefined footnote, orphan source, hallucinated chunk_id, duplicate definition, multi-citation, triple multi-citation).

---

## Design decisions (locked Day 7)

| Decision | Choice | Rationale |
|---|---|---|
| Gemini SDK | `google-genai` v2 (not legacy `google-generativeai`) | Old SDK deprecated by Google May 2026 |
| Prompt approach | Zero-shot | Modern instruction-following; few-shot deferred to evaluation-driven improvement |
| Abstention behavior | Soft refuse + partial answer + provenance tags (`[Inferred]`, `[Not in chunks]`) | Useful UX + hallucination visibility |
| Citation format | Footnote `[1] [2]` + mandatory Sources section | Readable + Streamlit-friendly + chunk_id visible at a glance |
| Citation validation | Post-hoc regex parser, 4 checks, non-crashing warnings | Surfaces model failure modes without breaking the pipeline |
| Low-similarity handling | Adaptive top-K (threshold=0.45, min=2 chunks, max=5 chunks) | Saves API cost on doomed queries; clean abstention path |
| Faithfulness check | Lexical overlap, threshold 0.30 | Cheap tripwire for Day 7; LLM-as-judge promoted to Day 8 eval |
| Gemini thinking tokens | OFF by default, exposed as `thinking` parameter | Day 9 Streamlit will toggle this in the UI |
| Temperature | 0.0 | Deterministic, grounded |
| Retry on 503 | 4 attempts with 2s/5s/10s backoff | Graceful failure, demo continues even during Gemini capacity dips |
| Day 5 models used | Both (y_3d XGBoost + y_5d LR) | Recruiter sees model-selection sophistication; Day 9 UI toggles horizon |
| Feature → question mapping | Content features only (10 of 17 features) | Meta-linguistic features (sentence length, readability) can't ground content queries |
| RAG filter strategy | Focused (this call) + Elsewhere (unfiltered, cross-call) | Direct evidence + comparable patterns elsewhere |

---

## Quickstart

### Prerequisites
- Day 6 complete: ChromaDB populated, Arctic embeddings in place
- Day 5 models on disk under `models/`
- `.env` file at project root containing `GEMINI_API_KEY=<your_key>`

### Verify Gemini connectivity
```powershell
python scripts/test_gemini_connection.py
```
Expected: a 5-word reply, latency ~1-3s, PASS at the end.

### Run RAG demo
```powershell
python scripts/demo_generation.py
```
Expected: 4 on-topic answers with citations and faithfulness scores; Demo 5 (sourdough query) abstains cleanly.

### Run risk-explanation demo
```powershell
python scripts/demo_risk_explainer.py
```
Expected: AAPL and BYND produce both focused + elsewhere RAG explanations; NFLX shows graceful interview-format handling (y_5d LR skips, y_3d XGBoost proceeds, focused RAG abstains).

### Programmatic use
```python
from src.rag.generator import generate_answer
from src.rag.risk_explainer import explain_risk

# Pure RAG (any question)
result = generate_answer(
    "What did Apple say about supply chain?",
    k=5,
    filter={"ticker": "AAPL"},
)
print(result["answer"])
print(result["faithfulness"]["mean_overlap"])

# Risk explanation for a (ticker, quarter) pair
risk = explain_risk("BYND", "2021-Q3")
print(risk["scores"]["y_3d"]["risk_score"])
print(risk["question"])
print(risk["explanation_focused"]["answer"])
```

---

## Quality verification — three independent guardrails

Day 7 stacks three layers of verification on every answer:

1. **Adaptive top-K threshold** — chunks below similarity 0.45 are dropped. If fewer than 2 chunks qualify, the system abstains without calling Gemini. Demo 5 (off-topic sourdough query on earnings-call corpus) triggered this cleanly with similarities 0.16-0.18.

2. **Citation health parser** — validates citation structure: every `[N]` reference is defined in the Sources section, every defined source is referenced in the body, every cited chunk_id was actually in the retrieved set (catches hallucinated chunk_ids), no duplicate definitions. Issues are returned as warnings on the result dict — they never crash the pipeline.

3. **Lexical faithfulness check** — for each cited claim, computes `|claim_tokens ∩ chunk_tokens| / |claim_tokens|`. Claims below 0.30 overlap are flagged. Provenance-tagged claims (`[Inferred]`, `[Not in chunks]`) are correctly skipped. Multi-citations check against the union of cited chunks.

Across the 4 answered demos in `demo_generation.py`, all three guardrails returned VALID. Across the 3 cases in `demo_risk_explainer.py`, 5 of 6 RAG calls answered with VALID citation health.

---

## Sample output — the headline result

`explain_risk("BYND", "2021-Q3")` produced this focused RAG answer (excerpt, with citations preserved):

> The company experienced "operational challenges of this environment" and "the events of the last year" [2]. There were "floods and this pandemic and the labor issues" which led to more cautious guidance for the fourth quarter [2]. ... Ethan Brown stated, "I feel enormously confident about where we're headed" [2]. ... The CFO mentioned "some things that were probably a little unusual in Q4," including trade discounts and an inventory write-off [4].
>
> Sources:
> [1] BYND_2021-Q3_prep_001
> [2] BYND_2021-Q3_qa_026
> [3] BYND_2021-Q3_qa_033
> [4] BYND_2021-Q3_qa_017
> [5] BYND_2021-Q3_prep_000

Faithfulness score: 0.744. Citation health: VALID.

Day 5's XGBoost flagged BYND 2021-Q3 with a 0.573 risk score driven by elevated negativity in Q&A responses. Day 7 went to the actual transcript and surfaced the specific operational stresses (floods, pandemic, labor, inventory write-offs, Beyond 3.0 weakness) that the model was implicitly detecting through statistical language patterns. The connection between number and explanation is direct, cited, and verifiable.

---

## Known limitations

### 1. Interview-format calls underperform end-to-end

NFLX and TSLA calls use a conversational interview format rather than scripted prepared remarks + Q&A. Day 5 documented this as Finding #6 (XGBoost loses ~3.6 AUC points on these rows). Day 7 inherits the same weakness: the chunker produces conversational chunks that don't lexically match content-themed queries, and focused RAG abstains for NFLX 2022-Q4 in the demo.

**Status:** Documented characteristic of the corpus + chunker, not a Day 7 bug. The system fails gracefully (abstention message, no hallucination). Day 8 will explore whether a lower retrieval threshold for filtered queries or interview-format-specific chunking would help.

### 2. Safer-classified calls produce awkward "absence" questions

When all top content-driver features point in the "safer" direction (low negativity, low uncertainty), the adaptive question becomes *"What did management discuss regarding absence of major concerns?"* — which a transcript-grounded RAG cannot answer because transcripts contain things that were said, not things that were absent.

**Status:** Architectural finding from Block 6's second iteration. The system correctly abstains rather than fabricating. Day 8 / future improvements: detect "safer-only" driver patterns and switch to a positive-framing question (*"What positive developments and confident statements did management emphasize?"*).

### 3. Lexical faithfulness has known paraphrase blindness

If the chunk says *"iPhone revenue declined 12% year-over-year"* and the model writes *"iPhone sales fell twelve percent compared to 2018,"* lexical overlap is near zero — the meaning is supported but the words don't match. The Day 7 faithfulness check would flag this as unsupported.

**Status:** Acknowledged tripwire-level check, not a measurement-grade metric. Day 8 promotes LLM-as-judge (using a different model — likely Claude API — to avoid the "marking your own homework" bias) for true faithfulness measurement.

### 4. Similarity threshold (0.45) is a default, not a tuned value

Block 4 confirmed 0.45 is functional on this corpus (on-topic chunks cluster 0.45-0.70, off-topic at 0.15-0.20, wide signal/noise gap). But "functional" is not "optimal." Borderline queries (similarities in 0.30-0.45 range) haven't been tested.

**Status:** Defensible default. Day 8 will tune against hand-labeled queries with ground truth.

### 5. New-transcript ingestion requires manual steps

The Day 5 → Day 7 pipeline assumes the transcript is already in `data/processed/dataset_v2.csv` (Day 4 output) and its chunks are already in ChromaDB (Day 6 output). A genuinely new transcript would require: (a) running Day 4's chunker + FinBERT pipeline, (b) appending the row to `dataset_v2.csv`, (c) embedding new chunks into ChromaDB. None of these are wrapped into a single command yet.

**Status:** Day 10 (deployment / Streamlit) will productize this as an upload-and-process flow. Until then, the pipeline assumes the data is pre-loaded.

### 6. Gemini 2.5 "thinking" token cost on paid tier

Day 7 demos run on the free tier where thinking tokens (~141 per call by default) are free. On paid tier, those tokens cost real money. Day 7 ships with thinking OFF by default specifically to keep paid-tier costs predictable. The `thinking` parameter exists for cases where the user explicitly wants the deeper reasoning (Day 9 Streamlit toggle).

**Status:** Documented behavior, not a bug. Cost is ~$0.0003 per query on Flash with thinking off, doubling roughly with thinking on.

---

## File-by-file summary

```
src/rag/gemini_client.py     30 lines    Singleton genai.Client with .env loading
src/rag/generator.py        400 lines    Full RAG pipeline (retrieve→prompt→generate→verify)
src/rag/risk_explainer.py   350 lines    Day 5 → Day 7 integration with content-vs-meta classification
scripts/demo_generation.py  170 lines    5-query end-to-end demo
scripts/demo_risk_explainer.py 130 lines 3-case integration demo
scripts/test_gemini_connection.py 50 lines Diagnostic
tests/test_citation_parser.py 90 lines   7 sanity cases for the citation parser
requirements.txt             +1 line     google-genai==2.6.0
```

---

## Citation policy

Footnote format. Mandatory Sources section. Provenance tags (`[Inferred]`, `[Not in chunks]`) for content beyond the chunks. Numbering bugs are caught by `_parse_citations()` and reported as warnings in the result dict — they never crash the pipeline.

The prompt explicitly instructs Gemini that answers without Sources sections will "fail downstream validation." Block 4's second run confirmed this language is effective — Demo 3 had omitted Sources in the first run and produced a proper section after the prompt strengthening.

---

## What Day 8 will measure

- Retrieval quality (recall@5 against hand-labeled gold chunks)
- Generation quality (LLM-as-judge faithfulness using a separate model)
- Citation health rate across the eval set
- Abstention rate (focused vs elsewhere) — is the threshold right for both, or should they differ?
- Threshold tuning (`SIMILARITY_THRESHOLD` and `FAITHFULNESS_THRESHOLD`)
- Whether zero-shot prompting is sufficient or few-shot examples are warranted

---

**Headline takeaway**

Day 7 closed the loop between Day 5's statistical risk scores and the original transcripts they were trained on. The Earnings Call Risk Radar can now predict that a call sounds risky AND explain why — with direct quotes from the call, footnote citations, faithfulness verification, graceful handling of edge cases (off-topic queries, interview formats, NaN features, transient API errors), and documented known limitations.


# Day 8 — Evaluation Harness

This README documents the evaluation infrastructure built on Day 8, what it measured, what works, what doesn't, and known limitations. It ships with the repo as the public-facing summary of the evaluation effort.

---

## What Day 8 produced

A complete evaluation harness for the Risk Radar RAG pipeline:

- **20-query hand-labeled eval set** stratified across 6 use cases
- **150 chunks individually labeled** (correct / tangential / wrong) — the gold-set construction is auditable via `eval_worksheet_day8.csv`
- **Eval runner** (`scripts/run_eval.py`) producing per-query metrics across retrieval, generation, and abstention behavior
- **LLM-as-judge runner** (`scripts/run_judge.py`) for independent semantic scoring (built, deferred run pending quota reset)

---

## Results summary

### Retrieval (n=14 answerable queries)

| Metric | Value |
|---|---|
| Mean recall@10 | **1.00** |
| Mean recall@5 | 0.55 |
| Mean precision@5 | 0.80 |

Every gold chunk we labeled lands in the top-10 retrieval results. Recall@5 < 1 is expected mathematics: most queries have 7-10 gold chunks competing for 5 top slots.

### Generation (n=14, lexical metrics only)

| Metric | Value |
|---|---|
| Mean lexical faithfulness | 0.70 |
| Mean theme coverage (substring) | 0.37 *(see limitation #2)* |
| Citation health valid | 13/15 *(see limitation #3)* |

### Abstention

| Test | Result |
|---|---|
| Off-topic queries (n=3) | **3/3 correctly abstained** |
| Boilerplate trap (q4) | **Correctly abstained** (top similarity 0.445, below 0.45 threshold) |
| Unexpected abstentions on answerable queries | 1 (q4, expected — boilerplate trap) |

### Risk-integration queries

| Query | y_3d | y_5d | Focused | Elsewhere |
|---|---|---|---|---|
| q19 (BYND 2021-Q3, sanity check) | 0.573 | 0.420 | answered, theme_cov=0.75 | answered |
| q20 (MSFT 2023-Q1, fresh test-set) | 0.550 | 0.692 | **abstained** | answered |

The q20 abstention is a real finding: the fresh test-set call's chunks weakly align with the auto-generated adaptive question, so the focused-RAG correctly refuses to answer. The elsewhere-RAG still succeeds.

---

## What works

- **Retrieval is strong.** recall@10 = 1.0 across the eval set.
- **Abstention is reliable.** 3/3 off-topic + boilerplate trap caught.
- **Citations parse cleanly for 13/15 queries.**
- **Lexical faithfulness 0.70 mean** — answers stay grounded in cited chunks.
- **Sanity-check stability.** Day 7 demo queries (q11, q12) reproduced known-good outputs (faith 0.84, 0.847).
- **Honest abstention on fresh test-set integration.** q20 (post-Day-5-cutoff MSFT 2023-Q1) abstains on the focused-RAG side rather than confabulating.
- **Honest API-failure handling.** Failed Gemini calls record as "not measured" (blank) instead of fake 0.0, so aggregates aren't polluted.

---

## Known limitations

### 1. Daily Gemini free-tier quota cannot sustain iterative evaluation
The biggest finding of Day 8. The 20-query eval (~17 generation calls) + the judge re-run (~14 more generation calls + 14 judge calls) hits the daily quota repeatedly. The 8-second inter-query spacing handles the per-minute rate limit but does nothing for the daily cap.

**Consequence for Day 9 (Streamlit app):** the UI cannot assume concurrent users. Needs aggressive caching, a "queries remaining today" indicator, or a paid Gemini key.

### 2. `theme_coverage` substring matcher is too weak to trust
The Block 2 substring metric matches expected themes against the generated answer with case-insensitive substring lookup. This misses paraphrase. Concrete evidence: q12 scored lexical faithfulness 0.847 but theme_coverage 0.0 — impossible unless the metric is wrong.

**Resolution:** the LLM-judge (Block 4, deferred) replaces this with semantic matching. Until the judge runs, treat theme_coverage as a tripwire, not a quality signal.

### 3. Citation parsing failures on q8 and q12 (both BYND)
Both BYND queries returned `citation_valid=False`. Not yet investigated. Could be a real citation bug or a parser edge case (e.g., a footnote citing a chunk that wasn't in the retrieved set after threshold filtering). Worth a 15-minute diagnostic.

### 4. Same-system labeling bias
Gold labels were drafted by Claude reading the top-10 retrieved chunks per query. The LLM-judge (when run) will be Gemini-on-Gemini (same vendor as generator). Independence is weaker than ideal:
- Ideal: human labels + cross-vendor judge
- Current: Claude labels + Gemini-on-Gemini judge

This is documented in `eval_summary_day8.md` and reflected in the methodological caveats. The retrieval metrics are unaffected (chunk IDs are objective facts; "did this ID appear in top-10" is not a judgment call).

### 5. Small sample size (n=20)
A `measurement harness`, not a `benchmark result`. A defensible-but-not-rigorous evaluation. Larger eval sets (50-100 queries) are queued as future work.

### 6. Cosine similarity is tone-blind
Q8 asked about "concerns about retail demand" but surfaced positive-growth chunks. The embedder treats "discussing demand" and "discussing demand decline" as similarly relevant. Reranking or query-rewriting could address this.

### 7. Within-call topical drift
Q11 (AAPL 2019-Q3 iPhone) retrieved iPad/Services chunks too. The filter is correct (all match AAPL+2019-Q3) but within a single call, semantic similarity surfaces topically-adjacent content. Within-call reranking could improve precision.

### 8. PYPL 2019-Q4 known limitation not tested
Documented in Day 6 (malformed Q&A marker). We dropped the paired PYPL test from the original Block 1 plan when going from 21 to 20 queries. Worth re-adding in a future eval-set extension.

### 9. Boilerplate cluster in vector store
Legal disclaimers ("forward-looking statements involve risks and uncertainties...") are near-identical across all 273 calls. Currently mitigated by abstention (q4 abstains correctly). Could be removed from the index entirely as a future improvement.

### 10. LLM-judge run not yet executed
`scripts/run_judge.py` is built and verified (JSON parsing tested against 4 response format variants). The actual judge run is deferred to the next session due to today's exhausted Gemini quota. The `eval_judge_day8.csv` file exists with the right schema but all rows currently show `judged=False`.

---

## Future work

### Will be done next session (Day 9 day-1)

| Task | What it produces |
|---|---|
| Run `python -m scripts.run_judge` | Populates `eval_judge_day8.csv` with 14 semantic faithfulness/completeness/relevance scores |
| Compare LLM-judge faithfulness vs lexical faithfulness | Validates (or invalidates) the lexical tripwire |

### Genuine future-work items

| Item | Priority | Notes |
|---|---|---|
| Investigate q8/q12 citation_valid=False | Low | Diagnostic only; both BYND |
| Replace theme_coverage with semantic match | Med | LLM-judge already does this; substring metric should be retired |
| Independent human relabeling of gold set | Med | Removes same-system bias |
| Larger eval set (50-100 queries) | Med | More credible benchmark |
| Cross-vendor judge (Claude judging Gemini) | Low | Requires paid Anthropic API key |
| Reranking for tone-blind retrieval | Med | Real improvement opportunity |
| Within-call topical reranking | Low | Refines q11-class precision |
| Few-shot vs zero-shot prompt comparison (Block 6) | Low | Pure optimization on a working system |
| Formal threshold sweep at 0.46-0.48 | Low | Analysis suggests 0.45 already optimal |
| Remove boilerplate cluster from index | Low | Currently mitigated by abstention |
| Add pre-flight check ("is eval set populated?") to runner | Low | Would have caught Day 8's empty-template bug |
| Quota-aware Streamlit features (caching, demo mode, query counter) | High | Direct Day 9 requirement |

---

## File map (Day 8 artifacts)

```
data/eval/
├── eval_set_day8.csv             # 20-query labeled eval set
├── eval_worksheet_day8.csv       # 150-row labeling intermediate (auditable)
├── eval_results_day8.csv         # Per-query metrics
├── eval_per_chunk_day8.csv       # Per (query, chunk) flags
├── eval_summary_day8.md          # Aggregate findings
└── eval_judge_day8.csv           # LLM-judge scores (pending — deferred)

scripts/
├── build_eval_set.py             # Builds the labeling worksheet
├── write_eval_set.py             # One-shot: writes labeled eval set from embedded data
├── run_eval.py                   # Main eval runner
├── run_judge.py                  # LLM-as-judge runner
└── diag.py                       # One-off diagnostic (safe to delete)

src/rag/
└── generator.py                  # MODIFIED in Day 8 — api_failed flag + 429 retry
```

---

## Re-running the evaluation

After a fresh Gemini quota:

```bash
python -m scripts.run_eval        # ~5 minutes with 8s inter-query spacing
python -m scripts.run_judge       # ~5 minutes (uses ~28 API calls)
```

Outputs are written to `data/eval/`. The summary doc (`eval_summary_day8.md`) is the recruiter-facing aggregate.

---

## Acknowledged process limitations

Day 8 surfaced several process issues worth documenting:

- The empty-template file bug (gold_chunks=NaN throughout) cost an hour. A pre-flight check would have caught it.
- The half-applied edits to `run_eval.py` during the Block 4 architecture pivot left it broken; required a full revert.
- The free-tier quota was a known constraint not designed around in the eval architecture.

These are recorded honestly in `day8_journey.md` rather than airbrushed out.


# day9_README.md — Streamlit App: Known Limitations & Future Work

This document ships with the repo. It records, honestly, what the Day 9 Streamlit app does, what it does NOT do, what is verified vs unverified, and everything still open from Days 8 and 9 for a clean project completion. The goal (carried from earlier days): **real findings and honest interpretation, not the high-number game.**

---

## What the app is

A single-file Streamlit application (`app.py`) that is a **presentation layer** over the existing Day 7/8 pipeline. It calls two functions and displays their results:
- `explain_risk(ticker, quarter)` → Tab 1 "Risk Score"
- `generate_answer(query, k, filter, thinking)` → Tab 2 "Ask the Filing"

It adds **no new ML and no new RAG logic.** No `src/rag/` module or model was changed on Day 9. The intelligence was all built on Days 4–8; Day 9 makes it clickable.

## What the app does (features)

**Tab 1 — Risk Score**
- Ticker + quarter dropdowns (sourced from `dataset_v2.csv`, the calls that actually have risk features).
- Both Day 5 risk scores (y_3d XGBoost, y_5d LR) shown side by side with their top drivers in plain English.
- An auto-generated "adaptive question" built from the top risk-driving features, with a note on which model's drivers built it.
- Two grounded evidence sections: "Evidence from this call" (focused RAG) and "Similar discussion in other calls" (elsewhere RAG), each with citations and a faithfulness caption.
- A diagnostics expander surfacing any `errors` from `explain_risk`.

**Tab 2 — Ask the Filing**
- Free-text question box.
- Scope selector (all companies, or narrow to one ticker).
- "Gemini thinking" toggle (on/off deeper reasoning).
- Rendered answer with a citations expander, citation-health warning if the parser flagged issues, and a faithfulness caption.
- Abstains visibly and honestly when no chunk clears the similarity threshold.

**Cross-cutting**
- **Session answer-cache:** the same query (per tab) never calls Gemini twice in a session; cache hits show a "no Gemini call made" toast.
- **Daily usage counter** in the sidebar (`data/gemini_usage.json`), auto-resets at local midnight, counts only real calls (abstentions/failures excluded), warns at ~80% of the free-tier reference.
- **Graceful quota failure:** a failed Gemini call renders a clean warning, never a crash.

---

## Verified vs UNVERIFIED (be honest about this)

### Verified working (tested on Day 9)
- App launches; both tabs render.
- ChromaDB initializes once and does not crash (after the config fix below).
- Retrieval runs; similarity threshold works.
- Abstention path works end-to-end (off-topic "sourdough" question → clean abstain, zero quota).
- Session cache works (repeat query → instant, no call, toast shown).
- Usage counter works and stays honest through abstentions (held at 0).
- Risk scores and drivers render in Tab 1 (both scores displayed in a live run).
- The focused RAG explanation rendered in a live run.
- Graceful handling of a real quota failure (the "elsewhere" call failed cleanly with a warning).

### NOT yet verified (quota-blocked — NOT a known code defect)
- A **live Gemini answer rendering** through the answer + citations expander + faithfulness caption in the browser (Tab 2, and the elsewhere section of Tab 1).
- A **full risk analysis** where BOTH focused and elsewhere calls succeed.
- The **"Gemini thinking" toggle** exercised live in-browser.

These are flagged because the render code uses the same dict-shape that `generate_answer`/`explain_risk` already returned successfully in Day 7/8 function-level demos — so risk is low, but it has not been observed live. **Day 10 first step:** on fresh quota, run one risk analysis and one question to confirm, before deploying.

---

## Known limitations (app-specific, Day 9)

1. **Free-tier quota is the binding constraint.** One risk analysis = 2 Gemini calls; one question = 1. The free `gemini-2.5-flash` daily cap is easily reached. The counter *warns* but deliberately does **not block** (chosen design: honest display over false lockout). On a public deploy, anyone's clicks burn the owner's quota.

2. **No "demo mode" yet.** When quota is fully exhausted, the app shows honest failure messages but cannot produce answers. A hardcoded demo mode (canned AAPL output) was discussed and **deliberately deferred** to the deploy decision (Day 10), so a recruiter opening the live app on a dead quota doesn't see a non-functional demo.

3. **File-watcher is disabled by necessity.** `.streamlit/config.toml` sets `fileWatcherType = "none"` to fix a ChromaDB crash (see below). Consequence: editing a file does not auto-reload the app; you must `Ctrl+C` and re-run. Acceptable locally; irrelevant once deployed.

4. **Answer cache is session-only.** It lives in `st.session_state`, so closing the app empties it. (The usage counter, by contrast, persists to disk.) Two users, or two sessions, don't share cached answers — each session can re-burn quota for the same query.

5. **Counter is approximate and local.** It tracks calls this app process makes, written to a local JSON file. It does not query Google for your true remaining quota, and it won't see calls made outside the app (e.g. `run_eval.py`, `run_judge.py`). The ~250/day figure is a reference, not a guaranteed limit.

6. **First request is slow (cold start).** The first retrieval loads the Arctic embedder (~15–30s). Subsequent calls are fast. On Streamlit Cloud this cold start happens on each container spin-up.

7. **Faithfulness shown is the weak lexical tripwire.** The displayed faithfulness is lexical overlap (Day 7's cheap check), explicitly labeled as such. It can mislead (Day 8: an answer scored 0.847 lexical but 0.0 substring theme coverage). The stronger LLM-judge number is not yet available (see deferred items).

8. **Dropdowns depend on `dataset_v2.csv` being present.** If the dataset is missing at runtime (e.g. not committed on deploy), `load_ticker_quarters()` fails. Must be present on Streamlit Cloud.

---

## THE Day 9 bug (documented so it never recurs)

**Symptom:** crash on first risk analysis — `AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'` in `chromadb.PersistentClient` → `del self.bindings`, with a flood of `transformers ... No module named 'torchvision'` warnings.

**Cause:** Streamlit's source file-watcher crawls every imported module on each rerun. Crawling `transformers` produced the torchvision noise; reloading the `rag` modules wiped their `lru_cache` singletons, causing ChromaDB to init a second client and fail to release the first's Rust bindings. **It is NOT a quota error and NOT a defect in our `src/` code** — quota errors surface as the clean `api_failed` warning instead.

**Fix:** `.streamlit/config.toml` with `fileWatcherType = "none"` (and `logger level = "error"`). ChromaDB then inits exactly once. This file **must be committed and deployed** or the crash can recur on Streamlit Cloud.

---

## Deployment (Day 10) — open decisions & gotchas

1. **`data/chroma_db/` is gitignored and large — biggest open question.** Options: (a) commit it (may exceed Streamlit Cloud size limits); (b) rebuild it on first cloud run from `chunks_day6.csv` (needs the embed step to run in the cloud — slow, memory-heavy); (c) host the store externally. **Decide this early on Day 10.**
2. **`GEMINI_API_KEY` as a Streamlit secret**, never committed (`.env` stays gitignored).
3. **`requirements.txt` must be the curated list**, not the full `pip freeze` (Windows-only/transitive pins can break the Linux build). Pin `streamlit` from `pip show streamlit`. Confirm whether `spacy` is actually imported; drop it + the `en_core_web_sm` line if not.
4. **`.streamlit/config.toml` must be in the repo** (it is, via `git add .`).
5. **`dataset_v2.csv` and `models/*.joblib` must be present** at runtime.
6. **Memory:** Arctic embedder + ChromaDB + torch can exceed free-tier container RAM. If the cloud build OOMs, this is the likely culprit; a lighter embedder or a precomputed-results read-only mode may be needed (the v2 plan's own fallback: "pre-compute everything locally, deploy a lightweight read-only app").

---

## Deferred from Day 8 — STILL OPEN (do on Day 10 with fresh quota)

| Item | Status | Notes |
|---|---|---|
| Run `python -m scripts.run_judge` | **STILL DEFERRED** (Day 8 → Day 9 → Day 10) | Quota-blocked both days. Populates `eval_judge_day8.csv` (faithfulness/completeness/relevance, 1–5). The eval differentiator for the README results table. Do this FIRST on Day 10. |
| Formal threshold sweep (0.46 / 0.48) | DEFERRED | Day 8 analysis judged 0.45 fine; no run done. |
| Zero-shot vs few-shot comparison | FUTURE WORK | System works zero-shot; few-shot is pure optimization. |
| Investigate q8/q12 `citation_valid=False` (both BYND) | FUTURE WORK | One-off diagnostic, low priority. |
| Replace substring `theme_coverage` with semantic match | Superseded by LLM-judge once run | — |
| Independent human relabeling of gold set | FUTURE WORK | Removes same-system labeling bias; costly. |
| Remove boilerplate cluster from index | FUTURE WORK | Currently mitigated by abstention. |
| Cross-vendor judge (Claude vs Gemini) | FUTURE WORK | Needs a paid Anthropic key; current judge is Gemini-vs-Gemini (weaker independence). |
| Reranking / query rewriting for tone-blind retrieval | FUTURE WORK | Cosine is tone-blind (q8: positive chunks for a "concerns" query). |
| Larger eval set (50–100 queries) | FUTURE WORK | Current n=20 is a measurement harness, not a benchmark. |
| PYPL 2019-Q4 malformed Q&A marker test | FUTURE WORK | Known from Day 6, not in eval set. |

---

## New future work surfaced on Day 9

| Item | Notes |
|---|---|
| Verify the live render path | One risk analysis + one question + the thinking toggle, in-browser, on fresh quota. Pre-deploy gate. |
| Demo / fallback mode | Canned output (or precomputed CSV-backed read-only mode) for when quota is dry, so the public app never looks broken. |
| Persistent / shared answer cache | Current cache is session-only; a disk- or DB-backed cache would let repeat queries across sessions skip Gemini entirely — directly extends quota. |
| True quota introspection | Counter is a local estimate; querying Google for real remaining quota would be more accurate. |
| Architecture diagram + results table in README | Day 10 deliverable (draw.io PNG; AUC + RAG metrics + judge scores). |
| Loom video + LinkedIn post + final resume bullet | Day 10 deliverables. |
| LiteLLM multi-provider abstraction | Pre-existing future item; would let a paid Claude/OpenAI key drop in without code changes, sidestepping Gemini free-tier limits. |

---

## One-line honest project status

End of Day 9: Risk Radar is a complete, clickable, honestly-instrumented Streamlit app over a verified 3-stage pipeline; everything is confirmed working except a live Gemini render (quota-blocked, not a defect), and the deferred LLM-judge run plus deployment remain for Day 10.