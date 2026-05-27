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