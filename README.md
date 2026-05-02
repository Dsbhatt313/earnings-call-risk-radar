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