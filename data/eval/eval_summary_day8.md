# Day 8 Evaluation Results

Automated evaluation of the Risk Radar RAG pipeline against a 20-query hand-labeled eval set.

**Labeling provenance:** gold labels drafted by Claude with human spot-check. Same-system labeling bias possible; see labeling summary. LLM-as-judge (Block 4) provides an independent generation-quality signal.

## Aggregate metrics (answerable queries)

- Mean recall@5: **0.55** (n=14)
- Mean recall@10: **1.0** (n=14)
- Mean precision@5: **0.8**
- Mean faithfulness (lexical overlap): **0.7**
- Mean theme coverage: **0.371**
- Citation health valid: **13/15**
- Unexpected abstentions (answerable): **1**

## Abstention behavior (off-topic queries)

- Correct abstentions: **3/3**
  - q16: ✅ abstained (top_sim=0.189)
  - q17: ✅ abstained (top_sim=0.239)
  - q18: ✅ abstained (top_sim=0.27)

## Integration (risk explainer)

- q19: y_3d=0.5733, y_5d=0.4203, focused_abstained=False, focused_theme_cov=0.75, elsewhere_abstained=False
- q20: y_3d=0.5499, y_5d=0.6918, focused_abstained=True, focused_theme_cov=, elsewhere_abstained=False

## Per-query detail

| q | stratum | recall@5 | recall@10 | prec@5 | faith | theme_cov | abstain | cite_ok | lat(s) |
|---|---------|----------|-----------|--------|-------|-----------|---------|---------|--------|
| 1 | standard | 0.5 | 1.0 | 1.0 | 0.587 | 0.4 | False | True | 26.0 |
| 2 | standard | 0.556 | 1.0 | 1.0 | 0.81 | 0.4 | False | True | 4.4 |
| 3 | standard | 0.625 | 1.0 | 1.0 | 0.699 | 0.4 | False | True | 5.04 |
| 4 | standard |  |  | 0.0 |  |  | True | True | 0.18 |
| 5 | standard | 0.375 | 1.0 | 0.6 | 0.613 | 0.2 | False | True | 3.91 |
| 6 | standard | 0.625 | 1.0 | 1.0 | 0.686 | 0.0 | False | True | 14.67 |
| 7 | ticker | 0.625 | 1.0 | 1.0 | 0.708 | 0.667 | False | True | 4.67 |
| 8 | ticker | 0.5 | 1.0 | 0.4 | 0.852 | 0.0 | False | False | 3.43 |
| 9 | ticker | 0.444 | 1.0 | 0.8 | 0.583 | 0.833 | False | True | 3.46 |
| 10 | ticker | 0.5 | 1.0 | 1.0 | 0.729 | 0.2 | False | True | 9.05 |
| 11 | ticker_quarter | 1.0 | 1.0 | 1.0 | 0.84 | 0.6 | False | True | 4.14 |
| 12 | ticker_quarter | 0.429 | 1.0 | 0.6 | 0.847 | 0.0 | False | False | 4.29 |
| 13 | ticker_quarter | 0.571 | 1.0 | 0.8 | 0.573 | 0.2 | False | True | 3.17 |
| 14 | section | 0.5 | 1.0 | 1.0 | 0.479 | 0.5 | False | True | 4.52 |
| 15 | section | 0.444 | 1.0 | 0.8 | 0.792 | 0.8 | False | True | 4.38 |
| 16 | off_topic |  |  | 0.0 |  |  | True | True | 0.2 |
| 17 | off_topic |  |  | 0.0 |  |  | True | True | 0.2 |
| 18 | off_topic |  |  | 0.0 |  |  | True | True | 0.18 |
| 19 | risk_integration |  |  |  | 0.744 |  | False | True | 19.32 |
| 20 | risk_integration |  |  |  |  |  | True |  | 3.5 |
