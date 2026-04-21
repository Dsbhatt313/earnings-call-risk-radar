# Earnings Call Risk Radar

A 3-stage pipeline that predicts stock risk from earnings call transcripts.

- Stage 1: FinBERT sentiment extraction
- Stage 2: XGBoost risk classifier
- Stage 3: RAG-based Q&A over transcripts (ChromaDB + Gemini)

**Status:** In development — Day 0 complete.