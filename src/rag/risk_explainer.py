"""
Risk explainer module: bridges Day 5 risk models with Day 7 RAG.

Public API: explain_risk(ticker, quarter) -> dict

For a given (ticker, quarter):
  1. Loads Day 5's shipping models (y_3d XGBoost, y_5d LR)
  2. Looks up the feature row from data/processed/dataset_v2.csv
  3. Runs both models to produce risk scores
  4. Extracts per-instance feature contributions:
     - XGBoost: SHAP values via TreeExplainer
     - LR: feature_value * coefficient (exact, no approximation)
  5. Builds an "adaptive question" from the top risk-driving features
  6. Calls generate_answer() twice:
     - Focused: filtered to this (ticker, quarter)
     - Elsewhere: unfiltered (same risk topics, other calls)
  7. Returns a structured dict for downstream display (Day 9 Streamlit)

Direction-aware mapping respects Day 5's finding that the same feature family
(e.g., positivity) can be safer in prepared remarks but risky in Q&A.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.rag.generator import generate_answer


# ---- Paths & constants ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_v2.csv"

FEATURE_COLS_PATH = MODELS_DIR / "feature_cols_day5.joblib"
MODEL_Y3D_PATH = MODELS_DIR / "model_y3d_xgb_day5.joblib"
MODEL_Y5D_PATH = MODELS_DIR / "model_y5d_lr_day5.joblib"

# How many top drivers to include in the adaptive question.
# Too few = vague; too many = unwieldy. 3 is a sane default.
TOP_DRIVERS_K = 3

# RAG retrieval parameters for explanation calls
RAG_K = 5  # max chunks per RAG call


# ---- Feature → risk-topic mapping (direction-aware) ----------------------

# Each entry: (topic_when_positive_contribution, topic_when_negative_contribution)
# Positive contribution = pushes risk UP. Negative contribution = pushes risk DOWN.
# Content features: their values indicate real business topics that should appear
# in the transcript chunks. These DRIVE the adaptive question.
CONTENT_FEATURE_TOPICS: dict[str, tuple[str, str]] = {
    "negative_count_zscore": (
        "negative developments, setbacks, or challenges",
        "positive developments and absence of major negative news",
    ),
    "uncertainty_count_zscore": (
        "risks, uncertainties, or unclear outcomes",
        "areas of confident, decisive commentary",
    ),
    "litigious_count_zscore": (
        "legal matters, regulatory issues, or compliance topics",
        "absence of legal or regulatory disclosures",
    ),
    "forward_looking_count_zscore": (
        "future guidance, outlook, and forward projections",
        "limited forward guidance",
    ),
    "prep_pos_mean_zscore": (
        "highlights, achievements, or positive results emphasized in prepared remarks",
        "subdued or cautious tone in prepared remarks",
    ),
    "prep_neu_mean_zscore": (
        "factual reporting and balanced commentary in prepared remarks",
        "more emotionally-charged language in prepared remarks",
    ),
    "prep_neg_mean_zscore": (
        "problems, headwinds, or challenges acknowledged in prepared remarks",
        "absence of negative disclosures in prepared remarks",
    ),
    "qa_pos_mean_zscore": (
        "positive talking points or upbeat responses to analyst questions",
        "more reserved or guarded responses to analyst questions",
    ),
    "qa_neu_mean_zscore": (
        "factual, deflective answers to analyst questions",
        "emotionally-engaged answers to analyst questions",
    ),
    "qa_neg_mean_zscore": (
        "negative news, problems, or concerns raised during Q&A",
        "absence of major concerns raised during Q&A",
    ),
}

# Meta-linguistic features: about HOW management spoke, not WHAT they said.
# Tracked for driver transparency but EXCLUDED from question construction.
META_FEATURES: set[str] = {
    "word_count_zscore",
    "sentence_count_zscore",
    "avg_sentence_length_zscore",
    "flesch_kincaid_grade_zscore",
    "numeric_density_zscore",
    "prep_n_chunks_zscore",
    "qa_n_chunks_zscore",
}

# Fallback question when all top drivers are meta-linguistic
GENERIC_RISK_QUESTION = (
    "What were the main risks, challenges, headwinds, or notable concerns "
    "that management discussed during this earnings call?"
)


# ---- Cached loaders ------------------------------------------------------

@lru_cache(maxsize=1)
def _load_feature_cols() -> list[str]:
    """Load Day 5's 17-feature list."""
    return list(joblib.load(FEATURE_COLS_PATH))


@lru_cache(maxsize=1)
def _load_model_y3d():
    """Load Day 5's XGBoost classifier for y_3d (handles NaN natively)."""
    return joblib.load(MODEL_Y3D_PATH)


@lru_cache(maxsize=1)
def _load_model_y5d():
    """Load Day 5's LR pipeline for y_5d (StandardScaler + LogisticRegression)."""
    return joblib.load(MODEL_Y5D_PATH)


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    """Load Day 4's combined feature dataset (273 × 52)."""
    return pd.read_csv(DATASET_PATH, parse_dates=["date_parsed"])


# ---- Per-instance driver extraction --------------------------------------

def _drivers_xgb(model, X_row: pd.DataFrame) -> list[tuple[str, float]]:
    """
    Per-instance SHAP values for XGBoost. Returns sorted [(feature, contribution)].
    Positive contribution = pushes prediction up (riskier).
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    # shap_values shape: (1, n_features) for a single row
    contributions = shap_values[0]
    pairs = list(zip(_load_feature_cols(), contributions))
    pairs.sort(key=lambda fc: abs(fc[1]), reverse=True)
    return pairs


def _drivers_lr(model, X_row: pd.DataFrame) -> list[tuple[str, float]]:
    """
    Per-instance feature contributions for an LR pipeline (StandardScaler + LR).

    For each feature:
        scaled_value = (raw_value - mean) / std
        contribution = scaled_value * coefficient

    This is exact for a linear model — no approximation.
    Positive contribution = pushes log-odds (and probability) up (riskier).
    """
    scaler = model.named_steps["scaler"] if "scaler" in model.named_steps else None
    # Try common scaler step names
    if scaler is None:
        for name, step in model.named_steps.items():
            if hasattr(step, "mean_") and hasattr(step, "scale_"):
                scaler = step
                break

    lr = None
    for name, step in model.named_steps.items():
        if hasattr(step, "coef_"):
            lr = step
            break

    if lr is None:
        raise RuntimeError("Could not find LogisticRegression step in pipeline")

    feature_cols = _load_feature_cols()
    raw_values = X_row.iloc[0].values

    if scaler is not None:
        scaled = (raw_values - scaler.mean_) / scaler.scale_
    else:
        scaled = raw_values

    coefs = lr.coef_[0]
    contributions = scaled * coefs

    pairs = list(zip(feature_cols, contributions))
    pairs.sort(key=lambda fc: abs(fc[1]), reverse=True)
    return pairs


# ---- Topic and question building -----------------------------------------

def _classify_feature(feature: str) -> str:
    """Return 'content', 'meta', or 'unknown' for a feature."""
    if feature in CONTENT_FEATURE_TOPICS:
        return "content"
    if feature in META_FEATURES:
        return "meta"
    return "unknown"


def _topic_for_driver(feature: str, contribution: float) -> str | None:
    """
    Return the direction-appropriate topic string for a content feature.

    Returns None for meta-linguistic or unknown features — they don't go
    into the question.
    """
    if feature not in CONTENT_FEATURE_TOPICS:
        return None
    pos_topic, neg_topic = CONTENT_FEATURE_TOPICS[feature]
    return pos_topic if contribution > 0 else neg_topic


def _build_adaptive_question(
    drivers: list[tuple[str, float]],
    k: int = TOP_DRIVERS_K,
) -> tuple[str, list[dict], list[dict], bool]:
    """
    Build an adaptive question from content drivers only.

    Returns (question, content_drivers_used, all_top_drivers, used_fallback):
      - question: the natural-language question for RAG
      - content_drivers_used: list of {feature, contribution, direction, topic}
                              for the content drivers that built the question
      - all_top_drivers: list of {feature, contribution, direction,
                                  classification, topic_or_none} — the top-k
                                  by magnitude regardless of class, for
                                  transparency
      - used_fallback: True if no content drivers were available and we used
                       the generic question
    """
    # All top-k by magnitude (for transparency in the output)
    all_top = drivers[:k]
    all_top_summary: list[dict] = []
    for feature, contrib in all_top:
        classification = _classify_feature(feature)
        topic = _topic_for_driver(feature, contrib)
        all_top_summary.append({
            "feature": feature,
            "contribution": round(float(contrib), 4),
            "direction": "risky" if contrib > 0 else "safer",
            "classification": classification,
            "topic": topic,  # None for meta/unknown
        })

    # Now scan ALL drivers (not just top-k) for content features.
    # The top-k by raw magnitude might be all meta; we still want to extract
    # the strongest content drivers if any exist below the top-k cutoff.
    content_drivers: list[tuple[str, float, str]] = []
    for feature, contrib in drivers:
        if feature in CONTENT_FEATURE_TOPICS:
            topic = _topic_for_driver(feature, contrib)
            content_drivers.append((feature, float(contrib), topic))
            if len(content_drivers) >= k:
                break

    # Fallback if no content drivers at all
    if not content_drivers:
        return (
            GENERIC_RISK_QUESTION,
            [],
            all_top_summary,
            True,
        )

    # Build the content-driver summary for transparency
    content_summary: list[dict] = [
        {
            "feature": f,
            "contribution": round(c, 4),
            "direction": "risky" if c > 0 else "safer",
            "topic": t,
        }
        for f, c, t in content_drivers
    ]

    # Assemble the question
    topics = [t for _, _, t in content_drivers]
    if len(topics) == 1:
        topic_phrase = topics[0]
    elif len(topics) == 2:
        topic_phrase = f"{topics[0]} and {topics[1]}"
    else:
        topic_phrase = ", ".join(topics[:-1]) + f", and {topics[-1]}"

    question = (
        f"What did management discuss in this earnings call regarding "
        f"{topic_phrase}? Quote specific statements from the transcript."
    )
    return question, content_summary, all_top_summary, False


# ---- Public API ----------------------------------------------------------

def explain_risk(ticker: str, quarter: str) -> dict:
    """
    Produce a risk explanation for (ticker, quarter): scores from both Day 5
    models, the adaptive question built from feature drivers, and two RAG
    explanations (focused + elsewhere).

    Args:
        ticker:  e.g., "AAPL"
        quarter: e.g., "2019-Q3" (must match the format in dataset_v2.csv)

    Returns dict with keys:
        ticker, quarter:                    echoes of inputs
        scores:                             {y_3d: {...}, y_5d: {...}}
                                            each with risk_score, model, top_drivers,
                                            or error if model couldn't predict
        question:                           the adaptive question used for RAG
        question_basis:                     which model's drivers built the question
        explanation_focused:                generate_answer() result (this call only)
        explanation_elsewhere:              generate_answer() result (unfiltered)
        errors:                             list of human-readable error strings
                                            (empty if everything worked)
    """
    errors: list[str] = []

    # Step 1: locate the feature row
    df = _load_dataset()
    mask = (df["ticker"] == ticker) & (df["quarter"] == quarter)
    matching = df[mask]
    if matching.empty:
        return {
            "ticker": ticker,
            "quarter": quarter,
            "scores": {"y_3d": None, "y_5d": None},
            "question": None,
            "question_basis": None,
            "explanation_focused": None,
            "explanation_elsewhere": None,
            "errors": [f"No row in dataset_v2.csv for ({ticker}, {quarter})"],
        }
    row = matching.iloc[[0]]  # keep as DataFrame for sklearn API
    feature_cols = _load_feature_cols()
    X_row = row[feature_cols]
    has_nan = X_row.isna().any().any()

    # Step 2: y_3d XGBoost prediction + SHAP drivers
    scores: dict[str, Any] = {}
    try:
        model_y3d = _load_model_y3d()
        risk_3d = float(model_y3d.predict_proba(X_row)[0, 1])
        drivers_3d = _drivers_xgb(model_y3d, X_row)
        scores["y_3d"] = {
            "risk_score": round(risk_3d, 4),
            "model": "XGBoost (Day 5)",
            "top_drivers": [
                {
                    "feature": f,
                    "contribution": round(float(c), 4),
                    "direction": "risky" if c > 0 else "safer",
                }
                for f, c in drivers_3d[:TOP_DRIVERS_K]
            ],
        }
    except Exception as e:
        scores["y_3d"] = {"error": f"y_3d prediction failed: {e}"}
        errors.append(f"y_3d: {e}")
        drivers_3d = []

    # Step 3: y_5d LR prediction (requires NaN-free row)
    if has_nan:
        scores["y_5d"] = {
            "error": (
                "y_5d (LR) cannot predict on rows with NaN features "
                "(this call has missing FinBERT features — likely an "
                "interview-format call like NFLX/TSLA)."
            )
        }
        errors.append("y_5d: NaN in features, LR skipped")
        drivers_5d: list[tuple[str, float]] = []
    else:
        try:
            model_y5d = _load_model_y5d()
            risk_5d = float(model_y5d.predict_proba(X_row)[0, 1])
            drivers_5d = _drivers_lr(model_y5d, X_row)
            scores["y_5d"] = {
                "risk_score": round(risk_5d, 4),
                "model": "Logistic Regression (Day 5)",
                "top_drivers": [
                    {
                        "feature": f,
                        "contribution": round(float(c), 4),
                        "direction": "risky" if c > 0 else "safer",
                    }
                    for f, c in drivers_5d[:TOP_DRIVERS_K]
                ],
            }
        except Exception as e:
            scores["y_5d"] = {"error": f"y_5d prediction failed: {e}"}
            errors.append(f"y_5d: {e}")
            drivers_5d = []

    # Step 4: pick which model's drivers to base the adaptive question on.
    # Prefer the higher-risk model (more interesting drivers); fall back if missing.
    question_basis: str | None = None
    if drivers_3d and drivers_5d:
        s3 = scores["y_3d"].get("risk_score", 0)
        s5 = scores["y_5d"].get("risk_score", 0)
        if s5 >= s3:
            primary_drivers = drivers_5d
            question_basis = "y_5d (LR)"
        else:
            primary_drivers = drivers_3d
            question_basis = "y_3d (XGBoost)"
    elif drivers_3d:
        primary_drivers = drivers_3d
        question_basis = "y_3d (XGBoost)"
    elif drivers_5d:
        primary_drivers = drivers_5d
        question_basis = "y_5d (LR)"
    else:
        return {
            "ticker": ticker,
            "quarter": quarter,
            "scores": scores,
            "question": None,
            "question_basis": None,
            "explanation_focused": None,
            "explanation_elsewhere": None,
            "errors": errors + ["No model could produce drivers; skipping RAG calls"],
        }

    # Step 5: build adaptive question (content-aware)
    question, content_drivers_used, all_top_drivers, used_fallback = (
        _build_adaptive_question(primary_drivers)
    )

    # Step 6: focused RAG (filter to this ticker + quarter)
    focused = generate_answer(
        question,
        k=RAG_K,
        filter={"$and": [{"ticker": ticker}, {"quarter": quarter}]},
    )

    # Step 7: elsewhere RAG (unfiltered — surfaces similar discussions in other calls)
    elsewhere = generate_answer(question, k=RAG_K)

    return {
        "ticker": ticker,
        "quarter": quarter,
        "scores": scores,
        "question": question,
        "question_basis": question_basis,
        "question_used_fallback": used_fallback,
        "question_drivers": content_drivers_used,  # only content drivers
        "all_top_drivers": all_top_drivers,         # full transparency
        "explanation_focused": focused,
        "explanation_elsewhere": elsewhere,
        "errors": errors,
    }