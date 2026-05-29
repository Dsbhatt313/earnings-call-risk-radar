import json
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date

from src.rag.risk_explainer import explain_risk
from src.rag.generator import generate_answer

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_v2.csv"
USAGE_PATH = PROJECT_ROOT / "data" / "gemini_usage.json"

FEATURE_LABELS = {
    "negative_count_zscore": "Negative language frequency",
    "uncertainty_count_zscore": "Uncertainty language",
    "litigious_count_zscore": "Legal / litigious language",
    "forward_looking_count_zscore": "Forward-looking statements",
    "prep_pos_mean_zscore": "Positive tone (prepared remarks)",
    "prep_neu_mean_zscore": "Neutral tone (prepared remarks)",
    "prep_neg_mean_zscore": "Negative tone (prepared remarks)",
    "qa_pos_mean_zscore": "Positive tone (Q&A)",
    "qa_neu_mean_zscore": "Neutral tone (Q&A)",
    "qa_neg_mean_zscore": "Negative tone (Q&A)",
    "word_count_zscore": "Total word count",
    "sentence_count_zscore": "Sentence count",
    "avg_sentence_length_zscore": "Average sentence length",
    "flesch_kincaid_grade_zscore": "Reading-grade complexity",
    "numeric_density_zscore": "Numeric density",
    "prep_n_chunks_zscore": "Prepared-remarks length",
    "qa_n_chunks_zscore": "Q&A length",
}

# Free-tier reference point only — display, never a hard block.
FREE_TIER_DAILY_REFERENCE = 250

st.set_page_config(page_title="Earnings Call Risk Radar", layout="wide")

# --------------------------------------------------------------------------
# Quota counter (persisted to a small JSON file, reset daily)
# --------------------------------------------------------------------------
def _read_usage() -> int:
    today = date.today().isoformat()
    try:
        data = json.loads(USAGE_PATH.read_text())
        if data.get("date") == today:
            return int(data.get("calls", 0))
    except Exception:
        pass
    return 0


def _add_usage(n: int) -> None:
    today = date.today().isoformat()
    current = _read_usage()
    try:
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        USAGE_PATH.write_text(json.dumps({"date": today, "calls": current + n}))
    except Exception:
        pass


def _count_calls(exp) -> int:
    """A real Gemini call happened only if it wasn't an abstention or failure."""
    if not exp:
        return 0
    if exp.get("abstained") or exp.get("api_failed"):
        return 0
    return 1

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data
def load_ticker_quarters() -> dict[str, list[str]]:
    df = pd.read_csv(DATASET_PATH, usecols=["ticker", "quarter"])
    mapping: dict[str, list[str]] = {}
    for t, q in df.drop_duplicates().itertuples(index=False):
        mapping.setdefault(str(t), []).append(str(q))
    for t in mapping:
        mapping[t] = sorted(set(mapping[t]))
    return dict(sorted(mapping.items()))

# --------------------------------------------------------------------------
# Render helpers
# --------------------------------------------------------------------------
def _format_driver(d: dict) -> str:
    label = FEATURE_LABELS.get(d.get("feature", ""), d.get("feature", ""))
    verb = "raises risk" if d.get("direction") == "risky" else "lowers risk"
    return f"**{label}** — {verb} (contribution {d.get('contribution', 0.0):+.3f})"


def _render_one_score(col, title: str, sc) -> None:
    with col:
        st.subheader(title)
        if sc is None:
            st.caption("No data for this call.")
            return
        if "error" in sc:
            st.warning(sc["error"])
            return
        st.metric("Risk score", f"{sc['risk_score']:.3f}")
        st.caption(f"Model: {sc.get('model', '')}")
        for d in (sc.get("top_drivers") or []):
            st.markdown("- " + _format_driver(d))


def _render_faithfulness(exp) -> None:
    f = exp.get("faithfulness")
    if not f:
        return
    st.caption(
        f"Faithfulness (lexical): {f.get('mean_overlap', 0.0):.2f} mean overlap "
        f"across {f.get('n_claims_checked', 0)} cited claims; "
        f"{f.get('n_unsupported', 0)} flagged as weakly supported. "
        "(Tripwire metric — the LLM-judge is the stronger check, pending quota.)"
    )


def _render_explanation(label: str, exp) -> None:
    if exp is None:
        st.caption(f"{label}: not available.")
        return
    if exp.get("api_failed"):
        st.warning(
            f"{label}: the Gemini call failed — most likely the free-tier daily "
            "quota. Try again after it resets."
        )
        return
    if exp.get("abstained"):
        st.info(
            f"{label}: the system abstained — retrieved chunks weren't relevant "
            "or confident enough. This is intended behaviour, not a failure."
        )
        return
    st.markdown(exp.get("answer", "_(no answer text)_"))
    citations = exp.get("citations") or {}
    if citations:
        with st.expander(f"Cited sources ({len(citations)})"):
            for marker, chunk_id in citations.items():
                st.markdown(f"- `{marker}` → `{chunk_id}`")
    health = exp.get("citation_health") or {}
    if health.get("valid") is False:
        warns = health.get("warnings") or []
        st.caption("Citation check flagged: " + ("; ".join(warns) if warns else "see logs"))
    _render_faithfulness(exp)


def render_risk_result(result: dict) -> None:
    st.subheader(f"Risk for {result['ticker']} {result['quarter']}")
    scores = result.get("scores") or {}
    col_a, col_b = st.columns(2)
    _render_one_score(col_a, "3-day horizon", scores.get("y_3d"))
    _render_one_score(col_b, "5-day horizon", scores.get("y_5d"))

    question = result.get("question")
    if question:
        st.divider()
        st.markdown("**Adaptive question** (auto-built from the top risk drivers):")
        st.markdown(f"> {question}")
        basis = result.get("question_basis")
        if basis:
            note = f"Built from {basis} drivers."
            if result.get("question_used_fallback"):
                note += " (Generic fallback — no content drivers available.)"
            st.caption(note)

    st.divider()
    st.markdown("### Evidence from this call")
    _render_explanation("This call", result.get("explanation_focused"))
    st.markdown("### Similar discussion in other calls")
    _render_explanation("Other calls", result.get("explanation_elsewhere"))

    errs = result.get("errors") or []
    if errs:
        with st.expander("Diagnostics / notes"):
            for e in errs:
                st.caption(e)

# --------------------------------------------------------------------------
# Sidebar — usage counter
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Gemini usage today")
    used = _read_usage()
    st.metric("Calls used (approx.)", used)
    st.caption(
        f"Free-tier reference: ~{FREE_TIER_DAILY_REFERENCE}/day for "
        "gemini-2.5-flash. A risk analysis = 2 calls; a question = 1. "
        "Counter resets at local midnight. Abstentions and failures aren't counted."
    )
    if used >= FREE_TIER_DAILY_REFERENCE * 0.8:
        st.warning("Approaching the free-tier daily reference. Answers may start failing.")

# --------------------------------------------------------------------------
# Header + tabs
# --------------------------------------------------------------------------
st.title("Earnings Call Risk Radar")
st.caption(
    "FinBERT sentiment -> XGBoost / LR risk classifier -> RAG over 273 earnings "
    "call transcripts. Grounded, cited, and honest about what it doesn't know."
)

tab_risk, tab_ask = st.tabs(["Risk Score", "Ask the Filing"])

# --------------------------------------------------------------------------
# Tab 1 — Risk Score
# --------------------------------------------------------------------------
with tab_risk:
    st.header("Risk Score")
    st.write(
        "Pick a company and quarter. The app scores risk with both Day 5 models, "
        "explains what drove it, and pulls grounded evidence from the transcript."
    )
    tq = load_ticker_quarters()
    tickers = list(tq.keys())

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.selectbox("Company (ticker)", tickers, key="risk_ticker")
    with col2:
        quarter = st.selectbox("Quarter", tq.get(ticker, []), key="risk_quarter")

    if st.button("Analyze risk", type="primary"):
        cache = st.session_state.setdefault("risk_cache", {})
        key = (ticker, quarter)
        if key in cache:
            st.session_state.risk_result = cache[key]
            st.toast("Loaded from this session — no Gemini call made.")
        else:
            with st.spinner("Scoring risk and retrieving evidence (first run loads models, ~15-30s)..."):
                result = explain_risk(ticker, quarter)
            calls = _count_calls(result.get("explanation_focused")) + \
                    _count_calls(result.get("explanation_elsewhere"))
            _add_usage(calls)
            cache[key] = result
            st.session_state.risk_result = result
            st.rerun()  # refresh sidebar counter immediately

    if st.session_state.get("risk_result"):
        st.divider()
        render_risk_result(st.session_state.risk_result)

# --------------------------------------------------------------------------
# Tab 2 — Ask the Filing
# --------------------------------------------------------------------------
with tab_ask:
    st.header("Ask the Filing")
    st.write(
        "Ask any question about the transcripts. Optionally narrow to one company. "
        "The system answers only from retrieved text, cites its sources, and "
        "abstains when it can't ground an answer."
    )
    tq2 = load_ticker_quarters()
    scope_options = ["All companies"] + list(tq2.keys())

    question = st.text_input(
        "Your question",
        placeholder="e.g. What did management say about supply chain disruptions?",
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        scope = st.selectbox("Scope", scope_options, key="ask_scope")
    with c2:
        thinking = st.toggle("Gemini thinking", value=False,
                             help="Slower, sometimes better reasoning. Uses more tokens.")

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            filt = None if scope == "All companies" else {"ticker": scope}
            cache = st.session_state.setdefault("ask_cache", {})
            key = (question.strip(), scope, thinking)
            if key in cache:
                st.session_state.ask_result = cache[key]
                st.toast("Loaded from this session — no Gemini call made.")
            else:
                with st.spinner("Retrieving chunks and asking Gemini..."):
                    result = generate_answer(question, k=5, filter=filt, thinking=thinking)
                _add_usage(_count_calls(result))
                cache[key] = result
                st.session_state.ask_result = result
                st.rerun()

    if st.session_state.get("ask_result"):
        st.divider()
        _render_explanation("Answer", st.session_state.ask_result)