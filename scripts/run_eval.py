"""
scripts/run_eval.py — Day 8 evaluation runner

Reads the labeled eval set (data/eval/eval_set_day8.csv), runs the RAG system
end-to-end on every query, scores against gold labels, and writes three artifacts:

  data/eval/eval_results_day8.csv    — per-query metrics
  data/eval/eval_per_chunk_day8.csv  — per (query, retrieved-chunk) flags
  data/eval/eval_summary_day8.md     — aggregate findings (recruiter-facing)

Run from project root:
    python -m scripts.run_eval
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# --- bootstrap ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.generator import generate_answer       # noqa: E402
from src.rag.risk_explainer import explain_risk     # noqa: E402

# --- paths ---
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_SET_PATH = EVAL_DIR / "eval_set_day8.csv"
RESULTS_PATH = EVAL_DIR / "eval_results_day8.csv"
PER_CHUNK_PATH = EVAL_DIR / "eval_per_chunk_day8.csv"
SUMMARY_PATH = EVAL_DIR / "eval_summary_day8.md"

# Retrieve k=10 so recall@10 is measurable; generation still threshold-filters to 5.
RETRIEVAL_K = 10

# Delay between queries to stay under Gemini free-tier per-minute rate limit.
# Does NOT bypass the daily quota.
INTER_QUERY_DELAY_SECONDS = 8

STRATUM_RISK_INTEGRATION = "risk_integration"
STRATUM_OFF_TOPIC = "off_topic"


# ---------------------------------------------------------------------------
# Loading & parsing the eval set
# ---------------------------------------------------------------------------

def _parse_pipe(value: str) -> list[str]:
    """Parse a pipe-separated cell into a list. Empty cell -> []."""
    if not value or (isinstance(value, float)):
        return []
    value = str(value).strip()
    if not value:
        return []
    return [v.strip() for v in value.split("|") if v.strip()]


def _parse_filter(value: str) -> dict | None:
    """Parse the JSON filter cell into a dict. Empty -> None."""
    if not value:
        return None
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return None
    return json.loads(value)


def load_eval_set() -> list[dict]:
    """Load the eval set CSV into a list of normalized query dicts."""
    rows = []
    with EVAL_SET_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({
                "query_id": int(r["query_id"]),
                "stratum": r["stratum"],
                "query": r["query"],
                "filter": _parse_filter(r.get("filter", "")),
                "ticker": r.get("ticker", "").strip(),
                "quarter": r.get("quarter", "").strip(),
                "is_answerable": str(r["is_answerable"]).strip().lower() == "true",
                "is_sanity_check": str(r["is_sanity_check"]).strip().lower() == "true",
                "gold_chunks": _parse_pipe(r.get("gold_chunks", "")),
                "expected_themes": _parse_pipe(r.get("expected_themes", "")),
                "reference_answer": r.get("reference_answer", ""),
                "notes": r.get("notes", ""),
            })
    return rows


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def recall_at_k(gold: list[str], retrieved_ids: list[str], k: int) -> float | None:
    """|gold ∩ top-k| / |gold|. None if no gold (undefined)."""
    if not gold:
        return None
    topk = set(retrieved_ids[:k])
    hits = sum(1 for g in gold if g in topk)
    return round(hits / len(gold), 3)


def precision_at_k(gold: list[str], retrieved_ids: list[str], k: int) -> float | None:
    """|gold ∩ top-k| / k. None if fewer than 1 retrieved."""
    if not retrieved_ids:
        return None
    topk = retrieved_ids[:k]
    if not topk:
        return None
    gold_set = set(gold)
    hits = sum(1 for r in topk if r in gold_set)
    return round(hits / len(topk), 3)


def mean_gold_similarity(gold: list[str], retrieved: list[dict]) -> float | None:
    """Mean similarity of gold chunks that were retrieved. None if none retrieved."""
    sim_by_id = {c["chunk_id"]: c["similarity"] for c in retrieved}
    sims = [sim_by_id[g] for g in gold if g in sim_by_id]
    if not sims:
        return None
    return round(sum(sims) / len(sims), 3)


def theme_coverage(themes: list[str], answer: str) -> float | None:
    """Fraction of themes appearing as case-insensitive substrings in answer.
    None if no themes defined."""
    if not themes:
        return None
    answer_lower = answer.lower()
    hits = sum(1 for t in themes if t.lower() in answer_lower)
    return round(hits / len(themes), 3)


def _fmt(value) -> str:
    """Format a value for CSV: None -> '' , else str."""
    if value is None:
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Per-query runners
# ---------------------------------------------------------------------------

def run_standard_query(q: dict) -> tuple[dict, list[dict]]:
    """
    Run an answerable or off-topic query through generate_answer().
    Returns (result_row, per_chunk_rows).
    """
    t0 = time.perf_counter()
    result = generate_answer(q["query"], k=RETRIEVAL_K, filter=q["filter"])
    latency = round(time.perf_counter() - t0, 2)

    retrieved = result["chunks_retrieved"]
    retrieved_ids = [c["chunk_id"] for c in retrieved]
    gold = q["gold_chunks"]

    # Retrieval metrics (skip for off-topic — gold is empty by design)
    is_offtopic = q["stratum"] == STRATUM_OFF_TOPIC
    r5 = recall_at_k(gold, retrieved_ids, 5)
    r10 = recall_at_k(gold, retrieved_ids, 10)
    p5 = precision_at_k(gold, retrieved_ids, 5)
    gold_sim = mean_gold_similarity(gold, retrieved)

    # Generation metrics
    abstained = result["abstained"]
    api_failed = result.get("api_failed", False)
    faith = result["faithfulness"]
    if api_failed:
        # API failure (e.g. quota exhausted): generation metrics are
        # NOT MEASURED, not zero. Record blanks so aggregates aren't polluted.
        faith_mean = None
        faith_unsupported = None
        cite_valid = None
        n_citations = None
        tcov = None
    else:
        faith_mean = faith["mean_overlap"] if faith else None
        faith_unsupported = faith["n_unsupported"] if faith else None
        cite_valid = result["citation_health"]["valid"]
        n_citations = len(result["citations"])
        tcov = theme_coverage(q["expected_themes"], result["answer"])

    top_sim = retrieved[0]["similarity"] if retrieved else None

    # For off-topic: "pass" = abstained. For answerable: "pass" = not abstained.
    # API failure = not a pass and not a fail of the SYSTEM — it's unmeasured.
    if is_offtopic:
        passed = abstained
    elif api_failed:
        passed = None  # unmeasured — don't count as system pass or fail
    else:
        passed = not abstained

    usage = result.get("usage", {})
    tokens_total = usage.get("tokens_total")

    result_row = {
        "query_id": q["query_id"],
        "stratum": q["stratum"],
        "is_answerable": q["is_answerable"],
        "is_sanity_check": q["is_sanity_check"],
        "abstained": abstained,
        "api_failed": api_failed,
        "passed": passed,
        "recall_at_5": _fmt(r5),
        "recall_at_10": _fmt(r10),
        "precision_at_5": _fmt(p5),
        "mean_gold_sim": _fmt(gold_sim),
        "top_sim": _fmt(round(top_sim, 3) if top_sim is not None else None),
        "theme_coverage": _fmt(tcov),
        "faithfulness_mean": _fmt(faith_mean),
        "faith_unsupported": _fmt(faith_unsupported),
        "citation_valid": cite_valid,
        "n_citations": n_citations,
        "n_gold": len(gold),
        "n_retrieved": len(retrieved),
        "latency_s": latency,
        "tokens_total": _fmt(tokens_total),
        # integration-only columns (blank here)
        "y_3d": "", "y_5d": "", "question_basis": "",
        "focused_abstained": "", "focused_theme_cov": "",
        "elsewhere_abstained": "", "elsewhere_theme_cov": "",
        "notes": q["notes"],
    }

    # Per-chunk rows
    gold_set = set(gold)
    per_chunk_rows = []
    for rank, c in enumerate(retrieved, start=1):
        per_chunk_rows.append({
            "query_id": q["query_id"],
            "rank": rank,
            "chunk_id": c["chunk_id"],
            "similarity": round(c["similarity"], 4),
            "was_gold": c["chunk_id"] in gold_set,
            "in_top5": rank <= 5,
        })

    return result_row, per_chunk_rows


def run_integration_query(q: dict) -> tuple[dict, list[dict]]:
    """
    Run a risk-integration query through explain_risk().
    Returns (result_row, per_chunk_rows=[]).
    """
    t0 = time.perf_counter()
    result = explain_risk(q["ticker"], q["quarter"])
    latency = round(time.perf_counter() - t0, 2)

    # Extract scores (three-way guard: None / error / dict)
    def _score(horizon: str):
        s = result["scores"].get(horizon)
        if s is None:
            return None
        if "error" in s:
            return f"ERROR:{s['error'][:30]}"
        return s["risk_score"]

    y3 = _score("y_3d")
    y5 = _score("y_5d")

    # Focused / elsewhere explanation scoring
    def _explanation_metrics(exp: dict | None):
        if exp is None:
            return ("", "")  # (abstained, theme_cov)
        abstained = exp["abstained"]
        if abstained:
            return (True, "")
        tcov = theme_coverage(q["expected_themes"], exp["answer"])
        return (False, _fmt(tcov))

    foc_abstained, foc_tcov = _explanation_metrics(result["explanation_focused"])
    els_abstained, els_tcov = _explanation_metrics(result["explanation_elsewhere"])

    # Aggregate tokens across both RAG calls
    tokens = 0
    for exp in (result["explanation_focused"], result["explanation_elsewhere"]):
        if exp and exp.get("usage"):
            tokens += exp["usage"].get("tokens_total", 0) or 0

    # "passed" for integration = at least the focused explanation didn't abstain
    foc = result["explanation_focused"]
    passed = bool(foc and not foc["abstained"])

    result_row = {
        "query_id": q["query_id"],
        "stratum": q["stratum"],
        "is_answerable": q["is_answerable"],
        "is_sanity_check": q["is_sanity_check"],
        "abstained": foc_abstained if isinstance(foc_abstained, bool) else "",
        "api_failed": "",
        "passed": passed,
        "recall_at_5": "", "recall_at_10": "", "precision_at_5": "",
        "mean_gold_sim": "", "top_sim": "",
        "theme_coverage": "",
        "faithfulness_mean": _fmt(
            foc["faithfulness"]["mean_overlap"]
            if foc and not foc["abstained"] and foc["faithfulness"] else None
        ),
        "faith_unsupported": _fmt(
            foc["faithfulness"]["n_unsupported"]
            if foc and not foc["abstained"] and foc["faithfulness"] else None
        ),
        "citation_valid": (
            foc["citation_health"]["valid"] if foc and not foc["abstained"] else ""
        ),
        "n_citations": (
            len(foc["citations"]) if foc and not foc["abstained"] else ""
        ),
        "n_gold": "", "n_retrieved": "",
        "latency_s": latency,
        "tokens_total": _fmt(tokens if tokens else None),
        "y_3d": _fmt(y3),
        "y_5d": _fmt(y5),
        "question_basis": result.get("question_basis", ""),
        "focused_abstained": _fmt(foc_abstained),
        "focused_theme_cov": foc_tcov,
        "elsewhere_abstained": _fmt(els_abstained),
        "elsewhere_theme_cov": els_tcov,
        "notes": q["notes"],
    }

    return result_row, []


# ---------------------------------------------------------------------------
# Summary doc
# ---------------------------------------------------------------------------

def write_summary(results: list[dict]) -> None:
    """Compose the recruiter-facing aggregate findings doc."""
    answerable = [r for r in results
                  if r["stratum"] not in (STRATUM_OFF_TOPIC, STRATUM_RISK_INTEGRATION)]
    offtopic = [r for r in results if r["stratum"] == STRATUM_OFF_TOPIC]
    integration = [r for r in results if r["stratum"] == STRATUM_RISK_INTEGRATION]

    def _floats(rows, key):
        out = []
        for r in rows:
            v = r.get(key, "")
            if v not in ("", None):
                try:
                    out.append(float(v))
                except (ValueError, TypeError):
                    pass
        return out

    def _mean(vals):
        return round(sum(vals) / len(vals), 3) if vals else None

    r5s = _floats(answerable, "recall_at_5")
    r10s = _floats(answerable, "recall_at_10")
    p5s = _floats(answerable, "precision_at_5")
    faiths = _floats(answerable, "faithfulness_mean")
    tcovs = _floats(answerable, "theme_coverage")

    n_abstain_answerable = sum(1 for r in answerable if r["abstained"] is True)
    n_offtopic_correct = sum(1 for r in offtopic if r["passed"] is True)
    n_cite_valid = sum(1 for r in answerable if r["citation_valid"] is True)

    lines = []
    lines.append("# Day 8 Evaluation Results\n")
    lines.append("Automated evaluation of the Risk Radar RAG pipeline against a "
                 "20-query hand-labeled eval set.\n")
    lines.append("**Labeling provenance:** gold labels drafted by Claude with "
                 "human spot-check. Same-system labeling bias possible; see "
                 "labeling summary. LLM-as-judge (Block 4) provides an "
                 "independent generation-quality signal.\n")

    lines.append("## Aggregate metrics (answerable queries)\n")
    lines.append(f"- Mean recall@5: **{_mean(r5s)}** (n={len(r5s)})")
    lines.append(f"- Mean recall@10: **{_mean(r10s)}** (n={len(r10s)})")
    lines.append(f"- Mean precision@5: **{_mean(p5s)}**")
    lines.append(f"- Mean faithfulness (lexical overlap): **{_mean(faiths)}**")
    lines.append(f"- Mean theme coverage: **{_mean(tcovs)}**")
    lines.append(f"- Citation health valid: **{n_cite_valid}/{len(answerable)}**")
    lines.append(f"- Unexpected abstentions (answerable): **{n_abstain_answerable}**\n")

    lines.append("## Abstention behavior (off-topic queries)\n")
    lines.append(f"- Correct abstentions: **{n_offtopic_correct}/{len(offtopic)}**")
    for r in offtopic:
        status = "PASS abstained" if r["passed"] else "FAIL did NOT abstain"
        lines.append(f"  - q{r['query_id']}: {status} (top_sim={r['top_sim']})")
    lines.append("")

    lines.append("## Integration (risk explainer)\n")
    for r in integration:
        lines.append(f"- q{r['query_id']}: y_3d={r['y_3d']}, y_5d={r['y_5d']}, "
                     f"focused_abstained={r['focused_abstained']}, "
                     f"focused_theme_cov={r['focused_theme_cov']}, "
                     f"elsewhere_abstained={r['elsewhere_abstained']}")
    lines.append("")

    lines.append("## Per-query detail\n")
    lines.append("| q | stratum | recall@5 | recall@10 | prec@5 | faith | theme_cov | abstain | cite_ok | lat(s) |")
    lines.append("|---|---------|----------|-----------|--------|-------|-----------|---------|---------|--------|")
    for r in results:
        lines.append(
            f"| {r['query_id']} | {r['stratum']} | {r['recall_at_5']} | "
            f"{r['recall_at_10']} | {r['precision_at_5']} | {r['faithfulness_mean']} | "
            f"{r['theme_coverage']} | {r['abstained']} | {r['citation_valid']} | "
            f"{r['latency_s']} |"
        )
    lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

RESULT_FIELDNAMES = [
    "query_id", "stratum", "is_answerable", "is_sanity_check",
    "abstained", "api_failed", "passed",
    "recall_at_5", "recall_at_10", "precision_at_5", "mean_gold_sim", "top_sim",
    "theme_coverage", "faithfulness_mean", "faith_unsupported",
    "citation_valid", "n_citations", "n_gold", "n_retrieved",
    "latency_s", "tokens_total",
    "y_3d", "y_5d", "question_basis",
    "focused_abstained", "focused_theme_cov",
    "elsewhere_abstained", "elsewhere_theme_cov",
    "notes",
]

PER_CHUNK_FIELDNAMES = [
    "query_id", "rank", "chunk_id", "similarity", "was_gold", "in_top5",
]


def main() -> None:
    print("=" * 70)
    print("Day 8 — run_eval")
    print("=" * 70)
    print(f"Eval set:    {EVAL_SET_PATH}")
    print(f"Results out: {RESULTS_PATH}")
    print()

    queries = load_eval_set()
    print(f"Loaded {len(queries)} queries.")
    print("First query triggers model + ChromaDB load (~30s)...")
    print()

    all_results = []
    all_per_chunk = []

    t_start = time.perf_counter()
    for idx, q in enumerate(queries):
        qid = q["query_id"]
        label = q["query"][:55] if q["query"] else f"{q['ticker']} {q['quarter']}"
        print(f"[q{qid:>2}] {q['stratum']:<18} {label}...", flush=True)

        # Space out queries to respect the per-minute rate limit (skip before first).
        if idx > 0:
            time.sleep(INTER_QUERY_DELAY_SECONDS)

        try:
            if q["stratum"] == STRATUM_RISK_INTEGRATION:
                row, chunk_rows = run_integration_query(q)
            else:
                row, chunk_rows = run_standard_query(q)
        except Exception as exc:
            print(f"       ERROR: {exc!r}")
            row = {fn: "" for fn in RESULT_FIELDNAMES}
            row.update({
                "query_id": qid, "stratum": q["stratum"],
                "notes": f"RUNNER ERROR: {exc!r}",
            })
            chunk_rows = []

        all_results.append(row)
        all_per_chunk.extend(chunk_rows)

        # Brief inline status
        if row.get("recall_at_5") not in ("", None):
            print(f"       recall@5={row['recall_at_5']} "
                  f"faith={row['faithfulness_mean']} "
                  f"abstain={row['abstained']} {row['latency_s']}s")
        else:
            print(f"       abstain={row['abstained']} passed={row['passed']} "
                  f"{row['latency_s']}s")

    elapsed = round(time.perf_counter() - t_start, 1)
    print()
    print(f"All queries done in {elapsed}s.")

    # Write results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"OK {RESULTS_PATH}  ({len(all_results)} rows)")

    with PER_CHUNK_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_CHUNK_FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(all_per_chunk)
    print(f"OK {PER_CHUNK_PATH}  ({len(all_per_chunk)} rows)")

    write_summary(all_results)
    print(f"OK {SUMMARY_PATH}")

    print()
    print(">> Done. Review eval_summary_day8.md for aggregate findings.")


if __name__ == "__main__":
    main()