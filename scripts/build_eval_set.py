"""
scripts/build_eval_set.py — Day 8 Block 1

Builds two CSV artifacts for the Day 8 evaluation:

1. data/eval/eval_worksheet_day8.csv — labeling worksheet.
   One row per (query, retrieved chunk) for the 17 answerable queries.
   You fill the `label` column manually (correct / tangential / wrong / blank).

2. data/eval/eval_set_day8.csv — final eval set template.
   One row per query (20 total).
   `gold_chunks`, `expected_themes`, `reference_answer` columns left blank
   for you to fill after labeling the worksheet.

The off-topic queries (16, 17, 18) skip the worksheet entirely — their
gold_chunks is empty by definition (system should abstain).

Run from project root:
    python -m scripts.build_eval_set
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# --- bootstrap so `python scripts/...` works as well as `python -m scripts...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import retrieve  # noqa: E402

# --- configuration ----------------------------------------------------------

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
WORKSHEET_PATH = EVAL_DIR / "eval_worksheet_day8.csv"
EVAL_SET_PATH = EVAL_DIR / "eval_set_day8.csv"

RETRIEVAL_K = 10  # top-10 per answerable query → label from this pool

# Strata
STRATUM_STANDARD = "standard"
STRATUM_TICKER = "ticker"
STRATUM_TICKER_QUARTER = "ticker_quarter"
STRATUM_SECTION = "section"
STRATUM_OFF_TOPIC = "off_topic"
STRATUM_RISK_INTEGRATION = "risk_integration"

# Low-similarity warning threshold for diagnostic output (not used for filtering here).
# We're capturing top-10 regardless of similarity; this is just to flag queries
# where labeling might be hard.
LOW_SIM_WARN = 0.40

# --- the 20 queries ---------------------------------------------------------

# Each query is a dict. `filter` is None for unfiltered, otherwise a ChromaDB
# filter dict in the same syntax as Day 6/7 retrieve() calls.
# Risk-integration queries (19, 20) do not have a `query` field — they are
# (ticker, quarter) pairs that go through explain_risk(), not generate_answer().

QUERIES = [
    # --- Stratum 1: Standard retrieval (no filter) ---
    {
        "query_id": 1,
        "stratum": STRATUM_STANDARD,
        "query": "What did management say about supply chain disruptions and production delays?",
        "filter": None,
        "is_answerable": True,
        "is_sanity_check": True,  # carries from Day 7 Demo 1
        "notes": "Day 7 Demo 1 carry-over for cross-day stability check",
    },
    {
        "query_id": 2,
        "stratum": STRATUM_STANDARD,
        "query": "How did rising inflation and input costs affect company margins?",
        "filter": None,
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "2021-2022 macro theme; breadth test",
    },
    {
        "query_id": 3,
        "stratum": STRATUM_STANDARD,
        "query": "What did management say about hiring freezes, layoffs, or workforce reductions?",
        "filter": None,
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "2022-2023 tech-sector theme; tone-distinction test",
    },
    {
        "query_id": 4,
        "stratum": STRATUM_STANDARD,
        "query": "What forward-looking risks did management warn investors about?",
        "filter": None,
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "Boilerplate-trap test (Day 6 known limit): legal disclaimers contain these words",
    },
    {
        "query_id": 5,
        "stratum": STRATUM_STANDARD,
        "query": "What did companies discuss about competitive pressure or market share losses?",
        "filter": None,
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "Negative business-framing retrieval test",
    },
    {
        "query_id": 6,
        "stratum": STRATUM_STANDARD,
        "query": "What was management's outlook on demand for the upcoming quarter?",
        "filter": None,
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "Forward guidance; near-universal topic",
    },

    # --- Stratum 2: Ticker-filtered ---
    {
        "query_id": 7,
        "stratum": STRATUM_TICKER,
        "query": "What did Apple discuss about Services revenue and its growth drivers?",
        "filter": {"ticker": "AAPL"},
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "AAPL strong-theme ticker-filter recall test",
    },
    {
        "query_id": 8,
        "stratum": STRATUM_TICKER,
        "query": "What concerns did Beyond Meat management raise about retail sales and consumer demand?",
        "filter": {"ticker": "BYND"},
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "BYND demand softness; clear substantive content",
    },
    {
        "query_id": 9,
        "stratum": STRATUM_TICKER,
        "query": "What did Netflix discuss about subscriber growth, churn, or content strategy?",
        "filter": {"ticker": "NFLX"},
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "Interview-format ticker test (Day 5 finding #6, Day 7 Block 6 NFLX abstention)",
    },
    {
        "query_id": 10,
        "stratum": STRATUM_TICKER,
        "query": "What did Microsoft management say about Azure and cloud growth?",
        "filter": {"ticker": "MSFT"},
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "Standard heavyweight ticker; straightforward content",
    },

    # --- Stratum 3: Ticker + quarter filtered ---
    {
        "query_id": 11,
        "stratum": STRATUM_TICKER_QUARTER,
        "query": "How did iPhone revenue perform and what drove the year-over-year change?",
        "filter": {"$and": [{"ticker": "AAPL"}, {"quarter": "2019-Q3"}]},
        "is_answerable": True,
        "is_sanity_check": True,  # Day 7 Demo 3 carry-over
        "notes": "Day 7 Demo 3 carry-over; known good output",
    },
    {
        "query_id": 12,
        "stratum": STRATUM_TICKER_QUARTER,
        "query": "What operational challenges and headwinds did Beyond Meat face this quarter?",
        "filter": {"$and": [{"ticker": "BYND"}, {"quarter": "2021-Q3"}]},
        "is_answerable": True,
        "is_sanity_check": True,  # Day 7 Block 6 BYND breakthrough
        "notes": "Day 7 Block 6 BYND breakthrough carry-over; stability check",
    },
    {
        "query_id": 13,
        "stratum": STRATUM_TICKER_QUARTER,
        "query": "What did management discuss about cost pressures, layoffs, and macro headwinds?",
        "filter": {"$and": [{"ticker": "MSFT"}, {"quarter": "2023-Q1"}]},
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "Fresh test-set query (post-2022-10-01); unseen by Day 5 models",
    },

    # --- Stratum 4: Section-filtered ---
    {
        "query_id": 14,
        "stratum": STRATUM_SECTION,
        "query": "What questions did analysts press Apple on regarding iPhone demand and sales trajectory?",
        "filter": {"$and": [{"ticker": "AAPL"}, {"section": "qa"}]},
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "Q&A-section retrieval test (Day 5 finding #3: Q&A > prep for risk)",
    },
    {
        "query_id": 15,
        "stratum": STRATUM_SECTION,
        "query": "What did management emphasize in their prepared remarks about strategic priorities and capital allocation?",
        "filter": {"section": "prep"},
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "Prep-section retrieval test; capital allocation is a prep-vs-Q&A divider",
    },

    # --- Stratum 5: Off-topic abstention test ---
    {
        "query_id": 16,
        "stratum": STRATUM_OFF_TOPIC,
        "query": "What is the best recipe for sourdough bread starter?",
        "filter": None,
        "is_answerable": False,
        "is_sanity_check": True,  # Day 7 Demo 5 carry-over
        "notes": "Day 7 Demo 5 carry-over; abstention anchor",
    },
    {
        "query_id": 17,
        "stratum": STRATUM_OFF_TOPIC,
        "query": "How do I train for a marathon?",
        "filter": None,
        "is_answerable": False,
        "is_sanity_check": False,
        "notes": "Fresh off-topic; abstention generalization test",
    },
    {
        "query_id": 18,
        "stratum": STRATUM_OFF_TOPIC,
        "query": "What are the best places to visit in Tokyo for first-time travelers?",
        "filter": None,
        "is_answerable": False,
        "is_sanity_check": False,
        "notes": "Fresh off-topic; far-domain test",
    },

    # --- Stratum 6: Risk-integration test ---
    # These do not have a `query` — they go through explain_risk(ticker, quarter)
    # and are evaluated by the eval runner, not the worksheet pipeline.
    {
        "query_id": 19,
        "stratum": STRATUM_RISK_INTEGRATION,
        "query": "",  # not used; integration is by (ticker, quarter)
        "filter": None,
        "ticker": "BYND",
        "quarter": "2021-Q3",
        "is_answerable": True,
        "is_sanity_check": True,  # Day 7 Block 6 breakthrough
        "notes": "Day 7 Block 6 BYND integration carry-over; pre-Day-5-test-cutoff",
    },
    {
        "query_id": 20,
        "stratum": STRATUM_RISK_INTEGRATION,
        "query": "",
        "filter": None,
        "ticker": "MSFT",
        "quarter": "2023-Q1",
        "is_answerable": True,
        "is_sanity_check": False,
        "notes": "Fresh test-set integration (post-2022-10-01); unseen by Day 5 models",
    },
]


# --- helpers ---------------------------------------------------------------

def _filter_to_str(filter_obj) -> str:
    """Serialize a filter dict to a CSV-safe JSON string. Empty string for None."""
    if filter_obj is None:
        return ""
    return json.dumps(filter_obj, separators=(",", ":"))


def _is_worksheet_query(q: dict) -> bool:
    """True if this query gets a worksheet row (answerable, not risk-integration)."""
    return q["is_answerable"] and q["stratum"] != STRATUM_RISK_INTEGRATION


# --- build the worksheet ----------------------------------------------------

def build_worksheet() -> tuple[list[dict], dict]:
    """
    For each answerable, non-integration query, retrieve top-K and write rows.
    Returns (rows, stats).
    """
    rows = []
    stats = {
        "queries_run": 0,
        "queries_skipped_offtopic": 0,
        "queries_skipped_integration": 0,
        "low_sim_queries": [],   # (query_id, max_sim) where max_sim < LOW_SIM_WARN
        "errors": [],
        "elapsed_s": 0.0,
    }

    t0 = time.perf_counter()

    for q in QUERIES:
        if not _is_worksheet_query(q):
            if q["stratum"] == STRATUM_OFF_TOPIC:
                stats["queries_skipped_offtopic"] += 1
            elif q["stratum"] == STRATUM_RISK_INTEGRATION:
                stats["queries_skipped_integration"] += 1
            continue

        qid = q["query_id"]
        query_text = q["query"]
        filter_obj = q["filter"]

        try:
            results = retrieve(query_text, k=RETRIEVAL_K, filter=filter_obj)
        except Exception as exc:
            stats["errors"].append((qid, repr(exc)))
            print(f"[query {qid}] ERROR: {exc!r}")
            continue

        stats["queries_run"] += 1

        if not results:
            print(f"[query {qid}] WARNING: 0 results returned (filter may be too strict)")
            continue

        max_sim = max(r.get("similarity", 0.0) for r in results)
        if max_sim < LOW_SIM_WARN:
            stats["low_sim_queries"].append((qid, round(max_sim, 3)))

        for rank, r in enumerate(results, start=1):
            rows.append({
                "query_id": qid,
                "query": query_text,
                "filter": _filter_to_str(filter_obj),
                "rank": rank,
                "chunk_id": r["chunk_id"],
                "similarity": round(r.get("similarity", 0.0), 4),
                "chunk_text": r["text"],
                "label": "",   # YOU fill: correct / tangential / wrong / blank
            })

        print(f"[query {qid}] retrieved {len(results)} chunks "
              f"(sim range {min(r['similarity'] for r in results):.3f}–{max_sim:.3f})")

    stats["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return rows, stats


# --- build the eval set template -------------------------------------------

def build_eval_set_template() -> list[dict]:
    """One row per query. Gold/themes/reference left blank for manual fill."""
    rows = []
    for q in QUERIES:
        rows.append({
            "query_id": q["query_id"],
            "stratum": q["stratum"],
            "query": q["query"],
            "filter": _filter_to_str(q["filter"]),
            "ticker": q.get("ticker", ""),     # only set for integration queries
            "quarter": q.get("quarter", ""),   # only set for integration queries
            "is_answerable": q["is_answerable"],
            "is_sanity_check": q["is_sanity_check"],
            "gold_chunks": "",         # YOU fill: pipe-separated chunk_ids
            "expected_themes": "",     # YOU fill: pipe-separated keywords
            "reference_answer": "",    # YOU fill: 3-5 sentences in your own words
            "notes": q["notes"],
        })
    return rows


# --- writers ----------------------------------------------------------------

def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


# --- main -------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Day 8 Block 1 — build_eval_set")
    print("=" * 70)
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Worksheet out: {WORKSHEET_PATH}")
    print(f"Eval set out:  {EVAL_SET_PATH}")
    print(f"Total queries: {len(QUERIES)}")
    print()

    print(">> Running retrieval for answerable, non-integration queries...")
    print(f"   (First retrieval will take ~30s — Arctic model cache warmup)")
    print()

    worksheet_rows, stats = build_worksheet()
    print()
    print(f">> Retrieval complete in {stats['elapsed_s']}s")
    print(f"   Queries retrieved:      {stats['queries_run']}")
    print(f"   Off-topic skipped:      {stats['queries_skipped_offtopic']}")
    print(f"   Integration skipped:    {stats['queries_skipped_integration']}")
    print(f"   Worksheet rows written: {len(worksheet_rows)} "
          f"(expected ~{stats['queries_run'] * RETRIEVAL_K})")
    if stats["low_sim_queries"]:
        print(f"   ⚠ Low-similarity queries (max_sim < {LOW_SIM_WARN}):")
        for qid, max_sim in stats["low_sim_queries"]:
            print(f"     query {qid}: max_sim={max_sim}")
    if stats["errors"]:
        print(f"   ❌ Errors:")
        for qid, err in stats["errors"]:
            print(f"     query {qid}: {err}")
    print()

    print(">> Writing worksheet CSV...")
    worksheet_fieldnames = [
        "query_id", "query", "filter", "rank",
        "chunk_id", "similarity", "chunk_text", "label",
    ]
    write_csv(worksheet_rows, WORKSHEET_PATH, worksheet_fieldnames)
    print(f"   ✓ {WORKSHEET_PATH}  ({len(worksheet_rows)} rows)")

    print(">> Writing eval set template CSV...")
    eval_set_rows = build_eval_set_template()
    eval_set_fieldnames = [
        "query_id", "stratum", "query", "filter", "ticker", "quarter",
        "is_answerable", "is_sanity_check",
        "gold_chunks", "expected_themes", "reference_answer", "notes",
    ]
    write_csv(eval_set_rows, EVAL_SET_PATH, eval_set_fieldnames)
    print(f"   ✓ {EVAL_SET_PATH}  ({len(eval_set_rows)} rows)")
    print()

    # Stratum breakdown for sanity
    print(">> Eval set composition:")
    from collections import Counter
    strata = Counter(q["stratum"] for q in QUERIES)
    for s, n in sorted(strata.items()):
        print(f"   {s:<22} {n}")
    print(f"   {'TOTAL':<22} {sum(strata.values())}")
    print()
    print(">> Done. Next: label the worksheet, then fill eval_set_day8.csv.")


if __name__ == "__main__":
    main()