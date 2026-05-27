"""
scripts/demo_generation.py

Day 7 - Block 4: End-to-end RAG demo. Runs the same 4 queries as Day 6's
demo_retrieval.py through the full generate_answer() pipeline, plus one
off-topic baseline query (Demo 5).

Usage:
    python scripts\\demo_generation.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.generator import (
    generate_answer,
    SIMILARITY_THRESHOLD,
    MIN_CHUNKS_FOR_ANSWER,
    MAX_CHUNKS,
    FAITHFULNESS_THRESHOLD,
)


def print_result(query: str, result: dict, header: str | None = None) -> None:
    """Pretty-print a generate_answer() result."""
    if header:
        print("\n" + "=" * 78)
        print(header)
        print("=" * 78)

    print(f"\nQuery: {query!r}")

    # Retrieval summary
    n_retrieved = len(result["chunks_retrieved"])
    n_used = len(result["chunks_used"])
    print(
        f"\nRetrieval: {n_retrieved} chunks retrieved, "
        f"{n_used} passed threshold (sim >= {SIMILARITY_THRESHOLD})"
    )
    if result["chunks_retrieved"]:
        sims = [f"{c['similarity']:.3f}" for c in result["chunks_retrieved"]]
        print(f"  Similarities (all retrieved): [{', '.join(sims)}]")
    if result["chunks_used"]:
        used_ids = [c["chunk_id"] for c in result["chunks_used"]]
        print(f"  Chunks sent to Gemini:")
        for cid in used_ids:
            print(f"    - {cid}")

    # Abstention case — no Gemini call was made
    if result["abstained"]:
        print(f"\nABSTAINED — no Gemini call made.")
        print(f"  Message: {result['answer']}")
        return

    # Answer
    print(f"\nAnswer:\n")
    indented = "\n".join("  " + line for line in result["answer"].splitlines())
    print(indented)

    # Citations
    print(f"\nParsed citations ({len(result['citations'])}):")
    if result["citations"]:
        for marker, chunk_id in sorted(
            result["citations"].items(),
            key=lambda kv: int(kv[0].strip("[]")),
        ):
            print(f"  {marker} -> {chunk_id}")
    else:
        print("  (none)")

    # Citation health
    health = result["citation_health"]
    if health["valid"]:
        print(f"\nCitation health: VALID (no warnings)")
    else:
        print(f"\nCitation health: {len(health['warnings'])} warning(s):")
        for w in health["warnings"]:
            print(f"  - {w}")

    # Faithfulness
    f = result.get("faithfulness")
    if f is not None:
        if f["valid"]:
            print(
                f"\nFaithfulness: VALID  "
                f"(mean overlap {f['mean_overlap']}, "
                f"{f['n_claims_checked']} cited claims checked, "
                f"{len(f['tagged_claims'])} tagged, "
                f"{len(f['uncited_claims'])} uncited)"
            )
        else:
            print(
                f"\nFaithfulness: {f['n_unsupported']}/{f['n_claims_checked']} "
                f"claims below threshold {FAITHFULNESS_THRESHOLD} "
                f"(mean overlap {f['mean_overlap']}):"
            )
            for w in f["warnings"]:
                print(f"  - {w}")

    # Token usage
    usage = result["usage"]
    if usage:
        print(
            f"\nTokens: in={usage['tokens_in']}  out={usage['tokens_out']}  "
            f"total={usage['tokens_total']}"
        )

    print(f"Thinking: {result['thinking']}")


def run_demo(num: int, query: str, header: str, **kwargs) -> dict:
    """Run one demo, time it, print the result, return it for aggregate stats."""
    t0 = time.perf_counter()
    result = generate_answer(query, **kwargs)
    elapsed = time.perf_counter() - t0
    print_result(query, result, header=header)
    print(f"\nLatency: {elapsed:.2f}s")
    return {"num": num, "result": result, "elapsed": elapsed}


def main() -> None:
    print("=" * 78)
    print("DAY 7 - BLOCK 4/5: End-to-end RAG generation + faithfulness demo")
    print("=" * 78)
    print(
        f"Settings: sim_threshold={SIMILARITY_THRESHOLD}, "
        f"min_chunks={MIN_CHUNKS_FOR_ANSWER}, max_chunks={MAX_CHUNKS}, "
        f"faithfulness_threshold={FAITHFULNESS_THRESHOLD}"
    )

    runs = []

    runs.append(run_demo(
        1,
        query="supply chain disruption and production delays",
        header="DEMO 1: Unfiltered query (broad)",
        k=5,
    ))

    runs.append(run_demo(
        2,
        query="supply chain disruption and production delays",
        header="DEMO 2: Same query, filtered to AAPL only",
        k=5,
        filter={"ticker": "AAPL"},
    ))

    runs.append(run_demo(
        3,
        query="iPhone revenue performance compared to last year",
        header="DEMO 3: Topic query, filtered to AAPL 2019-Q3",
        k=5,
        filter={"$and": [{"ticker": "AAPL"}, {"quarter": "2019-Q3"}]},
    ))

    runs.append(run_demo(
        4,
        query="CEO outlook on revenue growth in the coming quarter",
        header="DEMO 4: Outlook query, filtered to qa + interview sections",
        k=5,
        filter={"section": {"$in": ["qa", "interview"]}},
    ))

    runs.append(run_demo(
        5,
        query="best recipe for sourdough bread starter",
        header="DEMO 5: Off-topic baseline (sourdough vs. earnings calls)",
        k=5,
    ))

    # ---------------------------------------------------------------
    # Aggregate summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    total_latency = sum(r["elapsed"] for r in runs)
    total_tokens_in = sum(r["result"]["usage"].get("tokens_in", 0) for r in runs)
    total_tokens_out = sum(r["result"]["usage"].get("tokens_out", 0) for r in runs)
    n_abstained = sum(1 for r in runs if r["result"]["abstained"])
    n_unhealthy = sum(
        1 for r in runs
        if not r["result"]["abstained"]
        and not r["result"]["citation_health"]["valid"]
    )
    n_unfaithful = sum(
        1 for r in runs
        if not r["result"]["abstained"]
        and r["result"]["faithfulness"] is not None
        and not r["result"]["faithfulness"]["valid"]
    )

    print(f"\nTotal queries:       {len(runs)}")
    print(f"Abstained:           {n_abstained}")
    print(f"Answered:            {len(runs) - n_abstained}")
    print(f"Citation issues:     {n_unhealthy}")
    print(f"Faithfulness issues: {n_unfaithful}")
    print(f"Total latency:       {total_latency:.2f}s")
    print(f"Avg latency / call:  {total_latency / len(runs):.2f}s")
    print(f"Total tokens in:     {total_tokens_in}")
    print(f"Total tokens out:    {total_tokens_out}")
    print(f"Total tokens:        {total_tokens_in + total_tokens_out}")

    print("\nDemos complete.")


if __name__ == "__main__":
    main()