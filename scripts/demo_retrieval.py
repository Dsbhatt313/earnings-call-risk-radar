"""
scripts/demo_retrieval.py

Day 6 - Block 6: Demonstrate the retriever on real queries against the corpus.

Runs several example queries with and without metadata filters,
prints the top-K results with similarity scores and metadata.

Usage:
    python scripts\demo_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.retriever import retrieve


def print_results(query: str, results: list[dict], header: str | None = None) -> None:
    """Pretty-print a list of retrieval results."""
    if header:
        print("\n" + "=" * 70)
        print(header)
        print("=" * 70)
    print(f"\nQuery: {query!r}")
    print(f"Returned {len(results)} results:")
    for i, r in enumerate(results, 1):
        md = r["metadata"]
        print(f"\n  [{i}] sim={r['similarity']:.4f}  {r['chunk_id']}")
        print(f"      ticker={md['ticker']}  quarter={md['quarter']}  section={md['section']}")
        # Print first ~200 chars of text
        text_preview = r["text"][:220].replace("\n", " ")
        print(f"      text: {text_preview}...")


def main():
    # ---------------------------------------------------------------
    # Demo 1: Unfiltered query about supply chain risk
    # ---------------------------------------------------------------
    query1 = "supply chain disruption and production delays"
    results1 = retrieve(query1, k=5)
    print_results(query1, results1, header="DEMO 1: Unfiltered query")

    # ---------------------------------------------------------------
    # Demo 2: Same idea but filtered to AAPL only
    # ---------------------------------------------------------------
    query2 = "supply chain disruption and production delays"
    results2 = retrieve(query2, k=5, filter={"ticker": "AAPL"})
    print_results(query2, results2, header="DEMO 2: Same query, filtered to AAPL")

    # ---------------------------------------------------------------
    # Demo 3: Question about a specific topic on a specific call
    # ---------------------------------------------------------------
    query3 = "iPhone revenue performance compared to last year"
    results3 = retrieve(
        query3,
        k=5,
        filter={"$and": [{"ticker": "AAPL"}, {"quarter": "2019-Q3"}]},
    )
    print_results(query3, results3, header="DEMO 3: Topic query, filtered to AAPL 2019-Q3")

    # ---------------------------------------------------------------
    # Demo 4: Question about CEO statements on growth (Q&A section only)
    # ---------------------------------------------------------------
    query4 = "CEO outlook on revenue growth in the coming quarter"
    results4 = retrieve(
        query4,
        k=5,
        filter={"section": {"$in": ["qa", "interview"]}},
    )
    print_results(query4, results4, header="DEMO 4: Q&A/interview sections only")

    print("\n" + "=" * 70)
    print("Demos complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()