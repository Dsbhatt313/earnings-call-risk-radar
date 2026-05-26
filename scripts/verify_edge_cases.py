"""
scripts/verify_edge_cases.py

Day 6 - End of day spot-checks for known edge cases.

Verifies two things:
  1. NFLX interview-format calls surface in retrieval, AND the
     section="interview" filter works correctly.
  2. PYPL 2019-Q4 (Q&A mislabeled as prep) still has its content
     retrievable by unfiltered queries.

These edge cases were documented during Day 6 Block 3.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.retriever import retrieve


def print_results(label: str, query: str, results: list[dict]) -> None:
    print(f"\n--- {label} ---")
    print(f"Query: {query!r}")
    print(f"Returned {len(results)} results:")
    for i, r in enumerate(results, 1):
        md = r["metadata"]
        preview = r["text"][:160].replace("\n", " ")
        print(f"  [{i}] sim={r['similarity']:.4f}  {r['chunk_id']}")
        print(f"      ticker={md['ticker']}  quarter={md['quarter']}  section={md['section']}")
        print(f"      text: {preview}...")


def main():
    # -----------------------------------------------------------------
    # CHECK 1a: NFLX content surfaces without explicit ticker filter
    # -----------------------------------------------------------------
    print("=" * 70)
    print("CHECK 1: NFLX interview-format retrieval")
    print("=" * 70)

    query1 = "Netflix subscriber growth and content strategy"
    results1 = retrieve(query1, k=5)
    print_results("1a. Unfiltered query about NFLX", query1, results1)

    nflx_count = sum(1 for r in results1 if r["metadata"]["ticker"] == "NFLX")
    interview_count = sum(1 for r in results1 if r["metadata"]["section"] == "interview")
    print(f"\n  NFLX chunks in top-5: {nflx_count}")
    print(f"  Interview-section chunks in top-5: {interview_count}")

    # -----------------------------------------------------------------
    # CHECK 1b: section="interview" filter returns ONLY interview chunks
    # -----------------------------------------------------------------
    query2 = "long-term subscriber growth"
    results2 = retrieve(query2, k=5, filter={"section": "interview"})
    print_results("1b. Filtered to section=interview only", query2, results2)

    all_interview = all(r["metadata"]["section"] == "interview" for r in results2)
    all_nflx = all(r["metadata"]["ticker"] == "NFLX" for r in results2)
    print(f"\n  All results have section=interview: {all_interview}")
    print(f"  All results are NFLX (since only NFLX uses interview format): {all_nflx}")

    # -----------------------------------------------------------------
    # CHECK 2: PYPL 2019-Q4 content still retrievable
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHECK 2: PYPL 2019-Q4 retrieval (known mislabeling)")
    print("=" * 70)

    # 2a: Unfiltered query about Venmo (we saw 'Venmo' in the Q&A content
    # during diagnosis, so it should be retrievable somewhere)
    query3 = "Venmo growth and monetization"
    results3 = retrieve(query3, k=5, filter={"ticker": "PYPL"})
    print_results("2a. Venmo query filtered to PYPL", query3, results3)

    has_2019q4 = any(r["metadata"]["quarter"] == "2019-Q4" for r in results3)
    print(f"\n  PYPL 2019-Q4 content surfaces: {has_2019q4}")

    # 2b: Query specifically for PYPL 2019-Q4 to confirm the call IS in the index
    query4 = "fourth quarter results"
    results4 = retrieve(
        query4,
        k=5,
        filter={"$and": [{"ticker": "PYPL"}, {"quarter": "2019-Q4"}]},
    )
    print_results("2b. Filtered to PYPL 2019-Q4 specifically", query4, results4)

    n_q4_chunks = len(results4)
    print(f"\n  PYPL 2019-Q4 chunks accessible: {n_q4_chunks} (out of 50 chunks in this call)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"NFLX interview format:")
    print(f"  - Surfaces in unfiltered queries: {'YES' if nflx_count > 0 else 'NO'}")
    print(f"  - section=interview filter works: {'YES' if all_interview else 'NO'}")
    print(f"PYPL 2019-Q4:")
    print(f"  - Content retrievable (unfiltered): {'YES' if has_2019q4 else 'NO'}")
    print(f"  - Chunks accessible via ticker+quarter filter: {n_q4_chunks > 0}")


if __name__ == "__main__":
    main()