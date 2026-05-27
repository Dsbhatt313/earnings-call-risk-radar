"""
scripts/demo_risk_explainer.py

Day 7 - Block 6: Demo the risk explainer on 3 (ticker, quarter) pairs covering:
  - A standard call (AAPL 2019-Q3) — clean inference, both models work
  - A potentially high-risk call (BYND 2021-Q3) — supply chain disruption era
  - An interview-format call (NFLX 2022-Q4) — exercises NaN handling for y_5d

Usage:
    python scripts\\demo_risk_explainer.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.risk_explainer import explain_risk


def print_explanation(result: dict) -> None:
    """Pretty-print an explain_risk() result."""
    print(f"\n{'=' * 78}")
    print(f"RISK EXPLANATION: {result['ticker']} {result['quarter']}")
    print(f"{'=' * 78}")

    # Scores
    print("\n--- Day 5 Risk Scores ---")
    for horizon, score in result["scores"].items():
        if score is None:
            print(f"  {horizon}: (not computed)")
            continue
        if "error" in score:
            print(f"  {horizon}: ERROR — {score['error']}")
            continue
        print(
            f"  {horizon}: risk_score={score['risk_score']} "
            f"({score['model']})"
        )

    # Adaptive question
    if result["question"]:
        print(f"\n--- Adaptive Question ---")
        print(f"  Built from: {result['question_basis']}")

        # Show all top drivers with classification
        print(f"  Top drivers (signed by magnitude, classified):")
        for d in result.get("all_top_drivers", []):
            arrow = "↑" if d["direction"] == "risky" else "↓"
            tag = d["classification"]
            print(
                f"    {arrow} {d['feature']:30s} "
                f"contribution={d['contribution']:+.4f}  [{tag}]"
            )

        # Show which content drivers actually went into the question
        if result.get("question_used_fallback"):
            print(f"\n  No content drivers available — used generic fallback question.")
        else:
            print(f"\n  Content drivers used in question:")
            for d in result.get("question_drivers", []):
                arrow = "↑" if d["direction"] == "risky" else "↓"
                print(f"    {arrow} {d['feature']}: {d['topic']}")

        print(f"\n  Question:")
        print(f"    {result['question']}")
    else:
        print("\n--- Adaptive Question ---")
        print("  (could not be built — see errors below)")

    # Focused explanation
    print(f"\n--- Focused Explanation (this call only) ---")
    f = result["explanation_focused"]
    if f is None:
        print("  (skipped)")
    elif f["abstained"]:
        print(f"  ABSTAINED: {f['answer']}")
    else:
        answer_indented = "\n".join("  " + line for line in f["answer"].splitlines())
        print(answer_indented)
        if f["faithfulness"]:
            print(
                f"\n  Citation health: "
                f"{'VALID' if f['citation_health']['valid'] else 'WARNINGS'}  |  "
                f"Faithfulness: mean overlap "
                f"{f['faithfulness']['mean_overlap']}, "
                f"{f['faithfulness']['n_unsupported']} unsupported claim(s)"
            )

    # Elsewhere explanation
    print(f"\n--- Elsewhere Explanation (cross-call patterns) ---")
    e = result["explanation_elsewhere"]
    if e is None:
        print("  (skipped)")
    elif e["abstained"]:
        print(f"  ABSTAINED: {e['answer']}")
    else:
        answer_indented = "\n".join("  " + line for line in e["answer"].splitlines())
        print(answer_indented)
        if e["faithfulness"]:
            print(
                f"\n  Citation health: "
                f"{'VALID' if e['citation_health']['valid'] else 'WARNINGS'}  |  "
                f"Faithfulness: mean overlap "
                f"{e['faithfulness']['mean_overlap']}, "
                f"{e['faithfulness']['n_unsupported']} unsupported claim(s)"
            )

    # Errors
    if result["errors"]:
        print(f"\n--- Errors / Notes ---")
        for err in result["errors"]:
            print(f"  - {err}")


def main() -> None:
    print("=" * 78)
    print("DAY 7 - BLOCK 6: Risk explainer demo")
    print("=" * 78)
    print("Loading dataset and models on first call may take ~5-15s...")

    cases = [
        ("AAPL", "2019-Q3"),
        ("BYND", "2021-Q3"),
        ("NFLX", "2022-Q4"),  # likely interview-format → y_5d NaN handling tested
    ]

    total_t0 = time.perf_counter()
    for ticker, quarter in cases:
        t0 = time.perf_counter()
        result = explain_risk(ticker, quarter)
        elapsed = time.perf_counter() - t0
        print_explanation(result)
        print(f"\nLatency for this case: {elapsed:.2f}s")

    print(f"\n{'=' * 78}")
    print(f"Total demo time: {time.perf_counter() - total_t0:.2f}s")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()