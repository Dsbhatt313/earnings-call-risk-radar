"""Day 7 Block 3 sanity test for _parse_citations() — synthetic answers, no API calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.generator import _parse_citations


def run() -> None:
    retrieved_ids = {
        "AAPL_2023-Q2_prep_004",
        "AAPL_2023-Q2_prep_007",
        "AAPL_2023-Q2_qa_012",
    }

    # Case 1: clean, all valid
    clean = (
        "Revenue grew 12% YoY [1]. Services strength [2] offset iPhone weakness [3].\n"
        "\n"
        "Sources:\n"
        "[1] AAPL_2023-Q2_prep_004\n"
        "[2] AAPL_2023-Q2_prep_007\n"
        "[3] AAPL_2023-Q2_qa_012\n"
    )
    cites, health = _parse_citations(clean, retrieved_ids)
    assert health["valid"], f"Case 1 should be valid, got: {health['warnings']}"
    assert len(cites) == 3
    print("Case 1 (clean):              PASS")

    # Case 2: body references [4] but Sources only defines [1]-[3]
    missing_def = (
        "Revenue grew [1] and services strong [4].\n"
        "Sources:\n"
        "[1] AAPL_2023-Q2_prep_004\n"
    )
    _, health = _parse_citations(missing_def, retrieved_ids)
    assert not health["valid"]
    assert any("[4]" in w and "not defined" in w for w in health["warnings"])
    print("Case 2 (undefined footnote): PASS")

    # Case 3: Sources defines [2] but body never uses it
    orphan = (
        "Revenue grew [1].\n"
        "Sources:\n"
        "[1] AAPL_2023-Q2_prep_004\n"
        "[2] AAPL_2023-Q2_prep_007\n"
    )
    _, health = _parse_citations(orphan, retrieved_ids)
    assert not health["valid"]
    assert any("never referenced" in w for w in health["warnings"])
    print("Case 3 (orphan source):      PASS")

    # Case 4: hallucinated chunk_id (not in retrieved set)
    hallucinated = (
        "Revenue grew [1].\n"
        "Sources:\n"
        "[1] MSFT_2099-Q9_prep_999\n"
    )
    _, health = _parse_citations(hallucinated, retrieved_ids)
    assert not health["valid"]
    assert any("not in the retrieved chunks" in w for w in health["warnings"])
    print("Case 4 (hallucinated chunk): PASS")

    # Case 5: duplicate footnote number defined twice
    dupe = (
        "First [1] then [1].\n"
        "Sources:\n"
        "[1] AAPL_2023-Q2_prep_004\n"
        "[1] AAPL_2023-Q2_prep_007\n"
    )
    _, health = _parse_citations(dupe, retrieved_ids)
    assert not health["valid"]
    assert any("defined more than once" in w for w in health["warnings"])
    print("Case 5 (duplicate definition): PASS")

    # Case 6: multi-citation [1, 2] format
    multi = (
        "Both companies faced shortages [1, 2]. Apple also faced chip issues [1].\n"
        "Sources:\n"
        "[1] AAPL_2023-Q2_prep_004\n"
        "[2] AAPL_2023-Q2_prep_007\n"
    )
    cites, health = _parse_citations(multi, retrieved_ids)
    assert health["valid"], f"Case 6 should be valid, got: {health['warnings']}"
    assert len(cites) == 2
    print("Case 6 (multi-citation):     PASS")

    # Case 7: multi-citation with [1, 2, 3] including a number that
    # ONLY appears in multi-citation form — proves the split logic works
    triple = (
        "All three sources agree on this point [1, 2, 3].\n"
        "Sources:\n"
        "[1] AAPL_2023-Q2_prep_004\n"
        "[2] AAPL_2023-Q2_prep_007\n"
        "[3] AAPL_2023-Q2_qa_012\n"
    )
    _, health = _parse_citations(triple, retrieved_ids)
    assert health["valid"], f"Case 7 should be valid, got: {health['warnings']}"
    print("Case 7 (triple multi-cite):  PASS")

    print("\nAll 7 citation parser tests passed.")


if __name__ == "__main__":
    run()