"""
Block 1c diagnostic: confirm we can reach Gemini's API and get a response.

Sends one trivial prompt with gemini-2.5-flash. Prints model output,
token usage, and latency. If this script runs clean, the Gemini stack
is healthy end-to-end and Day 7 generation work can proceed.

Re-run this anytime Gemini behavior gets weird to isolate "is the API
itself the problem?" from "is our prompt/retrieval the problem?"
"""

import sys
import time
from pathlib import Path

# sys.path bootstrap — same pattern as Day 6 tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.gemini_client import get_client


TEST_MODEL = "gemini-2.5-flash"
TEST_PROMPT = "Reply with exactly five words confirming you are working."


def main() -> None:
    print(f"[Block 1c] Testing Gemini connection")
    print(f"  Model:  {TEST_MODEL}")
    print(f"  Prompt: {TEST_PROMPT!r}")
    print()

    client = get_client()

    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=TEST_MODEL,
        contents=TEST_PROMPT,
    )
    elapsed = time.perf_counter() - t0

    print(f"  Response text: {response.text!r}")
    print(f"  Latency:       {elapsed:.2f}s")

    # Token usage — confirms billing-relevant metadata is returned
    usage = response.usage_metadata
    if usage is not None:
        print(f"  Tokens in:     {usage.prompt_token_count}")
        print(f"  Tokens out:    {usage.candidates_token_count}")
        print(f"  Tokens total:  {usage.total_token_count}")

    print()
    print("[Block 1c] PASS — Gemini stack is healthy.")


if __name__ == "__main__":
    main()