"""
scripts/run_judge.py — Day 8 Block 4: LLM-as-judge

Independent semantic evaluation of generated answers. For each answerable
non-integration query: generates the answer (gemini-2.5-flash via
generate_answer), then a SEPARATE judge model (gemini-2.0-flash) scores it
1-5 on faithfulness, completeness, relevance, with a one-sentence rationale.

Methodological note: judge and generator share a vendor (Google). Weaker
independence than a cross-vendor judge, documented as a known limitation.
After Day 10 free-tier shifts on gemini-2.0-flash, the judge was pinned to the
same MODEL as the generator (gemini-2.5-flash). This further weakens independence
but preserves the structured semantic scoring; a cross-vendor judge is the
proper fix and is listed in "Future work."
The judge is still a different model doing structured semantic scoring,
catching paraphrase the lexical-overlap tripwire (Day 7) and substring
theme_coverage (Block 2) miss.

Day 10 free-tier reality check (May 2026):
- gemini-2.0-flash free-tier daily quota dropped to 0 on this project, so the
  original cross-model judge plan was not runnable.
- gemini-2.5-flash free-tier daily quota is 20 requests per project per day —
  insufficient for a full 14-query run (28 calls) in one calendar day.
- This run completed n=1 (q1) as methodology demonstration. Full n=14 requires
  a paid tier or a 2-day schedule. Documented in README "Limitations".

Run from project root (after quota reset; ~14 gen + 14 judge calls):
    python -m scripts.run_judge
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.genai import types
from google.genai.errors import ServerError, ClientError

from src.rag.gemini_client import get_client
from src.rag.generator import generate_answer
from scripts.run_eval import load_eval_set, STRATUM_OFF_TOPIC, STRATUM_RISK_INTEGRATION

# --- config ---
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
JUDGE_RESULTS_PATH = EVAL_DIR / "eval_judge_day8.csv"

JUDGE_MODEL = "gemini-2.5-flash"   # SAME model as generator — fallback after gemini-2.0-flash free-tier daily quota dropped to 0 on Day 10 (Google shrinks older-model free tiers as newer ones launch). Weakens independence (same vendor AND same model), documented in README "Limitations". Judge still does structured semantic scoring the lexical tripwire (Day 7) and substring theme_coverage (Block 2) miss.
RETRIEVAL_K = 10
INTER_CALL_DELAY_SECONDS = 8       # respect per-minute rate limit

MAX_JUDGE_ATTEMPTS = 4
JUDGE_BACKOFF = [2, 5, 10]

JUDGE_PROMPT = """You are an impartial evaluator grading an AI-generated answer about company earnings calls.

You will see: the QUESTION asked, the SOURCE CHUNKS the answer was allowed to use, a REFERENCE ANSWER (what a good answer looks like), the EXPECTED THEMES, and the GENERATED ANSWER to grade.

Grade the GENERATED ANSWER on three dimensions, each 1-5:

1. faithfulness (1-5): Are the answer's factual claims supported by the SOURCE CHUNKS?
   5 = every claim traceable to the chunks. 1 = major claims unsupported / fabricated.

2. completeness (1-5): Does the answer cover the substance of the REFERENCE ANSWER and EXPECTED THEMES?
   5 = covers all key points. 1 = misses most key points.

3. relevance (1-5): Does the answer actually address the QUESTION asked?
   5 = directly on target. 1 = off-topic or evasive.

Judge meaning, not wording. Paraphrase that conveys the same fact counts as covered.

Return ONLY a JSON object, no other text:
{{"faithfulness": <int>, "completeness": <int>, "relevance": <int>, "rationale": "<one sentence>"}}

# QUESTION
{question}

# SOURCE CHUNKS
{chunks_block}

# REFERENCE ANSWER
{reference}

# EXPECTED THEMES
{themes}

# GENERATED ANSWER
{answer}

# Your JSON grade"""


def _format_chunks_for_judge(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['chunk_id']}]\n{c['text']}")
    return "\n\n".join(blocks)


def _parse_judge_response(text: str) -> dict:
    """Extract the JSON grade from the judge's response."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in judge response")
    obj = json.loads(cleaned[start:end + 1])
    for k in ("faithfulness", "completeness", "relevance"):
        if k not in obj or not isinstance(obj[k], int) or not (1 <= obj[k] <= 5):
            raise ValueError(f"Invalid or missing score: {k}")
    obj.setdefault("rationale", "")
    return obj


def _call_judge(prompt: str) -> tuple[dict | None, str]:
    """Call the judge model with retry. Returns (parsed_grade, error_str)."""
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    last_err = None
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        try:
            resp = client.models.generate_content(
                model=JUDGE_MODEL, contents=prompt, config=config,
            )
            grade = _parse_judge_response(resp.text)
            return grade, ""
        except (ServerError, ClientError) as e:
            last_err = e
            code = getattr(e, "code", None)
            if code in (429, 500, 503) and attempt < MAX_JUDGE_ATTEMPTS:
                wait = JUDGE_BACKOFF[attempt - 1]
                print(f"  [judge {code} attempt {attempt}/{MAX_JUDGE_ATTEMPTS}] retry in {wait}s...")
                time.sleep(wait)
                continue
            return None, f"API error {code}: {e}"
        except ValueError as e:
            last_err = e
            if attempt < MAX_JUDGE_ATTEMPTS:
                print(f"  [judge parse fail attempt {attempt}] retry...")
                time.sleep(2)
                continue
            return None, f"parse error: {e}"
    return None, f"exhausted retries: {last_err}"


def main() -> None:
    print("=" * 70)
    print("Day 8 Block 4 — LLM-as-judge")
    print(f"Generator: gemini-2.5-flash   Judge: {JUDGE_MODEL}")
    print("=" * 70)

    queries = load_eval_set()
    to_judge = [
        q for q in queries
        if q["is_answerable"]
        and q["stratum"] not in (STRATUM_OFF_TOPIC, STRATUM_RISK_INTEGRATION)
    ]
    print(f"Judging {len(to_judge)} answerable queries.")
    print("First query triggers model load (~30s). ~18s/query (gen+judge+delays).\n")

    rows = []
    for idx, q in enumerate(to_judge):
        qid = q["query_id"]
        print(f"[q{qid:>2}] {q['query'][:55]}...", flush=True)

        if idx > 0:
            time.sleep(INTER_CALL_DELAY_SECONDS)

        # 1. Generate the answer (gemini-2.5-flash)
        try:
            result = generate_answer(q["query"], k=RETRIEVAL_K, filter=q["filter"])
        except Exception as e:
            print(f"       GEN ERROR: {e!r}")
            rows.append({"query_id": qid, "stratum": q["stratum"], "judged": False,
                         "faithfulness": "", "completeness": "", "relevance": "",
                         "rationale": "", "note": f"gen error: {e!r}"})
            continue

        if result["abstained"] or result.get("api_failed"):
            note = "abstained" if result["abstained"] else "gen api_failed"
            print(f"       skipped ({note})")
            rows.append({"query_id": qid, "stratum": q["stratum"], "judged": False,
                         "faithfulness": "", "completeness": "", "relevance": "",
                         "rationale": "", "note": note})
            continue

        time.sleep(INTER_CALL_DELAY_SECONDS)

        # 2. Judge the answer (gemini-2.0-flash)
        prompt = JUDGE_PROMPT.format(
            question=q["query"],
            chunks_block=_format_chunks_for_judge(result["chunks_used"]),
            reference=q["reference_answer"] or "(none provided)",
            themes=" | ".join(q["expected_themes"]) or "(none provided)",
            answer=result["answer"],
        )
        grade, err = _call_judge(prompt)

        if grade is None:
            print(f"       JUDGE FAILED: {err}")
            rows.append({"query_id": qid, "stratum": q["stratum"], "judged": False,
                         "faithfulness": "", "completeness": "", "relevance": "",
                         "rationale": "", "note": err})
        else:
            print(f"       faith={grade['faithfulness']} "
                  f"complete={grade['completeness']} rel={grade['relevance']}")
            rows.append({"query_id": qid, "stratum": q["stratum"], "judged": True,
                         "faithfulness": grade["faithfulness"],
                         "completeness": grade["completeness"],
                         "relevance": grade["relevance"],
                         "rationale": grade["rationale"], "note": ""})

    # Write results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "stratum", "judged",
                  "faithfulness", "completeness", "relevance", "rationale", "note"]
    with JUDGE_RESULTS_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nOK {JUDGE_RESULTS_PATH}  ({len(rows)} rows)")

    # Aggregate
    judged = [r for r in rows if r["judged"]]
    if judged:
        def _mean(k):
            return round(sum(r[k] for r in judged) / len(judged), 2)
        print(f"\nJudged {len(judged)}/{len(rows)} queries.")
        print(f"  Mean faithfulness: {_mean('faithfulness')}/5")
        print(f"  Mean completeness: {_mean('completeness')}/5")
        print(f"  Mean relevance:    {_mean('relevance')}/5")
    print("\n>> Done.")


if __name__ == "__main__":
    main()