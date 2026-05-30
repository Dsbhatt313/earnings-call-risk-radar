# Post-Deploy Follow-Ups

These are known follow-ups deferred from Day 10 due to the
gemini-2.5-flash free-tier daily cap of 20 requests. See README
"Limitations & Future Work" for the public-facing summary.

## 1. Complete LLM-judge eval (n=14)

**Status:** n=1 at deploy (q1 only). 14 queries deferred.

**When to do:** Any day with fresh quota OR after upgrading to paid tier.

**How:**
- Free-tier path (2 calendar days):
  - Day A: run `python -m scripts.run_judge`, capture first ~8-9 results
  - Day B: re-run for remaining queries
  - Add a resume mechanism first (see below) OR manually slice `to_judge`
  - Update `eval_judge_day8.csv` with new rows
- Paid-tier path (1 hour):
  - Upgrade `GEMINI_API_KEY` project to a paid tier
  - Single run completes all 14 query-pairs

**Code change needed first (resume mechanism, ~15 lines):**
Before the loop in `scripts/run_judge.py`, read the existing
`eval_judge_day8.csv`, build a `set(judged_qids)`, and skip queries
whose query_id is already in that set. This lets the script
incrementally complete the eval across multiple days.

## 2. Clean the eval CSV after judge run

**Status:** `eval_judge_day8.csv` currently has 1 real row + 14 rows
containing full Google API error JSON in the `note` column.

**When to do:** After Step 1 completes.

**How:**
PowerShell one-liner replacing verbose error JSON with `quota_exceeded`:

```powershell
$rows = Import-Csv data\eval\eval_judge_day8.csv
foreach ($r in $rows) {
    if ($r.note -like "API error 429*") {
        $r.note = "quota_exceeded (free-tier daily cap, gemini-2.5-flash, 20/day)"
    }
}
$rows | Export-Csv data\eval\eval_judge_day8.csv -NoTypeInformation -Encoding UTF8
```

(After Step 1 there should be 0 rows still in this state, but the
snippet remains here for reference in case any rows still failed.)

## 3. Update README

After Steps 1-2, edit README:
- Results table: replace `LLM-judge: n=1, q1 only (proof-of-method)` with
  real n=14 means (faithfulness, completeness, relevance, each /5)
- Limitations: remove "n=1 due to quota" caveat; keep cross-vendor judge
  as remaining limitation
- "What I Learned" Day 10 section: append "Returned post-deploy and
  completed the deferred eval on fresh quota."

## 4. Optional: cross-vendor judge

**Status:** True independence not achieved. Same-vendor (Google),
same-model (gemini-2.5-flash) at deploy.

**When:** Lower priority. Address only if doing further work on this
project's eval methodology.

**How:** Swap `JUDGE_MODEL` config in `run_judge.py` for an Anthropic
Claude API call (`claude-3-haiku-20240307` or similar — cheap, fast,
genuinely independent from Google's Gemini). Requires a separate API
key in `.env` (`ANTHROPIC_API_KEY=...`). Update the README
"Limitations" section accordingly.