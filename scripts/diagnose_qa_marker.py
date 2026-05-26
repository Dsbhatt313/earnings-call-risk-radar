"""
Diagnose why 219 transcripts are 'missing qa section'.

Scans all 273 transcripts for common variants of the Q&A section marker
and prints which ones don't match any known variant.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "transcripts_v2.csv"

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} transcripts.\n")

# Candidate marker variants
patterns = [
    "Questions and Answers:",
    "Questions and Answers",
    "Questions & Answers:",
    "Questions & Answers",
    "Q&A:",
    "Q & A:",
    "Q and A:",
    "Questions-and-Answers",
    "QUESTIONS AND ANSWERS",
]

print("Marker presence across all transcripts (substring match):")
for pat in patterns:
    count = df["transcript"].str.contains(pat, regex=False, na=False).sum()
    print(f"  {count:4d}  contains  {pat!r}")

# Now find transcripts that have NONE of these
combined_pattern = "|".join([p.replace("&", r"\&") for p in patterns])
has_any = df["transcript"].str.contains("|".join(patterns), regex=True, na=False)
missing = df[~has_any]
print(f"\nTranscripts with NONE of these markers: {len(missing)}")

# For the missing ones, look at lines mentioning "question" or "answer"
if len(missing) > 0:
    print(f"\nFirst 3 transcripts with no recognized marker:")
    for _, row in missing.head(3).iterrows():
        ticker = row["ticker"]
        quarter = row["quarter"]
        t = row["transcript"]
        print(f"\n--- {ticker} {quarter} ---")
        # Look for lines mentioning question
        lines = t.split("\n")
        question_lines = [ln.strip() for ln in lines if "question" in ln.lower()]
        print(f"  Lines mentioning 'question': {len(question_lines)}")
        for line in question_lines[:5]:
            print(f"    {line[:150]}")

# Now check: of the 273, which DO have "Questions and Answers:"?
# (Our chunker's specific marker.) Compare to: which have at least SOME marker?
strict_match = df["transcript"].str.contains("Questions and Answers:", regex=False, na=False).sum()
loose_match = has_any.sum()
print(f"\nSummary:")
print(f"  Transcripts matching exact 'Questions and Answers:' = {strict_match}")
print(f"  Transcripts matching at least one variant            = {loose_match}")
print(f"  Transcripts matching nothing                         = {len(df) - loose_match}")