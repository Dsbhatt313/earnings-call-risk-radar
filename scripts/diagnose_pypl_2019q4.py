"""
Diagnose why PYPL 2019-Q4 was parsed as having no Q&A section.

Either: (a) the transcript genuinely has no Q&A (legitimate edge case),
or:     (b) it uses a marker variant we don't recognize (parser bug).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "transcripts_v2.csv"

df = pd.read_csv(INPUT_CSV)
row = df[(df["ticker"] == "PYPL") & (df["quarter"] == "2019-Q4")].iloc[0]
t = row["transcript"]

print(f"PYPL 2019-Q4")
print(f"Length: {len(t):,} chars / {row['transcript_word_count']:,} words")
print()
print("Marker checks:")
markers_to_check = [
    "Prepared Remarks:",
    "Questions and Answers:",
    "Questions & Answers:",
    "Q&A",
    "Question-and-Answer",
    "QUESTION-AND-ANSWER",
]
for m in markers_to_check:
    pos = t.find(m)
    print(f"  {m!r}: {'FOUND at ' + str(pos) if pos != -1 else 'not found'}")

# Look at lines containing 'question' anywhere
print()
print("Lines containing 'question' (case-insensitive):")
lines = t.split("\n")
q_lines = [(i, ln.strip()) for i, ln in enumerate(lines) if "question" in ln.lower()]
print(f"  Total: {len(q_lines)} lines")
for i, ln in q_lines[:8]:
    print(f"  line {i}: {ln[:180]}")

# Print the very end of the transcript - sometimes the marker is just gone
print()
print("LAST 600 CHARACTERS OF TRANSCRIPT:")
print("-" * 60)
print(t[-600:])
print("-" * 60)