"""
scripts/run_chunker.py

Day 6 - Block 3d: Run the chunker on the full transcripts corpus.

Reads:  data/processed/transcripts_v2.csv  (273 transcripts)
Writes: data/processed/chunks_day6.csv     (~11,500 chunks)

Usage:
    python scripts\run_chunker.py
"""

import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.chunker import chunk_all_transcripts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "transcripts_v2.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "chunks_day6.csv"


if __name__ == "__main__":
    chunks_df = chunk_all_transcripts(
        input_csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
        verbose=True,
    )
    print(f"\nDone. {len(chunks_df)} chunks written to {OUTPUT_CSV}")