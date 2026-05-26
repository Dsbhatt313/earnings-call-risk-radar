"""
scripts/run_embedder.py

Day 6 - Block 4: Embed all chunks from chunks_day6.csv.

Reads:  data/processed/chunks_day6.csv
Writes: data/processed/embeddings_day6.npy  (~53 MB)

Usage:
    python scripts\run_embedder.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embedder import embed_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "chunks_day6.csv"
OUTPUT_NPY = PROJECT_ROOT / "data" / "processed" / "embeddings_day6.npy"


if __name__ == "__main__":
    embeddings = embed_chunks(
        input_csv_path=INPUT_CSV,
        output_npy_path=OUTPUT_NPY,
        verbose=True,
    )
    print(f"\nDone. Matrix shape {embeddings.shape} written to {OUTPUT_NPY}")