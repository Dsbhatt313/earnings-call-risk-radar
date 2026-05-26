"""
scripts/run_chroma_loader.py

Day 6 - Block 5: Load chunks + embeddings into ChromaDB.

Reads:  data/processed/chunks_day6.csv
        data/processed/embeddings_day6.npy
Writes: chroma_db/  (persistent ChromaDB directory)

Usage:
    python scripts\run_chroma_loader.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.vector_store import load_chunks_into_chroma

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_CSV = PROJECT_ROOT / "data" / "processed" / "chunks_day6.csv"
EMBEDDINGS_NPY = PROJECT_ROOT / "data" / "processed" / "embeddings_day6.npy"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"


if __name__ == "__main__":
    collection = load_chunks_into_chroma(
        chunks_csv_path=CHUNKS_CSV,
        embeddings_npy_path=EMBEDDINGS_NPY,
        chroma_db_path=CHROMA_DB_PATH,
        reset_first=True,  # clear existing data for clean re-runs
        verbose=True,
    )
    print(f"\nDone. Collection has {collection.count():,} entries.")
    print(f"ChromaDB persisted at: {CHROMA_DB_PATH}")