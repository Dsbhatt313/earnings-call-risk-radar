"""
scripts/build_chroma.py — Rebuild chroma_db from committed artifacts.

Reads the committed chunks_day6.csv + embeddings_day6.npy and loads them
into ChromaDB. Used on Streamlit Cloud's first run (chroma_db isn't committed)
and locally if the store gets corrupted.

Idempotent — checks if chroma_db is already populated and skips if so.
Safe to call on every app startup.

Why rebuild instead of committing chroma_db: data_level0.bin exceeds GitHub's
100 MB single-file limit. Git LFS works but adds bandwidth fragility on the
free tier. Committing the source artifacts (CSV + NPY, both well under 100 MB)
and rebuilding chroma_db on first run is the most reproducible deploy story.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.vector_store import load_chunks_into_chroma, get_or_create_collection

CHUNKS_CSV = PROJECT_ROOT / "data" / "processed" / "chunks_day6.csv"
EMBEDDINGS_NPY = PROJECT_ROOT / "data" / "processed" / "embeddings_day6.npy"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

EXPECTED_CHUNK_COUNT = 13518


def is_built() -> bool:
    """Return True if chroma_db is populated to (at least) the expected count."""
    if not CHROMA_DB_PATH.exists():
        return False
    try:
        collection = get_or_create_collection(CHROMA_DB_PATH)
        return collection.count() >= EXPECTED_CHUNK_COUNT
    except Exception:
        return False


def build_if_missing(verbose: bool = True) -> None:
    """Rebuild chroma_db from committed CSV + NPY if not already populated."""
    if is_built():
        if verbose:
            print(f"chroma_db already built ({EXPECTED_CHUNK_COUNT}+ chunks). Skipping.")
        return

    if not CHUNKS_CSV.exists():
        raise FileNotFoundError(
            f"Cannot rebuild: {CHUNKS_CSV} is missing. Must be committed to the repo."
        )
    if not EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            f"Cannot rebuild: {EMBEDDINGS_NPY} is missing. Must be committed to the repo."
        )

    if verbose:
        print("=" * 60)
        print("Building chroma_db from committed artifacts...")
        print(f"  chunks:     {CHUNKS_CSV}")
        print(f"  embeddings: {EMBEDDINGS_NPY}")
        print(f"  output:     {CHROMA_DB_PATH}")
        print("=" * 60)

    load_chunks_into_chroma(
        chunks_csv_path=CHUNKS_CSV,
        embeddings_npy_path=EMBEDDINGS_NPY,
        chroma_db_path=CHROMA_DB_PATH,
        reset_first=True,
        verbose=verbose,
    )

    if verbose:
        print("\nchroma_db built successfully.")


if __name__ == "__main__":
    build_if_missing()