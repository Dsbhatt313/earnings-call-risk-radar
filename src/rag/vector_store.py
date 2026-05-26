"""
src/rag/vector_store.py

Day 6 vector store loader for Risk Radar RAG pipeline.

Loads chunks_day6.csv + embeddings_day6.npy into a persistent ChromaDB
collection. The collection lives at chroma_db/ at the project root and
can be queried by future blocks (retrieval, evaluation, app).

Public entry points:
  - load_chunks_into_chroma(...) : one-shot bulk load
  - get_or_create_collection(...): for downstream blocks to access the same collection
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from chromadb.config import Settings

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COLLECTION_NAME = "risk_radar_chunks"
INSERT_BATCH_SIZE = 1000

# Metadata fields we want ChromaDB to filter on.
# (ChromaDB stores all metadata, but these are the canonical ones we'll use.)
METADATA_FIELDS = [
    "ticker",
    "quarter",
    "call_date",
    "section",
    "chunk_index",
    "n_tokens",
]


def get_or_create_collection(
    chroma_db_path: str | Path,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Open (or create) a ChromaDB collection at the given persistent path.

    Use this in downstream blocks (retrieval, eval, app) to access the
    same collection that load_chunks_into_chroma populated.

    Parameters
    ----------
    chroma_db_path : str | Path
        Directory where ChromaDB stores its files.
    collection_name : str
        Name of the collection within the ChromaDB instance.

    Returns
    -------
    chromadb.Collection
        The collection object. Ready to query or insert into.
    """
    chroma_db_path = Path(chroma_db_path)
    chroma_db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_db_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # get_or_create returns existing if it exists, else makes new.
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity for retrieval
    )

    return collection


def load_chunks_into_chroma(
    chunks_csv_path: str | Path,
    embeddings_npy_path: str | Path,
    chroma_db_path: str | Path,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = INSERT_BATCH_SIZE,
    reset_first: bool = True,
    verbose: bool = True,
) -> chromadb.Collection:
    """
    Load chunks + embeddings into a persistent ChromaDB collection.

    Steps:
      1. Read chunks_csv and embeddings_npy. Sanity-check that lengths match.
      2. Open (or create) the persistent ChromaDB collection.
      3. Optionally clear it (reset_first=True) so a re-run produces a clean state.
      4. Insert all chunks in batches with their IDs, embeddings, documents, metadata.
      5. Report stats.

    Parameters
    ----------
    chunks_csv_path : str | Path
        Path to chunks_day6.csv.
    embeddings_npy_path : str | Path
        Path to embeddings_day6.npy. Must have shape (len(chunks), 1024).
    chroma_db_path : str | Path
        Directory where ChromaDB stores its files.
    collection_name : str
        Name of the collection within the ChromaDB instance.
    batch_size : int
        Number of chunks to insert per batch call.
    reset_first : bool
        If True, delete existing collection contents before inserting.
        Recommended for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    chromadb.Collection
        The populated collection.
    """
    chunks_csv_path = Path(chunks_csv_path)
    embeddings_npy_path = Path(embeddings_npy_path)
    chroma_db_path = Path(chroma_db_path)

    if verbose:
        print(f"Reading {chunks_csv_path} ...")
    chunks_df = pd.read_csv(chunks_csv_path)
    if verbose:
        print(f"Loaded {len(chunks_df):,} chunks.")
        print(f"Reading {embeddings_npy_path} ...")
    embeddings = np.load(embeddings_npy_path)
    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")

    # Sanity: rows must align
    if len(chunks_df) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: chunks_df has {len(chunks_df)} rows but "
            f"embeddings has {embeddings.shape[0]} rows"
        )

    # Sanity: required columns
    required_cols = {"chunk_id", "text"} | set(METADATA_FIELDS)
    missing = required_cols - set(chunks_df.columns)
    if missing:
        raise ValueError(f"chunks_df missing columns: {missing}")

    # Open the collection
    if verbose:
        print(f"\nOpening ChromaDB at {chroma_db_path} ...")
    client = chromadb.PersistentClient(
        path=str(chroma_db_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # Reset if requested (delete then recreate)
    if reset_first:
        try:
            client.delete_collection(name=collection_name)
            if verbose:
                print(f"Deleted existing collection '{collection_name}'.")
        except Exception:
            pass  # collection didn't exist; that's fine

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Build the inputs ChromaDB needs
    ids = chunks_df["chunk_id"].tolist()
    documents = chunks_df["text"].tolist()
    # Metadata must be a list of dicts, one per chunk, with serializable values only.
    metadatas = []
    for _, row in chunks_df.iterrows():
        md = {}
        for field in METADATA_FIELDS:
            value = row[field]
            # ChromaDB accepts str/int/float/bool. Cast numpy types defensively.
            if isinstance(value, (np.integer,)):
                md[field] = int(value)
            elif isinstance(value, (np.floating,)):
                md[field] = float(value)
            else:
                md[field] = str(value) if not isinstance(value, (str, int, float, bool)) else value
        metadatas.append(md)

    # Insert in batches
    n = len(chunks_df)
    if verbose:
        print(f"\nInserting {n:,} chunks in batches of {batch_size} ...")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        if verbose:
            print(f"  inserted {end:,}/{n:,}")

    if verbose:
        final_count = collection.count()
        print(f"\nCollection '{collection_name}' now holds {final_count:,} chunks.")
        # Show a sample peek
        peek = collection.peek(limit=2)
        print(f"\nSample (first 2 entries):")
        for i, (cid, doc, md) in enumerate(zip(peek["ids"], peek["documents"], peek["metadatas"])):
            print(f"  {i+1}. id={cid}")
            print(f"     metadata={md}")
            print(f"     text={doc[:120]}...")

    return collection