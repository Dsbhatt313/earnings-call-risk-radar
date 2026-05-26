"""
src/rag/retriever.py

Day 6 retriever for Risk Radar RAG pipeline.

Embeds natural language queries and retrieves the most similar chunks from
the ChromaDB collection. The embedding model is loaded once at module level
so repeated queries are fast.

Public entry point: retrieve(query, k=5, filter=None).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ARCTIC_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"
COLLECTION_NAME = "risk_radar_chunks"

# Arctic was trained with this prefix for queries. NOT applied to documents.
QUERY_PREFIX = "query: "

# Default location of the persistent ChromaDB
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"


# -----------------------------------------------------------------------------
# Cached resources (load once, reuse)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the Arctic model once, cache it for the lifetime of the process."""
    return SentenceTransformer(ARCTIC_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_collection(chroma_db_path: str) -> chromadb.Collection:
    """Open the ChromaDB collection once, cache it for the lifetime of the process."""
    client = chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=COLLECTION_NAME)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def embed_query(query: str) -> list[float]:
    """
    Embed a natural language query for retrieval.

    Prepends Arctic's required query prefix and normalizes the resulting
    vector to unit length (so cosine math is clean).

    Parameters
    ----------
    query : str
        Natural language question.

    Returns
    -------
    list[float]
        1024-dim embedding as a Python list (ChromaDB expects this format).
    """
    model = _get_model()
    prefixed = QUERY_PREFIX + query
    embedding = model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()


def retrieve(
    query: str,
    k: int = 5,
    filter: dict[str, Any] | None = None,
    chroma_db_path: str | Path | None = None,
) -> list[dict]:
    """
    Retrieve the top-K most similar chunks for a natural language query.

    Parameters
    ----------
    query : str
        Natural language question.
    k : int
        How many top chunks to return.
    filter : dict | None
        Optional ChromaDB metadata filter. Examples:
          {"ticker": "AAPL"}                    # only AAPL chunks
          {"section": "qa"}                     # only Q&A chunks
          {"ticker": {"$in": ["AAPL", "MSFT"]}} # multiple tickers
        See ChromaDB docs for full filter syntax.
    chroma_db_path : str | Path | None
        Optional override of where ChromaDB lives. Defaults to project's chroma_db/.

    Returns
    -------
    list[dict]
        List of K results, each containing:
          - chunk_id (str)
          - text (str)
          - metadata (dict): ticker, quarter, call_date, section, etc.
          - similarity (float): cosine similarity, 0..1 (higher = more similar)
          - distance (float): the raw ChromaDB distance, for reference
    """
    if chroma_db_path is None:
        chroma_db_path = DEFAULT_CHROMA_DB_PATH

    collection = _get_collection(str(chroma_db_path))

    # Embed the query
    query_embedding = embed_query(query)

    # Query ChromaDB
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=filter,  # ChromaDB accepts None here (no filter)
    )

    # ChromaDB returns parallel lists nested inside lists (one inner list per
    # query). We sent one query, so we index [0] everywhere.
    ids = raw_results["ids"][0]
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    # Convert to a list of friendly dicts
    results = []
    for chunk_id, doc, md, dist in zip(ids, documents, metadatas, distances):
        results.append({
            "chunk_id": chunk_id,
            "text": doc,
            "metadata": md,
            "similarity": 1.0 - dist,  # cosine distance -> cosine similarity
            "distance": dist,
        })

    return results