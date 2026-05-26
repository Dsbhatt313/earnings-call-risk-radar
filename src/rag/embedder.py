"""
src/rag/embedder.py

Day 6 embedder for Risk Radar RAG pipeline.

Loads chunks from chunks_day6.csv, embeds each one using Snowflake's
Arctic-l-v2.0 model, and persists the resulting matrix as a NumPy .npy file.

The output matrix has shape (n_chunks, 1024). Row i corresponds to the chunk
at row i in chunks_day6.csv -- order is strictly preserved.

Public entry point: embed_chunks(input_csv, output_npy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ARCTIC_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"
BATCH_SIZE = 32              # Sweet spot for CPU; bump to 64+ if you have a GPU
EXPECTED_DIM = 1024          # Arctic-l-v2.0's output dimensionality


def embed_chunks(
    input_csv_path: str | Path,
    output_npy_path: str | Path,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
) -> np.ndarray:
    """
    Embed all chunks in input_csv and save the resulting matrix to output_npy.

    Important behavior:
      - Order is strictly preserved: matrix[i] corresponds to chunks_df.iloc[i].
      - Embeddings are L2-normalized so cosine similarity == dot product.
      - We do NOT add the Arctic query prefix here -- this function embeds
        DOCUMENTS, not queries. Query embedding happens separately at search time.

    Parameters
    ----------
    input_csv_path : str | Path
        Path to chunks_day6.csv (must have a 'text' column).
    output_npy_path : str | Path
        Where to save the embeddings matrix (.npy format).
    batch_size : int
        Number of chunks to encode per batch.
    verbose : bool
        Print progress and stats.

    Returns
    -------
    np.ndarray
        The embeddings matrix, shape (n_chunks, 1024), dtype float32.
    """
    input_csv_path = Path(input_csv_path)
    output_npy_path = Path(output_npy_path)

    if verbose:
        print(f"Reading {input_csv_path} ...")
    chunks_df = pd.read_csv(input_csv_path)
    n_chunks = len(chunks_df)
    if verbose:
        print(f"Loaded {n_chunks:,} chunks.")
        print(f"Loading Arctic model '{ARCTIC_MODEL_NAME}' ...")
        print("(Already cached locally; load takes ~30s.)")

    model = SentenceTransformer(ARCTIC_MODEL_NAME)

    # Sanity check: make sure 'text' column exists and has no NaN
    if "text" not in chunks_df.columns:
        raise ValueError(f"chunks_df missing 'text' column. Got: {list(chunks_df.columns)}")
    n_nan = chunks_df["text"].isna().sum()
    if n_nan > 0:
        raise ValueError(f"Found {n_nan} NaN values in 'text' column. Investigate before embedding.")

    texts = chunks_df["text"].tolist()

    if verbose:
        print(f"\nEmbedding {n_chunks:,} chunks with batch_size={batch_size} ...")
        print("(This will take ~15-45 min on CPU. Progress bar below.)")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=verbose,
        normalize_embeddings=True,  # L2-normalize so cosine similarity = dot product
        convert_to_numpy=True,
    )

    # Sanity checks on the output
    if embeddings.shape != (n_chunks, EXPECTED_DIM):
        raise RuntimeError(
            f"Unexpected embedding shape: got {embeddings.shape}, "
            f"expected ({n_chunks}, {EXPECTED_DIM})"
        )

    # Verify normalization (each row should have norm ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        bad = np.abs(norms - 1.0).max()
        raise RuntimeError(
            f"Embeddings not unit-normalized: max deviation {bad:.6f}"
        )

    # Save to disk
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy_path, embeddings)

    if verbose:
        print(f"\nEmbeddings saved to {output_npy_path}")
        print(f"Shape: {embeddings.shape}")
        print(f"Dtype: {embeddings.dtype}")
        print(f"File size on disk: ~{output_npy_path.stat().st_size / (1024 * 1024):.1f} MB")
        print(f"\nSample stats:")
        print(f"  Mean norm:    {norms.mean():.6f} (should be ~1.0)")
        print(f"  Min/max:      {embeddings.min():.4f} / {embeddings.max():.4f}")
        # Show first chunk_id and first 5 values of its embedding for traceability
        print(f"\nFirst row ({chunks_df.iloc[0]['chunk_id']}): first 5 dims = {embeddings[0][:5]}")
        print(f"Last row  ({chunks_df.iloc[-1]['chunk_id']}): first 5 dims = {embeddings[-1][:5]}")

    return embeddings