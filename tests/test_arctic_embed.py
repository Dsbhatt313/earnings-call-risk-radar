"""
Day 6 - Block 2: Sanity test for Arctic embedding model.

Loads the model, embeds two finance-flavored strings, checks:
  1. Model loads without error
  2. Embedding dimension is 1024
  3. Semantically similar strings get high cosine similarity
  4. Semantically different strings get lower cosine similarity
  5. Query-prefix mechanism works (queries use "query: " prefix; documents don't)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("Loading Snowflake/snowflake-arctic-embed-l-v2.0 ...")
    print("(First run downloads ~1.1 GB. Subsequent runs load from cache.)")
    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
    print("Model loaded.\n")

    # Two similar finance sentences (should have HIGH similarity)
    doc_a = "Our revenue grew 15 percent year-over-year driven by strong demand."
    doc_b = "Sales increased fifteen percent compared to last year due to robust customer demand."

    # One unrelated sentence (should have LOWER similarity to the first two)
    doc_c = "The new cafeteria menu features more vegetarian options."

    # A query about revenue growth (should match doc_a and doc_b, not doc_c)
    # Arctic v2.0 expects queries to be prefixed with "query: "
    query = "query: How did revenue grow this quarter?"

    print("Embedding documents and query ...")
    emb_a = model.encode(doc_a, normalize_embeddings=True)
    emb_b = model.encode(doc_b, normalize_embeddings=True)
    emb_c = model.encode(doc_c, normalize_embeddings=True)
    emb_q = model.encode(query, normalize_embeddings=True)

    print(f"Embedding shape: {emb_a.shape}")
    print(f"Embedding dtype: {emb_a.dtype}\n")

    # Sanity check 1: dimension
    assert emb_a.shape == (1024,), f"Expected (1024,), got {emb_a.shape}"
    print("PASS: dimension is 1024")

    # Sanity check 2: similar sentences are close
    sim_ab = cosine_similarity(emb_a, emb_b)
    print(f"\nCosine similarity (similar sentences A vs B): {sim_ab:.4f}")
    assert sim_ab > 0.7, f"Expected > 0.7 for similar sentences, got {sim_ab}"
    print("PASS: similar sentences have high similarity")

    # Sanity check 3: unrelated sentence is farther
    sim_ac = cosine_similarity(emb_a, emb_c)
    print(f"Cosine similarity (unrelated A vs C):          {sim_ac:.4f}")
    assert sim_ac < sim_ab, "Expected unrelated pair to be less similar than related pair"
    print("PASS: unrelated sentence has lower similarity than related one")

    # Sanity check 4: query retrieves correct doc
    sim_qa = cosine_similarity(emb_q, emb_a)
    sim_qb = cosine_similarity(emb_q, emb_b)
    sim_qc = cosine_similarity(emb_q, emb_c)
    print(f"\nQuery vs doc A (revenue growth):  {sim_qa:.4f}")
    print(f"Query vs doc B (sales increase):  {sim_qb:.4f}")
    print(f"Query vs doc C (cafeteria menu):  {sim_qc:.4f}")
    assert sim_qa > sim_qc and sim_qb > sim_qc, "Query should match revenue docs more than cafeteria doc"
    print("PASS: query retrieves relevant docs, not unrelated one")

    print("\nAll sanity checks passed. Block 2 done.")


if __name__ == "__main__":
    main()