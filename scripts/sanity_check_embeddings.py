"""
Quick semantic sanity check on the embeddings.

Compares within-company vs between-company chunk similarities to confirm
the embeddings carry real signal. If within-company chunks aren't more
similar to each other than between-company chunks, something is wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_CSV = PROJECT_ROOT / "data" / "processed" / "chunks_day6.csv"
EMBEDDINGS_NPY = PROJECT_ROOT / "data" / "processed" / "embeddings_day6.npy"


def main():
    chunks = pd.read_csv(CHUNKS_CSV)
    emb = np.load(EMBEDDINGS_NPY)

    print(f"chunks: {len(chunks)}")
    print(f"emb shape: {emb.shape}")
    print(f"row counts match: {len(chunks) == emb.shape[0]}")
    print()

    # Pick first 3 prep chunks for each company
    aapl_idx = chunks[(chunks.ticker == "AAPL") & (chunks.section == "prep")].index[:3].tolist()
    xom_idx = chunks[(chunks.ticker == "XOM") & (chunks.section == "prep")].index[:3].tolist()

    print(f"AAPL prep chunk indices: {aapl_idx}")
    print(f"XOM  prep chunk indices: {xom_idx}")
    print()

    # Within-company similarities (AAPL vs AAPL)
    print("WITHIN-COMPANY similarities (AAPL prep chunks vs each other):")
    print(f"  AAPL[0] vs AAPL[1]: {float(emb[aapl_idx[0]] @ emb[aapl_idx[1]]):.4f}")
    print(f"  AAPL[0] vs AAPL[2]: {float(emb[aapl_idx[0]] @ emb[aapl_idx[2]]):.4f}")
    print(f"  AAPL[1] vs AAPL[2]: {float(emb[aapl_idx[1]] @ emb[aapl_idx[2]]):.4f}")
    print()

    # Between-company similarities (AAPL vs XOM)
    print("BETWEEN-COMPANY similarities (AAPL vs XOM prep chunks):")
    print(f"  AAPL[0] vs XOM[0]:  {float(emb[aapl_idx[0]] @ emb[xom_idx[0]]):.4f}")
    print(f"  AAPL[0] vs XOM[1]:  {float(emb[aapl_idx[0]] @ emb[xom_idx[1]]):.4f}")
    print(f"  AAPL[1] vs XOM[0]:  {float(emb[aapl_idx[1]] @ emb[xom_idx[0]]):.4f}")
    print(f"  AAPL[1] vs XOM[1]:  {float(emb[aapl_idx[1]] @ emb[xom_idx[1]]):.4f}")
    print()

    # Check the chunks aren't identical (sanity: different chunks should be < 1.0)
    print("Chunk text previews (so you can sanity-check the content):")
    for label, idx in [("AAPL[0]", aapl_idx[0]), ("AAPL[1]", aapl_idx[1]),
                        ("XOM[0]", xom_idx[0]), ("XOM[1]", xom_idx[1])]:
        text = chunks.iloc[idx]["text"][:120]
        chunk_id = chunks.iloc[idx]["chunk_id"]
        print(f"  {label} ({chunk_id}): {text}...")


if __name__ == "__main__":
    main()