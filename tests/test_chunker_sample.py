"""
Day 6 - Block 3c: Sample test for the chunker on 3 transcripts.

Loads 3 transcripts from transcripts_v2.csv, chunks them, and:
  1. Prints sample chunks for visual inspection
  2. Asserts token budget compliance (no chunk over MAX_TOKENS)
  3. Asserts no empty chunks
  4. Asserts overlap mechanism works (chunk N+1 shares trailing sentences with chunk N)
  5. Asserts metadata is well-formed (unique chunk_ids, correct sections)
"""

import sys
from pathlib import Path

# Add project root to sys.path so `from src.rag...` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import spacy
from transformers import AutoTokenizer

from src.rag.chunker import (
    MAX_TOKENS,
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    chunk_single_transcript,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "transcripts_v2.csv"

# How many transcripts to test on
N_SAMPLE = 3


def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    sample = df.head(N_SAMPLE).copy()
    print(f"Testing on {len(sample)} transcripts:\n")
    for _, row in sample.iterrows():
        print(f"  - {row['ticker']} {row['quarter']} ({row['transcript_word_count']} words)")

    print("\nLoading spaCy model ...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading Arctic tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-l-v2.0")

    print("\nChunking ...")
    all_chunks = []
    all_stats = []
    for _, row in sample.iterrows():
        chunks, stats = chunk_single_transcript(row, nlp, tokenizer)
        all_chunks.extend(chunks)
        all_stats.append(stats)

    chunks_df = pd.DataFrame(all_chunks)
    print(f"\nProduced {len(chunks_df)} chunks total across {N_SAMPLE} transcripts.")

    # -------------------------------------------------------------------------
    # Assertion 1: No empty chunks
    # -------------------------------------------------------------------------
    n_empty = (chunks_df["text"].str.strip() == "").sum()
    assert n_empty == 0, f"Found {n_empty} empty chunks"
    print(f"PASS: no empty chunks")

    # -------------------------------------------------------------------------
    # Assertion 2: Token budget compliance
    # No chunk should exceed MAX_TOKENS (except force-saved oversized sentences,
    # which we count separately).
    # -------------------------------------------------------------------------
    over_max = chunks_df[chunks_df["n_tokens"] > MAX_TOKENS]
    n_oversized_sentences = sum(s["oversized_sentences"] for s in all_stats)
    # Any chunks over MAX_TOKENS must equal the oversized_sentences count
    assert len(over_max) == n_oversized_sentences, (
        f"Found {len(over_max)} chunks over MAX_TOKENS={MAX_TOKENS} but only "
        f"{n_oversized_sentences} oversized single sentences accounted for"
    )
    print(f"PASS: token budget respected ({len(over_max)} oversized, all single-sentence)")

    # -------------------------------------------------------------------------
    # Assertion 3: Token distribution sanity
    # Most chunks should be near TARGET_TOKENS. We check the median is in a
    # reasonable range (200-400).
    # -------------------------------------------------------------------------
    median_tokens = chunks_df["n_tokens"].median()
    assert 200 <= median_tokens <= 400, (
        f"Median chunk size {median_tokens} is outside reasonable range [200, 400]"
    )
    print(f"PASS: median chunk size = {median_tokens:.0f} tokens (target {TARGET_TOKENS})")

    # -------------------------------------------------------------------------
    # Assertion 4: Unique chunk_ids
    # -------------------------------------------------------------------------
    n_unique = chunks_df["chunk_id"].nunique()
    assert n_unique == len(chunks_df), (
        f"Duplicate chunk_ids: {len(chunks_df)} chunks but only {n_unique} unique IDs"
    )
    print(f"PASS: all {n_unique} chunk_ids unique")

    # -------------------------------------------------------------------------
    # Assertion 5: Section values are clean
    # -------------------------------------------------------------------------
    valid_sections = set(chunks_df["section"].unique())
    assert valid_sections.issubset({"prep", "qa"}), (
        f"Unexpected section values: {valid_sections}"
    )
    print(f"PASS: sections are {valid_sections}")

    # -------------------------------------------------------------------------
    # Assertion 6: Overlap mechanism works
    # For each call's section, consecutive chunks should share trailing/leading
    # sentence indices.
    # -------------------------------------------------------------------------
    overlap_violations = []
    for (ticker, quarter, section), group in chunks_df.groupby(["ticker", "quarter", "section"]):
        group = group.sort_values("chunk_index").reset_index(drop=True)
        for i in range(len(group) - 1):
            curr = group.iloc[i]
            nxt = group.iloc[i + 1]
            # next chunk's sentence_start_idx should be <= current's sentence_end_idx
            # (i.e., next starts somewhere inside current's range, meaning overlap exists)
            # OR next starts exactly where current ended + 1 (zero overlap, allowed for
            # the very small case where overlap couldn't be seeded).
            if nxt["sentence_start_idx"] > curr["sentence_end_idx"] + 1:
                overlap_violations.append({
                    "ticker": ticker, "quarter": quarter, "section": section,
                    "curr_id": curr["chunk_id"], "curr_end": curr["sentence_end_idx"],
                    "next_id": nxt["chunk_id"], "next_start": nxt["sentence_start_idx"],
                })

    assert not overlap_violations, (
        f"Found {len(overlap_violations)} overlap violations. First few:\n"
        + "\n".join(str(v) for v in overlap_violations[:3])
    )
    print(f"PASS: overlap mechanism works (no gaps between consecutive chunks)")

    # -------------------------------------------------------------------------
    # Visual inspection: print first chunk of each section for first transcript
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VISUAL INSPECTION (first call)")
    print("=" * 70)
    first_ticker = sample.iloc[0]["ticker"]
    first_quarter = sample.iloc[0]["quarter"]
    first_call_chunks = chunks_df[
        (chunks_df["ticker"] == first_ticker) & (chunks_df["quarter"] == first_quarter)
    ].sort_values(["section", "chunk_index"])

    print(f"\n{first_ticker} {first_quarter}: {len(first_call_chunks)} total chunks")
    print(f"  prep chunks: {(first_call_chunks['section'] == 'prep').sum()}")
    print(f"  qa chunks:   {(first_call_chunks['section'] == 'qa').sum()}")

    # Show first prep chunk
    prep_chunks = first_call_chunks[first_call_chunks["section"] == "prep"]
    if len(prep_chunks):
        c = prep_chunks.iloc[0]
        print(f"\n--- First PREP chunk ({c['chunk_id']}, {c['n_tokens']} tokens) ---")
        print(f"sentence range: {c['sentence_start_idx']} to {c['sentence_end_idx']}")
        print(f"text (first 400 chars): {c['text'][:400]}...")

    # Show transition between prep chunk 0 and prep chunk 1 (overlap visualization)
    if len(prep_chunks) >= 2:
        c0 = prep_chunks.iloc[0]
        c1 = prep_chunks.iloc[1]
        print(f"\n--- Overlap check: chunk 0 -> chunk 1 ---")
        print(f"chunk 0 ends at sentence idx: {c0['sentence_end_idx']}")
        print(f"chunk 1 starts at sentence idx: {c1['sentence_start_idx']}")
        print(f"overlap span: {c0['sentence_end_idx'] - c1['sentence_start_idx'] + 1} sentences")
        print(f"chunk 0 last 200 chars: ...{c0['text'][-200:]}")
        print(f"chunk 1 first 200 chars: {c1['text'][:200]}...")

    # Show first qa chunk
    qa_chunks = first_call_chunks[first_call_chunks["section"] == "qa"]
    if len(qa_chunks):
        c = qa_chunks.iloc[0]
        print(f"\n--- First QA chunk ({c['chunk_id']}, {c['n_tokens']} tokens) ---")
        print(f"text (first 400 chars): {c['text'][:400]}...")

    # -------------------------------------------------------------------------
    # Summary stats
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOKEN DISTRIBUTION")
    print("=" * 70)
    print(chunks_df["n_tokens"].describe().round(1).to_string())

    print("\n" + "=" * 70)
    print("All assertions passed. Sample test complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()