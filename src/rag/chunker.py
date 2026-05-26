"""
src/rag/chunker.py

Day 6 chunker for Risk Radar RAG pipeline.

Reads transcripts_v2.csv, produces sentence-aware chunks with overlap,
respecting Arctic embedding model's tokenizer budget. Writes one row per
chunk to chunks_day6.csv with rich metadata for downstream retrieval.

Public entry point: chunk_all_transcripts(input_csv, output_csv).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import spacy
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

ARCTIC_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"
SPACY_MODEL_NAME = "en_core_web_sm"

TARGET_TOKENS = 300          # Soft target: close chunk once reached
MAX_TOKENS = 400             # Hard ceiling: never exceed
OVERLAP_TOKENS = 50          # Sentences worth ~50 tokens carry over to next chunk

# Markers in the raw transcript text
PREP_MARKER = "Prepared Remarks:"
QA_MARKERS = ["Questions and Answers:", "Questions & Answers:"]

# Closing-matter markers (first match ends the Q&A section)
CLOSING_MARKERS = [
    "Duration:",
    "Call participants:",
    "[Operator Closing Remarks]",
    "More AAPL analysis",        # OK to be generic - we match the prefix
    "All earnings call transcripts",
]


# -----------------------------------------------------------------------------
# Text cleaning
# -----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove bracketed annotations like [Operator Instructions] and collapse
    runs of whitespace. Leaves all real content (speaker labels, prose) intact.

    Parameters
    ----------
    text : str
        Raw text to clean.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Strip [anything in square brackets]
    text = re.sub(r"\[.*?\]", "", text)
    # Collapse all whitespace runs (including newlines) to single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------------------------------------------------------
# Section splitting
# -----------------------------------------------------------------------------
#
# Known limitation: 1 of 273 transcripts (PYPL 2019-Q4) has a malformed source
# where the "Questions and Answers:" marker appears AFTER the Q&A content
# rather than before it. Our parser sees the marker near the end of the file
# and incorrectly sweeps the entire Q&A into the prep section.
#
# Net effect: PYPL 2019-Q4 has 0 qa chunks; its Q&A content is mislabeled as
# prep. Retrieval queries WITHOUT a section filter still find this content
# normally. Only section-filtered queries (section=qa) miss it.
#
# Not fixed because: adding a heuristic to detect "marker too close to end"
# could mis-fire on calls with legitimately short Q&A sections. The cost of
# the heuristic exceeds the benefit of fixing 1 row.
# -----------------------------------------------------------------------------

def split_into_sections(transcript: str) -> tuple[str, str, str]:
    """
    Split the raw transcript into (prepared_remarks, qa_section, interview_section).

    Earnings calls come in three formats:
      1. Traditional with "Questions and Answers:" marker
         -> prep_text populated, qa_text populated, interview_text empty
      2. Traditional with "Questions & Answers:" marker (ampersand variant)
         -> same as above
      3. NFLX-style interview format with NO Q&A marker
         -> prep_text empty, qa_text empty, interview_text contains everything

    Detection logic:
      - Look for any known Q&A marker. If found -> traditional format.
      - If no Q&A marker AND "Prepared Remarks:" missing or near-empty content
        -> interview format. Entire transcript becomes interview_text.
      - Closing matter (Duration:, Call participants:, etc.) is stripped from
        whichever section ends the transcript.

    Parameters
    ----------
    transcript : str
        Raw transcript text from the CSV.

    Returns
    -------
    tuple[str, str, str]
        (prep_text, qa_text, interview_text), any can be empty strings.
    """
    if not isinstance(transcript, str) or not transcript.strip():
        return "", "", ""

    # Find earliest QA marker if any exist
    qa_marker_pos = -1
    qa_marker_used = None
    for marker in QA_MARKERS:
        pos = transcript.find(marker)
        if pos != -1 and (qa_marker_pos == -1 or pos < qa_marker_pos):
            qa_marker_pos = pos
            qa_marker_used = marker

    # Helper: trim closing matter (Duration:, Call participants:, etc.) from text end
    def _trim_closing(text: str) -> str:
        earliest = len(text)
        for closing_marker in CLOSING_MARKERS:
            pos = text.find(closing_marker)
            if pos != -1 and pos < earliest:
                earliest = pos
        return text[:earliest]

    # Case 1: traditional format (QA marker found)
    if qa_marker_pos != -1:
        prep_marker_pos = transcript.find(PREP_MARKER)
        if prep_marker_pos == -1:
            prep_text = ""
        else:
            prep_start = prep_marker_pos + len(PREP_MARKER)
            prep_text = transcript[prep_start:qa_marker_pos]
        qa_start = qa_marker_pos + len(qa_marker_used)
        qa_text = _trim_closing(transcript[qa_start:])
        return prep_text, qa_text, ""

    # Case 2: no QA marker -> interview format
    # Use everything after "Prepared Remarks:" if present, else whole transcript
    prep_marker_pos = transcript.find(PREP_MARKER)
    if prep_marker_pos != -1:
        interview_start = prep_marker_pos + len(PREP_MARKER)
        interview_text = _trim_closing(transcript[interview_start:])
    else:
        interview_text = _trim_closing(transcript)
    return "", "", interview_text


# -----------------------------------------------------------------------------
# Sentence splitting + token counting
# -----------------------------------------------------------------------------

def split_sentences(text: str, nlp: spacy.language.Language) -> list[str]:
    """
    Split cleaned text into a list of sentence strings using spaCy.

    Parameters
    ----------
    text : str
        Cleaned text (post clean_text).
    nlp : spacy.language.Language
        Loaded spaCy English model.

    Returns
    -------
    list[str]
        List of non-empty sentence strings.
    """
    if not text.strip():
        return []
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents]
    # Drop any sentences that ended up empty after stripping
    return [s for s in sentences if s]


def count_tokens(text: str, tokenizer: Any) -> int:
    """
    Count tokens in text using Arctic's tokenizer.

    Uses add_special_tokens=False so we count only the content tokens,
    not the [CLS]/[SEP] that get added at embedding time. This gives us
    a reliable measure of how much of the model's 512-token budget the
    chunk's content consumes.

    Parameters
    ----------
    text : str
        Text to tokenize.
    tokenizer : transformers.PreTrainedTokenizer
        Arctic's tokenizer.

    Returns
    -------
    int
        Number of content tokens.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


# -----------------------------------------------------------------------------
# Core chunking algorithm
# -----------------------------------------------------------------------------

def build_chunks_from_sentences(
    sentences: list[str],
    tokenizer: Any,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[dict]:
    """
    Build overlapping chunks from a list of sentences, respecting token budget.

    Algorithm:
      - Walk sentences in order, accumulating into the current chunk.
      - When adding the next sentence would exceed max_tokens, close current.
      - When current already meets target_tokens, also close at next boundary.
      - On close: seed the next chunk with trailing sentences from the previous
        chunk that sum to ~overlap_tokens.
      - Edge case: a single sentence longer than max_tokens is force-saved as
        its own oversized chunk and a warning is collected (returned in result).

    Parameters
    ----------
    sentences : list[str]
        Sentences for this section (already cleaned and split).
    tokenizer : transformers.PreTrainedTokenizer
        Arctic's tokenizer for token counting.
    target_tokens : int
        Soft target chunk size.
    overlap_tokens : int
        How many tokens of trailing sentences to seed the next chunk with.
    max_tokens : int
        Hard ceiling per chunk.

    Returns
    -------
    list[dict]
        Each dict has: text, sentence_start_idx, sentence_end_idx, n_tokens, n_chars.
        Empty list if no sentences provided.
    """
    if not sentences:
        return []

    # Pre-count tokens for each sentence once (we'll reference repeatedly)
    sentence_token_counts = [count_tokens(s, tokenizer) for s in sentences]

    chunks: list[dict] = []
    current_indices: list[int] = []   # indices into `sentences` for current chunk
    current_token_count = 0

    def _save_current_chunk() -> None:
        """Finalize the current chunk and append to chunks."""
        if not current_indices:
            return
        chunk_sentences = [sentences[i] for i in current_indices]
        chunk_text = " ".join(chunk_sentences)
        chunks.append({
            "text": chunk_text,
            "sentence_start_idx": current_indices[0],
            "sentence_end_idx": current_indices[-1],
            "n_tokens": current_token_count,
            "n_chars": len(chunk_text),
        })

    def _compute_overlap_seed() -> tuple[list[int], int]:
        """
        Walk backward through current_indices to collect trailing sentences
        whose token sum reaches overlap_tokens. Returns (new_indices, new_token_count).
        """
        seed_indices: list[int] = []
        seed_tokens = 0
        # Walk from the end of current_indices backward
        for idx in reversed(current_indices):
            sent_tokens = sentence_token_counts[idx]
            # Stop BEFORE adding if this would push us well past overlap_tokens
            # AND we already have at least one sentence (so seed is never empty
            # when current had content).
            if seed_indices and seed_tokens + sent_tokens > overlap_tokens * 1.5:
                break
            seed_indices.insert(0, idx)
            seed_tokens += sent_tokens
            if seed_tokens >= overlap_tokens:
                break
        return seed_indices, seed_tokens

    for i, sentence in enumerate(sentences):
        sent_tokens = sentence_token_counts[i]

        # Edge case: single sentence exceeds max_tokens entirely.
        # Force-save any current chunk first, then save this sentence alone.
        if sent_tokens > max_tokens:
            if current_indices:
                _save_current_chunk()
                current_indices = []
                current_token_count = 0
            # Save the oversized sentence as its own chunk
            chunks.append({
                "text": sentence,
                "sentence_start_idx": i,
                "sentence_end_idx": i,
                "n_tokens": sent_tokens,
                "n_chars": len(sentence),
                "_warning": "oversized_single_sentence",
            })
            continue

        # Would adding this sentence exceed the hard ceiling?
        if current_token_count + sent_tokens > max_tokens:
            _save_current_chunk()
            seed_indices, seed_tokens = _compute_overlap_seed()
            current_indices = seed_indices
            current_token_count = seed_tokens

        # If we've already hit target, close at this sentence boundary
        elif current_token_count >= target_tokens:
            _save_current_chunk()
            seed_indices, seed_tokens = _compute_overlap_seed()
            current_indices = seed_indices
            current_token_count = seed_tokens

        # Add this sentence to current chunk
        current_indices.append(i)
        current_token_count += sent_tokens

    # Save trailing chunk (whatever's left)
    if current_indices:
        _save_current_chunk()

    return chunks


# -----------------------------------------------------------------------------
# Per-transcript orchestration
# -----------------------------------------------------------------------------

def chunk_single_transcript(
    row: pd.Series,
    nlp: spacy.language.Language,
    tokenizer: Any,
) -> tuple[list[dict], dict]:
    """
    Produce all chunks for a single transcript row.

    Handles three transcript formats:
      - Traditional: produces 'prep' and 'qa' chunks
      - Interview: produces 'interview' chunks only

    Parameters
    ----------
    row : pd.Series
        One row from transcripts_v2.csv. Must have: ticker, quarter, date_parsed,
        transcript.
    nlp : spacy.language.Language
        Loaded spaCy model.
    tokenizer : transformers.PreTrainedTokenizer
        Arctic tokenizer.

    Returns
    -------
    tuple[list[dict], dict]
        (chunks, stats)
        chunks: list of chunk dicts with full metadata.
        stats:  per-call stats including counts by section and format.
    """
    ticker = row["ticker"]
    quarter = row["quarter"]
    call_date = row["date_parsed"]
    transcript = row["transcript"]

    stats = {
        "ticker": ticker,
        "quarter": quarter,
        "n_prep_chunks": 0,
        "n_qa_chunks": 0,
        "n_interview_chunks": 0,
        "missing_prep": False,
        "missing_qa": False,
        "is_interview_format": False,
        "oversized_sentences": 0,
    }

    # Split into sections (now three-way)
    prep_raw, qa_raw, interview_raw = split_into_sections(transcript)

    # Format detection and missing-section accounting
    if interview_raw.strip():
        stats["is_interview_format"] = True
    else:
        if not prep_raw.strip():
            stats["missing_prep"] = True
        if not qa_raw.strip():
            stats["missing_qa"] = True

    all_chunks: list[dict] = []

    sections_to_process = [
        ("prep", prep_raw),
        ("qa", qa_raw),
        ("interview", interview_raw),
    ]

    for section_name, section_text in sections_to_process:
        if not section_text.strip():
            continue

        cleaned = clean_text(section_text)
        sentences = split_sentences(cleaned, nlp)
        if not sentences:
            continue

        section_chunks = build_chunks_from_sentences(sentences, tokenizer)

        for idx, chunk in enumerate(section_chunks):
            if "_warning" in chunk:
                stats["oversized_sentences"] += 1
                del chunk["_warning"]
            chunk["chunk_id"] = f"{ticker}_{quarter}_{section_name}_{idx:03d}"
            chunk["ticker"] = ticker
            chunk["quarter"] = quarter
            chunk["call_date"] = call_date
            chunk["section"] = section_name
            chunk["chunk_index"] = idx
            all_chunks.append(chunk)

            if section_name == "prep":
                stats["n_prep_chunks"] += 1
            elif section_name == "qa":
                stats["n_qa_chunks"] += 1
            else:
                stats["n_interview_chunks"] += 1

    return all_chunks, stats


# -----------------------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------------------

def chunk_all_transcripts(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read transcripts CSV, chunk every transcript, save chunks CSV.

    Parameters
    ----------
    input_csv_path : str | Path
        Path to transcripts_v2.csv (must have columns: ticker, quarter,
        date_parsed, transcript).
    output_csv_path : str | Path
        Where to write the chunks CSV.
    verbose : bool
        If True, print progress and summary stats.

    Returns
    -------
    pd.DataFrame
        The chunks dataframe (also written to disk).
    """
    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)

    if verbose:
        print(f"Reading {input_csv_path} ...")
    df = pd.read_csv(input_csv_path)
    if verbose:
        print(f"Loaded {len(df)} transcripts.")
        print(f"Loading spaCy model '{SPACY_MODEL_NAME}' ...")
    nlp = spacy.load(SPACY_MODEL_NAME)

    if verbose:
        print(f"Loading Arctic tokenizer '{ARCTIC_MODEL_NAME}' ...")
    tokenizer = AutoTokenizer.from_pretrained(ARCTIC_MODEL_NAME)

    if verbose:
        print("Chunking transcripts ...")
    all_chunks: list[dict] = []
    all_stats: list[dict] = []

    for i, row in df.iterrows():
        chunks, stats = chunk_single_transcript(row, nlp, tokenizer)
        all_chunks.extend(chunks)
        all_stats.append(stats)
        if verbose and (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(df)} transcripts ...")

    chunks_df = pd.DataFrame(all_chunks)

    # Reorder columns for readability
    column_order = [
        "chunk_id", "ticker", "quarter", "call_date", "section",
        "chunk_index", "sentence_start_idx", "sentence_end_idx",
        "n_tokens", "n_chars", "text",
    ]
    chunks_df = chunks_df[column_order]

    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(output_csv_path, index=False)

    if verbose:
        stats_df = pd.DataFrame(all_stats)
        print("\n" + "=" * 60)
        print("CHUNKING SUMMARY")
        print("=" * 60)
        n_interview_calls = stats_df["is_interview_format"].sum()
        n_traditional_calls = len(df) - n_interview_calls
        print(f"Transcripts processed:        {len(df)}")
        print(f"  traditional format:         {n_traditional_calls}")
        print(f"  interview format:           {n_interview_calls}")
        print(f"Total chunks produced:        {len(chunks_df)}")
        print(f"  prep chunks:                {(chunks_df['section'] == 'prep').sum()}")
        print(f"  qa chunks:                  {(chunks_df['section'] == 'qa').sum()}")
        print(f"  interview chunks:           {(chunks_df['section'] == 'interview').sum()}")
        print(f"Traditional calls missing prep:  {stats_df[~stats_df['is_interview_format']]['missing_prep'].sum()}")
        print(f"Traditional calls missing qa:    {stats_df[~stats_df['is_interview_format']]['missing_qa'].sum()}")
        print(f"Oversized sentences:          {stats_df['oversized_sentences'].sum()}")
        print(f"\nToken count distribution:")
        print(chunks_df["n_tokens"].describe().round(1).to_string())
        print(f"\nChunks per call distribution:")
        per_call = chunks_df.groupby(["ticker", "quarter"]).size()
        print(per_call.describe().round(1).to_string())
        print(f"\nOutput written to: {output_csv_path}")

    return chunks_df