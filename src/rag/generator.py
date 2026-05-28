"""
Generator module for the Risk Radar RAG pipeline.

Public API: generate_answer(query, k=5, filter=None, thinking=False) -> dict

Day 7 design decisions (see day7_journey.md for full reasoning):
  - Zero-shot prompt (few-shot deferred to evaluation-driven improvement)
  - Soft refuse + partial answer + provenance tags
  - Footnote citations with post-hoc parser and citation_health flag
  - Lexical-overlap faithfulness tripwire (LLM-as-judge promoted to Day 8)
  - Adaptive top-K: similarity threshold 0.45, min 2 chunks, max 5 chunks
  - Temperature 0 for deterministic grounded output
  - Gemini 2.5 thinking OFF by default (toggleable for Day 9 Streamlit)
  - Retry-with-backoff on transient Gemini ServerErrors
"""

from __future__ import annotations

import re
import time
from typing import Any

from google.genai import types
from google.genai.errors import ServerError, ClientError
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SK_STOPWORDS

from src.rag.gemini_client import get_client
from src.rag.retriever import retrieve


# ---- Module constants ----------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0

# Adaptive top-K parameters
SIMILARITY_THRESHOLD = 0.45    # confirmed functional in Block 4; tune on Day 8
MIN_CHUNKS_FOR_ANSWER = 2      # if fewer chunks pass threshold, abstain entirely
MAX_CHUNKS = 5                 # ceiling regardless of how many pass threshold

# Retry config for Gemini calls (handles 503 UNAVAILABLE etc.)
MAX_GEMINI_ATTEMPTS = 4
GEMINI_BACKOFF_SECONDS = [2, 5, 10]  # waits BETWEEN attempts

# Chunk_id format: TICKER_YYYY-QN_section_NNN (e.g., AAPL_2023-Q2_prep_004)
CHUNK_ID_PATTERN = re.compile(r"[A-Z]+_\d{4}-Q\d_[a-z]+_\d{3}")

# Citation pattern in answer body. Matches [1], [1, 2], [1, 2, 3], etc.
# Capture group returns the inner string like "1" or "1, 2" — caller splits
# on commas to get individual numbers.
FOOTNOTE_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Sources section pattern: line starting with [N] followed by chunk_id
SOURCES_LINE_PATTERN = re.compile(
    r"^\[(\d+)\]\s+([A-Z]+_\d{4}-Q\d_[a-z]+_\d{3})\s*$",
    re.MULTILINE,
)

# ---- Faithfulness checking constants -------------------------------------

FAITHFULNESS_THRESHOLD = 0.30  # claims below this lexical overlap are flagged

# Sentence-end splitter: handles ., !, ? followed by whitespace.
# Not perfect for "Inc." or "$89.5B." but good enough for tripwire-level checking.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

# Tokens that don't count as content (treated as noise).
# sklearn's frozen set is ~300 common English words; we add finance-specific noise.
STOPWORDS = frozenset(_SK_STOPWORDS) | {
    "company", "companies", "quarter", "year", "yoy", "qoq",
}

# ---- Prompt template -----------------------------------------------------

PROMPT_TEMPLATE = """You are an analyst answering questions about company earnings call transcripts.

You will be given a question and a numbered list of transcript chunks retrieved from a vector database. Each chunk has a unique chunk_id identifying its company, quarter, section, and position.

# Rules

1. **Grounding.** Base your answer on the provided chunks. Do not use information from your training data unless explicitly tagged (see rule 4).

2. **Soft refuse with partial answers.** If the chunks partially address the question, answer what they cover and clearly note what they don't. If the chunks don't address the question at all, say so plainly. Do not fabricate to fill gaps.

3. **Citations (footnote format).**
   - Mark every factual claim with a footnote number like [1], [2] placed immediately after the claim.
   - Number sources in order of first appearance in your answer. The first cited source is [1], the second NEW source is [2], etc.
   - Never reuse a number for a different source. Never reference a number you do not define.
   - If multiple consecutive facts come from the same source, cite it once at the end of that group, not after every sentence.
   - **MANDATORY: Your answer MUST end with a "Sources:" section.** This is not optional under any circumstances, including when:
     - The answer is short (even a one-sentence answer needs Sources).
     - All citations come from the same chunk.
     - Only one chunk is cited.
     - The answer is an abstention or partial refuse.
     An answer without a Sources section is incomplete and will fail downstream validation. Format exactly:
     Sources:
     [1] <chunk_id>
     [2] <chunk_id>

4. **Provenance tagging for non-chunk content.**
   - If you add useful context from general knowledge (not from the chunks), prefix that claim with [Not in chunks] and do NOT assign it a footnote.
   - If you are inferring rather than quoting, prefix that claim with [Inferred].
   - These tags are only for claims that go beyond what the chunks explicitly state. Default behavior is to stay inside the chunks.

5. **Tone.** Direct, factual, no hedging language like "it seems" or "perhaps" unless the chunks themselves are uncertain. Match the seriousness of a financial analyst memo.

# Question
{question}

# Retrieved chunks
{chunks_block}

# Your answer"""


# ---- Internal helpers: retrieval & prompt construction -------------------

def _filter_chunks_by_threshold(chunks: list[dict]) -> list[dict]:
    """Keep chunks with similarity >= threshold, capped at MAX_CHUNKS."""
    qualified = [c for c in chunks if c["similarity"] >= SIMILARITY_THRESHOLD]
    return qualified[:MAX_CHUNKS]


def _format_chunks_block(chunks: list[dict]) -> str:
    """Format chunks for insertion into the prompt template."""
    blocks = []
    for c in chunks:
        header = f"[chunk_id: {c['chunk_id']} | similarity: {c['similarity']:.3f}]"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


# ---- Internal helpers: Gemini call with retry ----------------------------

def _call_gemini(prompt: str, thinking: bool) -> tuple[str, dict]:
    """
    Send the prompt to Gemini and return (answer_text, usage_dict).

    Wraps the call with temperature=0 and the chosen thinking setting.
    Implements explicit retry-with-backoff on transient server errors
    (503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED, 500 INTERNAL).

    If all retries fail, returns an error message as the answer_text
    so the caller can continue rather than crashing.
    """
    client = get_client()

    thinking_config = types.ThinkingConfig(
        thinking_budget=0 if not thinking else -1
    )
    generation_config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        thinking_config=thinking_config,
    )

    last_error: Exception | None = None
    for attempt in range(1, MAX_GEMINI_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=generation_config,
            )

            usage = {}
            if response.usage_metadata is not None:
                usage = {
                    "tokens_in": response.usage_metadata.prompt_token_count,
                    "tokens_out": response.usage_metadata.candidates_token_count,
                    "tokens_total": response.usage_metadata.total_token_count,
                }

            return response.text, usage

        except ServerError as e:
            # 5xx errors: transient on Google's side, retry
            last_error = e
            if attempt < MAX_GEMINI_ATTEMPTS:
                wait = GEMINI_BACKOFF_SECONDS[attempt - 1]
                print(
                    f"  [Gemini {e.code} on attempt {attempt}/{MAX_GEMINI_ATTEMPTS}] "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                break

        except ClientError as e:
            # 429 RESOURCE_EXHAUSTED can be a transient per-minute rate limit,
            # so retry it like a server error. Other 4xx are our fault — don't retry.
            if e.code == 429:
                last_error = e
                if attempt < MAX_GEMINI_ATTEMPTS:
                    wait = GEMINI_BACKOFF_SECONDS[attempt - 1]
                    print(
                        f"  [Gemini 429 RESOURCE_EXHAUSTED on attempt "
                        f"{attempt}/{MAX_GEMINI_ATTEMPTS}] retrying in {wait}s "
                        f"(may be per-minute limit)..."
                    )
                    time.sleep(wait)
                    continue
                else:
                    break
            return (
                f"[Gemini call failed with client error {e.code}: {e}. "
                f"This is likely a bug in our request, not a transient issue.]",
                {"api_failed": True},
            )

    # All retries exhausted
    return (
        f"[Gemini call failed after {MAX_GEMINI_ATTEMPTS} attempts. "
        f"Last error: {last_error}. The service is likely overloaded or the "
        f"daily quota is exhausted; try again later.]",
        {"api_failed": True},
    )


# ---- Internal helpers: citation parsing ----------------------------------

def _parse_citations(
    answer_text: str,
    retrieved_chunk_ids: set[str],
) -> tuple[dict[str, str], dict]:
    """
    Extract footnote citations from the answer's Sources section, validate them
    against retrieved chunks and inline references, and return (citations, health).

    citations: dict mapping "[N]" -> chunk_id
    health: dict with 'valid' bool and 'warnings' list
    """
    warnings: list[str] = []

    # Parse Sources section
    sources_matches = SOURCES_LINE_PATTERN.findall(answer_text)
    citations: dict[str, str] = {}
    for num, chunk_id in sources_matches:
        key = f"[{num}]"
        if key in citations:
            warnings.append(
                f"Footnote {key} defined more than once in Sources section"
            )
        citations[key] = chunk_id

    # Find all [N] references in the body (excluding the Sources section itself).
    # Supports single citations like [1] and multi-citations like [1, 2, 3].
    sources_split = re.split(r"\n\s*Sources\s*:\s*\n", answer_text, maxsplit=1)
    body_text = sources_split[0] if len(sources_split) > 1 else answer_text

    body_refs: set[str] = set()
    for match in FOOTNOTE_PATTERN.findall(body_text):
        for num in match.split(","):
            body_refs.add(num.strip())

    # Check 1: every body reference has a definition
    for ref_num in body_refs:
        if f"[{ref_num}]" not in citations:
            warnings.append(
                f"Footnote [{ref_num}] referenced in answer but not defined in Sources"
            )

    # Check 2: every defined source is referenced
    for cite_key in citations:
        ref_num = cite_key.strip("[]")
        if ref_num not in body_refs:
            warnings.append(
                f"Footnote {cite_key} defined in Sources but never referenced in answer"
            )

    # Check 3: every cited chunk_id was actually in the retrieved set
    for cite_key, chunk_id in citations.items():
        if chunk_id not in retrieved_chunk_ids:
            warnings.append(
                f"Footnote {cite_key} cites chunk_id '{chunk_id}' which was "
                f"not in the retrieved chunks (possible hallucination)"
            )

    # Check 4: there's a Sources section but no body references at all
    if not body_refs and citations:
        warnings.append("Sources section present but no footnotes used in body")

    return citations, {
        "valid": len(warnings) == 0,
        "warnings": warnings,
    }


# ---- Internal helpers: faithfulness checking -----------------------------

def _tokenize_for_overlap(text: str) -> set[str]:
    """
    Tokenize text for lexical overlap: lowercase, strip punctuation,
    drop stopwords and very short tokens. Returns a set (deduplicated).
    """
    cleaned = re.sub(r"[^a-z0-9%]+", " ", text.lower())
    tokens = cleaned.split()
    return {t for t in tokens if t not in STOPWORDS and len(t) >= 2}


def _split_into_claims(answer_body: str) -> list[str]:
    """
    Split the answer body into sentence-level claims.

    Skips empty lines, headers (start with #), and very short fragments.
    Each returned claim should contain enough content to evaluate.
    """
    # Strip the Sources section if present (defensive — caller usually pre-strips)
    sources_split = re.split(r"\n\s*Sources\s*:\s*\n", answer_body, maxsplit=1)
    body = sources_split[0] if len(sources_split) > 1 else answer_body

    claims: list[str] = []
    for raw_sentence in SENTENCE_SPLIT_PATTERN.split(body):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        stripped = sentence.lstrip("*-#• ").strip()
        if len(stripped) < 20:
            continue
        claims.append(stripped)
    return claims


def _claim_cites(claim: str) -> set[str]:
    """Extract footnote numbers cited in a single claim. Handles [1] and [1, 2]."""
    refs: set[str] = set()
    for match in FOOTNOTE_PATTERN.findall(claim):
        for num in match.split(","):
            refs.add(num.strip())
    return refs


def check_faithfulness(
    answer_text: str,
    citations: dict[str, str],
    chunks_used: list[dict],
) -> dict:
    """
    Compute lexical overlap faithfulness for each cited claim in the answer.

    For each claim:
      - Identify its cited footnote(s)
      - Look up the corresponding chunk text(s)
      - Compute (claim_tokens ∩ chunk_tokens) / claim_tokens
      - Flag if overlap < FAITHFULNESS_THRESHOLD

    Claims with no citations are reported separately (skipped from scoring).
    Claims with provenance tags ([Not in chunks], [Inferred]) are skipped —
    they're intentionally outside the chunks.

    Returns a dict with keys:
      scores:            list of {claim, cited, overlap, supported}
      n_claims_checked:  int
      n_unsupported:     int
      uncited_claims:    list of claims with no [N] (informational)
      tagged_claims:     list of provenance-tagged claims (informational)
      mean_overlap:      average overlap across checked claims
      valid:             True if no unsupported claims
      warnings:          human-readable for each unsupported claim
    """
    # Build chunk_id -> chunk text lookup
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunks_used}

    # Pre-tokenize each chunk once (cache for speed)
    chunk_tokens_by_id = {
        cid: _tokenize_for_overlap(text)
        for cid, text in chunk_text_by_id.items()
    }

    claims = _split_into_claims(answer_text)

    scores: list[dict] = []
    uncited_claims: list[str] = []
    tagged_claims: list[str] = []
    warnings: list[str] = []

    for claim in claims:
        # Skip provenance-tagged claims (intentionally outside chunks)
        if "[Not in chunks]" in claim or "[Inferred]" in claim:
            tagged_claims.append(claim)
            continue

        cited_nums = _claim_cites(claim)
        if not cited_nums:
            uncited_claims.append(claim)
            continue

        # Resolve cited numbers to chunk_ids via the citations dict
        cited_chunk_ids: list[str] = []
        for num in cited_nums:
            chunk_id = citations.get(f"[{num}]")
            if chunk_id and chunk_id in chunk_tokens_by_id:
                cited_chunk_ids.append(chunk_id)

        if not cited_chunk_ids:
            # Unresolvable citations — Block 3 parser already warned; skip here.
            continue

        # Union of all cited chunks' tokens
        cited_tokens: set[str] = set()
        for cid in cited_chunk_ids:
            cited_tokens |= chunk_tokens_by_id[cid]

        # Claim tokens (excluding the footnote markers themselves)
        claim_no_cites = FOOTNOTE_PATTERN.sub("", claim)
        claim_tokens = _tokenize_for_overlap(claim_no_cites)

        if not claim_tokens:
            continue  # Claim was all stopwords/markers

        overlap = len(claim_tokens & cited_tokens) / len(claim_tokens)
        supported = overlap >= FAITHFULNESS_THRESHOLD

        scores.append({
            "claim": claim,
            "cited": sorted(cited_chunk_ids),
            "overlap": round(overlap, 3),
            "supported": supported,
        })

        if not supported:
            preview = claim[:80] + ("..." if len(claim) > 80 else "")
            warnings.append(
                f"Low overlap ({overlap:.2f}) for claim cited to "
                f"{cited_chunk_ids}: '{preview}'"
            )

    n_unsupported = sum(1 for s in scores if not s["supported"])
    mean_overlap = (
        sum(s["overlap"] for s in scores) / len(scores) if scores else 0.0
    )

    return {
        "scores": scores,
        "n_claims_checked": len(scores),
        "n_unsupported": n_unsupported,
        "uncited_claims": uncited_claims,
        "tagged_claims": tagged_claims,
        "mean_overlap": round(mean_overlap, 3),
        "valid": n_unsupported == 0,
        "warnings": warnings,
    }


# ---- Public API ----------------------------------------------------------

def generate_answer(
    query: str,
    k: int = 5,
    filter: dict | None = None,
    thinking: bool = False,
) -> dict:
    """
    End-to-end RAG: retrieve chunks, threshold-filter, prompt Gemini,
    parse citations, run faithfulness check, return structured result.

    Args:
        query:    natural language question
        k:        max chunks to retrieve from ChromaDB (before threshold filter)
        filter:   optional ChromaDB metadata filter (e.g., {"ticker": "AAPL"})
        thinking: whether to enable Gemini's thinking tokens (Day 9 toggle)

    Returns dict with keys:
        query:            the original query
        answer:           generated answer text (or abstention message)
        citations:        {"[1]": chunk_id, ...}
        citation_health:  {"valid": bool, "warnings": [...]}
        faithfulness:     {scores, n_claims_checked, ..., valid, warnings}
                          or None if we abstained (no Gemini call made)
        chunks_used:      list of chunk dicts actually sent to Gemini
        chunks_retrieved: list of all chunks the retriever returned (pre-filter)
        abstained:        True if we skipped the Gemini call
        usage:            {"tokens_in": N, "tokens_out": N, "tokens_total": N}
        thinking:         True/False, echoes the parameter
    """
    # Step 1: retrieve
    retrieved = retrieve(query, k=k, filter=filter)

    # Step 2: threshold filter
    qualifying = _filter_chunks_by_threshold(retrieved)

    # Step 3: abstain if not enough qualifying chunks
    if len(qualifying) < MIN_CHUNKS_FOR_ANSWER:
        return {
            "query": query,
            "answer": (
                f"No relevant transcript content found for this question. "
                f"Retrieved {len(retrieved)} chunks but only {len(qualifying)} "
                f"met the similarity threshold of {SIMILARITY_THRESHOLD} "
                f"(need at least {MIN_CHUNKS_FOR_ANSWER})."
            ),
            "citations": {},
            "citation_health": {"valid": True, "warnings": []},
            "faithfulness": None,
            "chunks_used": [],
            "chunks_retrieved": retrieved,
            "abstained": True,
            "api_failed": False,
            "usage": {},
            "thinking": thinking,
        }

    # Step 4: build prompt
    chunks_block = _format_chunks_block(qualifying)
    prompt = PROMPT_TEMPLATE.format(question=query, chunks_block=chunks_block)

    # Step 5: call Gemini
    answer_text, usage = _call_gemini(prompt, thinking=thinking)

    # Step 6: parse citations
    retrieved_ids = {c["chunk_id"] for c in qualifying}
    citations, citation_health = _parse_citations(answer_text, retrieved_ids)

    # Step 7: faithfulness check (lexical overlap tripwire)
    faithfulness = check_faithfulness(answer_text, citations, qualifying)

    api_failed = bool(usage.get("api_failed", False))

    return {
        "query": query,
        "answer": answer_text,
        "citations": citations,
        "citation_health": citation_health,
        "faithfulness": None if api_failed else faithfulness,
        "chunks_used": qualifying,
        "chunks_retrieved": retrieved,
        "abstained": False,
        "api_failed": api_failed,
        "usage": usage,
        "thinking": thinking,
    }