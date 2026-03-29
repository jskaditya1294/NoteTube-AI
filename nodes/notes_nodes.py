"""LangGraph nodes: fetch_transcript, fetch_important_frames, generate_notes, review_quality, revise_notes."""

import concurrent.futures
import logging
import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.state import NotesWorkflowState, MAX_ITERATIONS
from services.video import extract_video_id, get_transcript, get_chapters
from services.frames import fetch_important_frames
from prompts.notes_prompts import (
    NOTES_GENERATOR_SYSTEM,
    NOTES_REVISER_SYSTEM,
    REVIEWER_SYSTEM,
    user_prompt_generate_notes,
    user_prompt_revise_notes,
    user_prompt_review,
    NOTES_QA_GENERATOR_SYSTEM,
    user_prompt_generate_qa,
    MERGE_NOTES_SYSTEM,
    user_prompt_generate_chunk_notes,
    user_prompt_merge_chunk_notes,
)
from utils.chunking import build_overlapping_chunks, estimate_tokens, inject_chapter_markers

logger = logging.getLogger(__name__)


# ── Structured output schema for the reviewer ──
class ReviewResult(BaseModel):
    """Structured review verdict from the quality reviewer LLM."""
    passed: bool = Field(
        description="True if the notes meet all quality criteria (PASS), False otherwise (FAIL)."
    )
    feedback: str = Field(
        description=(
            "If passed is False: a numbered list of specific improvements needed. "
            "If passed is True: a one-sentence confirmation of why the notes are high-quality."
        )
    )


# Best choice if available in your deployment (#3: max_retries for resilience)
_generator_llm = ChatOpenAI(model="gpt-5.1", temperature=0.2, max_retries=3)
_reviewer_llm = ChatOpenAI(model="gpt-5.1", temperature=0.0, max_retries=3)
_reviewer_structured = _reviewer_llm.with_structured_output(ReviewResult)

# Qwen 3.5 via Hugging Face router for Q&A generation (#5: lazy init)
_qa_llm: Optional[ChatOpenAI] = None


def _get_qa_llm() -> ChatOpenAI:
    """Lazy init so HF_TOKEN from .env is available even if dotenv loads after import."""
    global _qa_llm
    if _qa_llm is None:
        _qa_llm = ChatOpenAI(
            model="Qwen/Qwen3.5-9B:together",
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN", ""),
            temperature=0.3,
            max_retries=3,
            max_tokens=16384,  # Qwen 3.5 is a thinking model — needs headroom for <think> + answer
        )
    return _qa_llm


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]


def _extract_md_headings(md_text: str) -> list[str]:
    """Extract ## headings from markdown notes."""
    import re
    return re.findall(r'^##\s+(.+)$', md_text, re.MULTILINE)


def _generate_chunk_notes_sync(
    chunk_text: str, idx: int, total_chunks: int,
) -> tuple[int, str, dict]:
    """Synchronous LLM call for a single chunk — run in thread pool for parallelism."""
    messages = [
        SystemMessage(content=NOTES_GENERATOR_SYSTEM),
        HumanMessage(content=user_prompt_generate_chunk_notes(chunk_text, idx, total_chunks)),
    ]
    response = _generator_llm.invoke(messages)
    notes = response.content if hasattr(response, "content") else str(response)
    metadata = {
        "index": idx,
        "token_estimate": estimate_tokens(chunk_text, model_name="gpt-5.1"),
        "preview_start": chunk_text[:120],
        "preview_end": chunk_text[-120:],
    }
    stripped = notes.strip()
    if not stripped:
        logger.warning("Chunk %d/%d returned EMPTY notes (input %d tokens)",
                       idx, total_chunks, metadata["token_estimate"])
    else:
        headings = _extract_md_headings(stripped)
        logger.info("Chunk %d/%d → %d tokens output, headings=%s",
                    idx, total_chunks, estimate_tokens(stripped, model_name="gpt-5.1"), headings)
    return idx, stripped, metadata


def _merge_batch_sync(batch: list[str], batch_label: str = "") -> str:
    """Synchronous LLM call for merging a batch of partial notes."""
    # Collect section headings from all inputs so the merge prompt can audit them
    input_headings: list[str] = []
    for pn in batch:
        input_headings.extend(_extract_md_headings(pn))

    messages = [
        SystemMessage(content=MERGE_NOTES_SYSTEM),
        HumanMessage(content=user_prompt_merge_chunk_notes(batch, input_headings)),
    ]
    response = _generator_llm.invoke(messages)
    merged = response.content if hasattr(response, "content") else str(response)
    merged = merged.strip()

    # Audit: log which headings survived and which were lost
    output_headings = set(_extract_md_headings(merged))
    lost = set(input_headings) - output_headings
    if lost:
        logger.warning("Merge %s LOST headings: %s", batch_label, sorted(lost))
    else:
        logger.info("Merge %s OK — all %d input headings preserved", batch_label, len(input_headings))
    return merged


# Merge batch size: keep small to avoid LLM losing sections.
# With batch=3 and 21 chunks: Round 1 → 7 merged, Round 2 → 3 merged, Round 3 → 1 final.
_MERGE_BATCH_SIZE = 3


def generate_notes_chunked(transcript: str) -> tuple[str, list[str], list[str], list[dict]]:
    chunks = build_overlapping_chunks(
        transcript,
        target_tokens=1400,
        max_tokens=1800,
        overlap_tokens=220,
        model_name="gpt-5.1",
    )

    if not chunks:
        return "", [], [], []

    total_chunks = len(chunks)
    logger.info("Processing %d chunks in parallel", total_chunks)

    # Phase 1: Generate notes for all chunks in parallel via thread pool
    # Per-future error handling: one failure doesn't lose all other results.
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(total_chunks, 8)) as pool:
        futures = {
            pool.submit(_generate_chunk_notes_sync, chunk_text, idx, total_chunks): idx
            for idx, chunk_text in enumerate(chunks, start=1)
        }
        results: list[tuple[int, str, dict]] = []
        for f in concurrent.futures.as_completed(futures):
            chunk_idx = futures[f]
            try:
                results.append(f.result())
            except Exception as e:
                logger.error("Chunk %d FAILED: %s — inserting raw chunk text as fallback", chunk_idx, e)
                # Fallback: pass the raw chunk text so the merge still has something
                results.append((chunk_idx, chunks[chunk_idx - 1], {"index": chunk_idx, "error": str(e)}))

    # Sort by index to preserve order
    results = sorted(results, key=lambda x: x[0])
    partial_notes = [notes for _, notes, _ in results]
    chunk_metadata = [meta for _, _, meta in results]

    # Filter out empty partial notes (log warning but don't silently drop them)
    non_empty = [(i, pn) for i, pn in enumerate(partial_notes) if pn.strip()]
    if len(non_empty) < len(partial_notes):
        empty_indices = [i + 1 for i, pn in enumerate(partial_notes) if not pn.strip()]
        logger.warning("Empty partial notes for chunks: %s", empty_indices)

    logger.info("Phase 1 complete: %d/%d chunks produced notes", len(non_empty), total_chunks)

    # Phase 2: Hierarchical merge (parallel within each round)
    # Use small batch size to prevent LLM from dropping sections.
    current_notes = [pn for pn in partial_notes if pn.strip()]
    merge_round = 0

    while len(current_notes) > 1:
        merge_round += 1
        batches_list = list(batched(current_notes, _MERGE_BATCH_SIZE))
        logger.info("Merge round %d: %d notes → %d batches (batch_size=%d)",
                    merge_round, len(current_notes), len(batches_list), _MERGE_BATCH_SIZE)

        # Merge multi-item batches in parallel, pass through single-item batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batches_list), 6)) as pool:
            future_map = {}
            pass_through = {}
            for i, batch in enumerate(batches_list):
                if len(batch) == 1:
                    pass_through[i] = batch[0]
                else:
                    label = f"R{merge_round}B{i}"
                    future_map[pool.submit(_merge_batch_sync, batch, label)] = i

            merged_map = {}
            for f in concurrent.futures.as_completed(future_map):
                bi = future_map[f]
                try:
                    merged_map[bi] = f.result()
                except Exception as e:
                    logger.error("Merge R%dB%d FAILED: %s — concatenating inputs as fallback",
                                 merge_round, bi, e)
                    # Fallback: concatenate the batch items so nothing is lost
                    merged_map[bi] = "\n\n".join(batches_list[bi])

        # Reconstruct order
        current_notes = []
        for i in range(len(batches_list)):
            if i in pass_through:
                current_notes.append(pass_through[i])
            else:
                current_notes.append(merged_map[i])

    final_notes = current_notes[0] if current_notes else ""
    return final_notes, chunks, partial_notes, chunk_metadata
 


def fetch_transcript(state: NotesWorkflowState) -> dict:
    raw_input = state.get("video_id") or ""
    if not raw_input:
        return {"error": "Missing video_id", "transcript": "", "chapters": []}

    try:
        video_id = extract_video_id(raw_input)
        transcript = get_transcript(video_id)
        chapters = get_chapters(video_id)
        return {"video_id": video_id, "transcript": transcript, "chapters": chapters, "error": ""}
    except Exception as e:
        return {"transcript": "", "chapters": [], "error": str(e)}


def fetch_important_frames_node(state: NotesWorkflowState) -> dict:
    """Extract most useful frames: SSIM scene detection, batched vision scoring, auto-cleanup video."""
    video_id = state.get("video_id") or ""
    if not video_id:
        return {"important_frames": [], "frames_error": "Missing video_id"}
    output_dir = state.get("output_dir") or "outputs"
    try:
        important, err = fetch_important_frames(
            video_id=video_id,
            output_dir=Path(output_dir),
        )
        return {"important_frames": important, "frames_error": err or ""}
    except Exception as e:
        return {"important_frames": [], "frames_error": str(e)}


def generate_notes(state: NotesWorkflowState) -> dict:
    """LLM 1: generate initial notes from transcript using chunk -> summarize -> merge."""
    transcript = state.get("transcript") or ""
    if not transcript:
        return {"notes": "", "error": "Transcript is empty"}

    # Inject YouTube chapter markers so the chunker can respect topic boundaries
    chapters = state.get("chapters") or []
    if chapters:
        transcript_for_chunking = inject_chapter_markers(transcript, chapters)
        logger.info("Injected %d chapter markers into transcript", len(chapters))
    else:
        transcript_for_chunking = transcript

    final_notes, chunks, partial_notes, chunk_metadata = generate_notes_chunked(transcript_for_chunking)

    return {
        "notes": final_notes,
        "chunks": chunks,
        "partial_notes": partial_notes,
        "chunk_metadata": chunk_metadata,
    }



def review_quality(state: NotesWorkflowState) -> dict:
    """LLM 2: review notes via structured output; set review_passed and review_feedback."""
    # #8: Send only a transcript excerpt — the reviewer primarily audits the notes,
    # not the full transcript. This saves tokens on long videos.
    transcript = (state.get("transcript") or "")[:15000]
    notes = state.get("notes") or ""

    if not notes:
        return {"review_passed": False, "review_feedback": "No notes were generated."}

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=user_prompt_review(transcript, notes)),
    ]

    try:
        result: ReviewResult = _reviewer_structured.invoke(messages)
        return {"review_passed": result.passed, "review_feedback": result.feedback}
    except Exception:
        # Fallback: if structured output fails, treat as fail so we don't silently pass bad notes
        return {"review_passed": False, "review_feedback": "Review LLM did not return a valid structured response. Treating as FAIL."}


def revise_notes(state: NotesWorkflowState) -> dict:
    """LLM 1: revise notes using reviewer feedback."""
    # #4: Send only a trimmed transcript excerpt — the reviser mainly needs the notes
    # + feedback. A short transcript reference suffices for spot-checking.
    transcript = (state.get("transcript") or "")[:20000]
    notes = state.get("notes") or ""
    feedback = state.get("review_feedback") or ""
    iteration = state.get("iteration") or 0

    messages = [
        SystemMessage(content=NOTES_REVISER_SYSTEM),
        HumanMessage(content=user_prompt_revise_notes(transcript, notes, feedback)),
    ]
    response = _generator_llm.invoke(messages)
    revised = response.content if hasattr(response, "content") else str(response)
    return {"notes": revised, "iteration": iteration + 1}


def generate_notes_qa(state: NotesWorkflowState) -> dict:
    """Generate interview-style Q&A from the finalized notes using Qwen 3.5."""
    notes = state.get("notes") or ""
    if not notes.strip():
        return {"notes_qa_markdown": ""}

    messages = [
        SystemMessage(content=NOTES_QA_GENERATOR_SYSTEM),
        HumanMessage(content=user_prompt_generate_qa(notes)),
    ]
    response = _get_qa_llm().invoke(messages)
    qa_text = response.content if hasattr(response, "content") else str(response)
    return {"notes_qa_markdown": qa_text.strip()}