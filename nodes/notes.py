"""LangGraph nodes: fetch_transcript, fetch_important_frames, generate_notes, review_quality, revise_notes."""

from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.state import NotesWorkflowState, MAX_ITERATIONS
from services.video import extract_video_id, get_transcript
from services.frames import fetch_important_frames
from prompts.notes import (
    NOTES_GENERATOR_SYSTEM,
    NOTES_REVISER_SYSTEM,
    REVIEWER_SYSTEM,
    user_prompt_generate_notes,
    user_prompt_revise_notes,
    user_prompt_review,
    parse_review_response,
    NOTES_QA_GENERATOR_SYSTEM,
    user_prompt_generate_qa,
    MERGE_NOTES_SYSTEM,
    user_prompt_generate_chunk_notes,
    user_prompt_merge_chunk_notes,
)
from utils.chunking import build_overlapping_chunks, estimate_tokens


# Best choice if available in your deployment
_generator_llm = ChatOpenAI(model="gpt-5.1", temperature=0.2)
_reviewer_llm = ChatOpenAI(model="gpt-5.1", temperature=0.0)


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]
        
    

def generate_notes_chunked(transcript: str) -> tuple[str, list[str], list[str], list[dict]]:
    chunks = build_overlapping_chunks(
        transcript,
        target_tokens=1400,
        max_tokens=1800,
        overlap_tokens=220,
        model_name="gpt-5.1",
    )

    partial_notes: list[str] = []
    chunk_metadata: list[dict] = []

    total_chunks = len(chunks)

    for idx, chunk_text in enumerate(chunks, start=1):
        messages = [
            SystemMessage(content=NOTES_GENERATOR_SYSTEM),
            HumanMessage(content=user_prompt_generate_chunk_notes(chunk_text, idx, total_chunks)),
        ]
        response = _generator_llm.invoke(messages)
        notes = response.content if hasattr(response, "content") else str(response)
        partial_notes.append(notes.strip())

        chunk_metadata.append({
            "index": idx,
            "token_estimate": estimate_tokens(chunk_text, model_name="gpt-5.1"),
            "preview_start": chunk_text[:120],
            "preview_end": chunk_text[-120:],
        })

    current_notes = partial_notes[:]

    while len(current_notes) > 1:
        merged_round: list[str] = []

        for batch in batched(current_notes, 6):
            if len(batch) == 1:
                merged_round.append(batch[0])
                continue

            messages = [
                SystemMessage(content=MERGE_NOTES_SYSTEM),
                HumanMessage(content=user_prompt_merge_chunk_notes(batch)),
            ]
            response = _generator_llm.invoke(messages)
            merged = response.content if hasattr(response, "content") else str(response)
            merged_round.append(merged.strip())

        current_notes = merged_round

    final_notes = current_notes[0] if current_notes else ""
    return final_notes, chunks, partial_notes, chunk_metadata
 


def fetch_transcript(state: NotesWorkflowState) -> dict:
    raw_input = state.get("video_id") or ""
    if not raw_input:
        return {"error": "Missing video_id", "transcript": ""}

    try:
        video_id = extract_video_id(raw_input)
        transcript = get_transcript(video_id)
        return {"video_id": video_id, "transcript": transcript, "error": ""}
    except Exception as e:
        return {"transcript": "", "error": str(e)}


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

    final_notes, chunks, partial_notes, chunk_metadata = generate_notes_chunked(transcript)

    return {
        "notes": final_notes,
        "chunks": chunks,
        "partial_notes": partial_notes,
        "chunk_metadata": chunk_metadata,
    }



def review_quality(state: NotesWorkflowState) -> dict:
    """LLM 2: review notes; set review_passed and review_feedback."""
    transcript = state.get("transcript") or ""
    notes = state.get("notes") or ""
    
    if not notes:
        return {"review_passed": False, "review_feedback": "No notes were generated."}
    
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=user_prompt_review(transcript, notes)),
    ]
    response = _reviewer_llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    passed, feedback = parse_review_response(text)
    return {"review_passed": passed, "review_feedback": feedback}


def revise_notes(state: NotesWorkflowState) -> dict:
    """LLM 1: revise notes using reviewer feedback."""
    transcript = state.get("transcript") or ""
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
    """Generate interview-style Q&A from the finalized notes."""
    notes = state.get("notes") or ""
    if not notes.strip():
        return {"notes_qa_markdown": ""}

    messages = [
        SystemMessage(content=NOTES_QA_GENERATOR_SYSTEM),
        HumanMessage(content=user_prompt_generate_qa(notes)),
    ]
    response = _generator_llm.invoke(messages)
    qa_text = response.content if hasattr(response, "content") else str(response)
    return {"notes_qa_markdown": qa_text.strip()}