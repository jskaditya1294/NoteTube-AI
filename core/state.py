"""State schema for the YouTube notes workflow (notes + frames + interview question mining)."""

from typing import Any, TypedDict

MAX_ITERATIONS = 3
MAX_INTERVIEW_LOOPS = 3

#NotesWorkflowState is not “data” — it is the workflow’s memory.
#Every step reads from it, writes to it, and passes it forward.


class NotesWorkflowState(TypedDict, total=False):
    # Notes pipeline
    video_id: str
    transcript: str
    chapters: list[dict]
    output_dir: str
    max_important_frames: int
    important_frames: list[dict]
    frames_error: str
    chunks: list[str]
    partial_notes: list[str]
    chunk_metadata: list[dict]
    notes: str
    review_passed: bool
    review_feedback: str
    iteration: int
    error: str
    # Interview question mining (runs after notes are finalized; uses transcript + notes above)
    topics_data: dict[str, Any]
    question_bank: list[dict]
    excluded_unattributed: list[dict]
    search_coverage: list[dict]
    raw_search_blob: str
    critic_decision: str
    critic_full: dict[str, Any]
    interview_loop: int
    interview_cycles: int
    feedback_node1: str
    node2_extra_queries: list[str]
    interview_bank_markdown: str
    skip_interview: bool
    # Notes-based QA (runs after notes are finalized)
    notes_qa_markdown: str