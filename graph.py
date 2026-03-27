"""Single LangGraph workflow: notes pipeline → interview question mining (uses same transcript + notes)."""

from typing import Literal

from langgraph.graph import StateGraph, START, END

from state import NotesWorkflowState, MAX_ITERATIONS
from nodes import fetch_transcript, fetch_important_frames_node, generate_notes, review_quality, revise_notes, generate_notes_qa
from interview_nodes import (
    interview_topic_miner,
    interview_question_harvester,
    interview_critic,
    interview_format_export,
)


def route_after_review(state: NotesWorkflowState) -> Literal["pass", "pass_skip_interview", "revise"]:
    """If review passed (or max iterations), run interview mining or END if skip_interview; else revise."""
    if state.get("review_passed") or (state.get("iteration") or 0) >= MAX_ITERATIONS:
        return "pass_skip_interview" if state.get("skip_interview") else "pass"
    return "revise"


def route_after_notes_qa(state: NotesWorkflowState) -> Literal["interview", "end"]:
    """After notes QA, continue to interview mining or end."""
    return "end" if state.get("skip_interview") else "interview"


def route_after_fetch(state: NotesWorkflowState) -> Literal["continue", "end"]:
    if state.get("error"):
        return "end"
    return "continue"


def route_after_critic(state: NotesWorkflowState) -> Literal["export", "revise", "research"]:
    """Interview critic: export, loop to topic miner, or loop to harvester."""
    decision = state.get("critic_decision") or "APPROVE_AND_EXPORT"
    cycles = state.get("interview_cycles") or 0
    if decision == "APPROVE_AND_EXPORT" or cycles >= 3:
        return "export"
    if decision == "REVISE_TOPICS":
        return "revise"
    return "research"


workflow = StateGraph(NotesWorkflowState)
# Notes pipeline
workflow.add_node("fetch_transcript", fetch_transcript)
workflow.add_node("fetch_important_frames", fetch_important_frames_node)
workflow.add_node("generate_notes", generate_notes)
workflow.add_node("review_quality", review_quality)
workflow.add_node("revise_notes", revise_notes)
workflow.add_node("generate_notes_qa", generate_notes_qa)
# Interview question mining (uses state["transcript"] and state["notes"] from above)
workflow.add_node("interview_topic_miner", interview_topic_miner)
workflow.add_node("interview_question_harvester", interview_question_harvester)
workflow.add_node("interview_critic", interview_critic)
workflow.add_node("interview_format_export", interview_format_export)

workflow.add_edge(START, "fetch_transcript")
workflow.add_conditional_edges(
    "fetch_transcript",
    route_after_fetch,
    {"continue": "fetch_important_frames", "end": END},
)
workflow.add_edge("fetch_important_frames", "generate_notes")
workflow.add_edge("generate_notes", "review_quality")
workflow.add_conditional_edges(
    "review_quality",
    route_after_review,
    {
        "pass": "generate_notes_qa",
        "pass_skip_interview": "generate_notes_qa",
        "revise": "revise_notes",
    },
)
workflow.add_edge("revise_notes", "review_quality")

# After notes QA: either continue to interview mining or end
workflow.add_conditional_edges(
    "generate_notes_qa",
    route_after_notes_qa,
    {
        "interview": "interview_topic_miner",
        "end": END,
    },
)

# Interview: topic miner uses notes + transcript; then harvester → critic (loop) → format → END
workflow.add_edge("interview_topic_miner", "interview_question_harvester")
workflow.add_edge("interview_question_harvester", "interview_critic")
workflow.add_conditional_edges(
    "interview_critic",
    route_after_critic,
    {
        "export": "interview_format_export",
        "revise": "interview_topic_miner",
        "research": "interview_question_harvester",
    },
)
workflow.add_edge("interview_format_export", END)

graph = workflow.compile()