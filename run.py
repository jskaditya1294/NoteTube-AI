

"""CLI: accept video ID or URL, run workflow, print final notes, save final notes, save graph PNG."""

import argparse
import re
from pathlib import Path

# Load .env first so OPENAI_API_KEY and TAVILY_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from video import extract_video_id
from graph import graph
from state import NotesWorkflowState
from export_utils import (
    save_notes_as_md_and_docx,
    save_combined_questions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube transcript → notes with quality review"
    )
    parser.add_argument(
        "video",
        help="YouTube video ID or URL (e.g. https://www.youtube.com/watch?v=...)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder where final notes files will be saved"
    )
    parser.add_argument(
        "--save-graph-png",
        action="store_true",
        help="Save the LangGraph workflow as a PNG flowchart"
    )
    parser.add_argument(
        "--save-graph-mermaid",
        action="store_true",
        help="Save the LangGraph workflow Mermaid text to a .mmd file"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=6,
        help="Max number of key images to include in notes (default: 6)"
    )
    parser.add_argument(
        "--skip-interview",
        action="store_true",
        help="Skip interview question mining (web search + Node1–3 graph)"
    )

    args = parser.parse_args()

    try:
        video_id = extract_video_id(args.video)
    except ValueError as e:
        print("Error:", str(e))
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save graph as PNG flowchart
    try:
        png_path = out_dir / "workflow_graph.png"
        graph.get_graph().draw_mermaid_png(output_file_path=str(png_path))
        print(f"Saved graph PNG to: {png_path.resolve()}")
    except Exception as e:
        print("Graph PNG save failed:", str(e))

    # Save graph as Mermaid text
    if args.save_graph_mermaid:
        try:
            mermaid_path = out_dir / "workflow_graph.mmd"
            mermaid_path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
            print(f"Saved Mermaid graph to: {mermaid_path.resolve()}")
        except Exception as e:
            print("Graph Mermaid save failed:", str(e))

    initial_state: NotesWorkflowState = {
        "video_id": video_id,
        "transcript": "",
        "output_dir": str(out_dir.resolve()),
        "max_important_frames": args.max_images,
        "important_frames": [],
        "frames_error": "",
        "notes": "",
        "review_passed": False,
        "review_feedback": "",
        "iteration": 0,
        "error": "",
        "skip_interview": args.skip_interview,
        "interview_loop": 0,
        "interview_cycles": 0,
        "feedback_node1": "",
        "node2_extra_queries": [],
    }

    result = graph.invoke(initial_state)

    if result.get("error"):
        print("Error:", result["error"])
        return

    if result.get("frames_error"):
        print("Frames warning:", result["frames_error"])
    important_frames = result.get("important_frames") or []
    if important_frames:
        print(f"Key images captured: {len(important_frames)}")

    final_notes = result.get("notes", "") or ""

    print("Final notes (iteration", result.get("iteration", 0), "):")
    print("---")
    print(final_notes or "(no notes)")
    print("---")

    if final_notes.strip():
        try:
            # Extract topic title from notes' first heading
            topic_title = ""
            for line in final_notes.splitlines():
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    topic_title = stripped[2:].strip()
                    break
            if not topic_title:
                topic_title = f"YouTube_Notes_{video_id}"

            # Create topic folder under output dir
            safe_topic = re.sub(r'[<>:"/\\|?*\n]+', "_", topic_title)
            safe_topic = re.sub(r"\s+", "_", safe_topic)[:120]
            topic_dir = out_dir / safe_topic
            topic_dir.mkdir(parents=True, exist_ok=True)

            # Save notes: <topic>_notes.md/docx
            saved = save_notes_as_md_and_docx(
                notes=final_notes,
                output_dir=str(topic_dir),
                video_id=video_id,
                title=topic_title,
                important_frames=important_frames,
            )
            print("Saved markdown:", saved["markdown_path"])
            print("Saved docx    :", saved["docx_path"])

            # Save combined questions: <topic>_questions.md/docx
            qa_md = (result.get("notes_qa_markdown") or "").strip()
            bank_md = (result.get("interview_bank_markdown") or "").strip()
            if qa_md or bank_md:
                q_saved = save_combined_questions(
                    output_dir=str(topic_dir),
                    qa_markdown=qa_md,
                    interview_markdown=bank_md,
                    title=topic_title,
                )
                print("Questions MD  :", q_saved["questions_md"])
                print("Questions DOCX:", q_saved["questions_docx"])
        except Exception as e:
            print("Saving failed:", str(e))
    else:
        print("No notes generated, so nothing was saved.")


if __name__ == "__main__":
    main()