

"""CLI: accept video ID or URL, run workflow, print final notes, save final notes, save graph PNG."""

import argparse
import logging
import re
import shutil
from pathlib import Path

# Load .env first so OPENAI_API_KEY and TAVILY_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from services.video import extract_video_id
from core.graph import graph
from core.state import NotesWorkflowState
from utils.export import (
    save_notes_as_md_and_docx,
    save_combined_questions,
    _safe_filename,
)

logger = logging.getLogger(__name__)


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

    # Always save workflow graph at the start of every run
    try:
        png_path = out_dir / "workflow_graph.png"
        graph.get_graph().draw_mermaid_png(output_file_path=str(png_path))
        print(f"Workflow graph saved to: {png_path.resolve()}")
    except Exception as e:
        print(f"Workflow graph save failed: {e}")

    initial_state: NotesWorkflowState = {
        "video_id": video_id,
        "transcript": "",
        "output_dir": str(out_dir.resolve()),
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

    # #10: Wrap graph invocation with error handling
    try:
        result = graph.invoke(initial_state)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return
    except Exception as e:
        logger.exception("Workflow failed")
        print(f"Workflow failed: {e}")
        return

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

            # Create topic folder under output dir (#20: reuse _safe_filename from export)
            safe_topic = _safe_filename(topic_title)
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

            # Move extracted frames into the topic folder
            if important_frames:
                frames_dest = topic_dir / f"imp_frames_{safe_topic}"
                frames_dest.mkdir(parents=True, exist_ok=True)
                moved = 0
                for frame in important_frames:
                    src = Path(frame.get("path", ""))
                    if src.exists():
                        dst = frames_dest / src.name
                        shutil.move(str(src), str(dst))
                        frame["path"] = str(dst.resolve())
                        moved += 1
                if moved:
                    print(f"Moved {moved} frames to: {frames_dest.resolve()}")

                # Clean up the old video_id folder if empty
                old_frames_dir = out_dir / video_id / "frames"
                if old_frames_dir.exists() and not any(old_frames_dir.iterdir()):
                    old_frames_dir.rmdir()
                old_video_dir = out_dir / video_id
                if old_video_dir.exists() and not any(old_video_dir.iterdir()):
                    old_video_dir.rmdir()

                # Re-save markdown so image paths point to the new location
                saved = save_notes_as_md_and_docx(
                    notes=final_notes,
                    output_dir=str(topic_dir),
                    video_id=video_id,
                    title=topic_title,
                    important_frames=important_frames,
                )

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