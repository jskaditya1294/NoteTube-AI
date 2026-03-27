"""Extract the most useful frames from a YouTube video: scene-based sampling, vision filter, top-N cap."""

from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

VISION_MODEL = "gpt-4o"
FRAME_MAX_WIDTH = 768

# At most this many images in the final notes (enforced via second-pass "top N" selection)
MAX_IMPORTANT_FRAMES = 6

# Scene change: min mean pixel diff (grayscale) to consider a new scene; tune if too many/few
SCENE_DIFF_THRESHOLD = 25.0
# Min seconds between saved scene frames (avoid duplicate slides)
MIN_SCENE_GAP_SEC = 12.0
# Fallback: if scene detection yields very few frames, sample by interval (seconds)
FALLBACK_INTERVAL_SEC = 45.0
MAX_CANDIDATE_FRAMES = 25


def download_video(video_id: str, output_dir: Path) -> Path:
    """Download YouTube video with yt-dlp to output_dir / video_id / video.mp4."""
    out_dir = output_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mp4"
    cmd = [
        sys.executable, "-m", "yt_dlp", "--no-playlist",
        "-f", "best[height<=720]/best[height<=480]/best",
        "--output", str(video_path), "--no-overwrites",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr or result.stdout}")
    if not video_path.exists():
        raise FileNotFoundError(f"Download completed but file not found: {video_path}")
    return video_path


def extract_frames_at_scene_changes(
    video_path: Path,
    min_gap_sec: float = MIN_SCENE_GAP_SEC,
    diff_threshold: float = SCENE_DIFF_THRESHOLD,
    max_frames: int = MAX_CANDIDATE_FRAMES,
    fallback_interval_sec: float = FALLBACK_INTERVAL_SEC,
) -> list[tuple[Path, float]]:
    """
    Extract one frame per scene change (large visual change) so we get one frame per slide/diagram
    instead of many from the same content. Falls back to interval-based sampling if too few scenes.
    Returns list of (frame_path, timestamp_sec).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("opencv-python and numpy required. pip install opencv-python numpy")

    frames_dir = video_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    min_gap_frames = max(1, int(fps * min_gap_sec))
    results: list[tuple[Path, float]] = []
    prev_gray = None
    frame_index = 0
    last_saved_at_frame = -min_gap_frames - 1

    while len(results) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = frame_index / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        is_scene_change = False
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff >= diff_threshold and (frame_index - last_saved_at_frame) >= min_gap_frames:
                is_scene_change = True
        else:
            is_scene_change = True

        if is_scene_change:
            last_saved_at_frame = frame_index
            name = f"frame_{len(results) + 1:03d}_{int(t_sec)}s.jpg"
            out_path = frames_dir / name
            h, w = frame.shape[:2]
            if w > FRAME_MAX_WIDTH:
                scale = FRAME_MAX_WIDTH / w
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            results.append((out_path, t_sec))

        prev_gray = gray
        frame_index += 1

    cap.release()

    # If scene detection gave very few frames (e.g. static lecture), fallback to interval sampling
    if len(results) < 5:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval_frames = max(1, int(fps * fallback_interval_sec))
        results = []
        frame_index = 0
        saved = 0
        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            t_sec = frame_index / fps
            if frame_index % interval_frames == 0:
                name = f"frame_{saved + 1:03d}_{int(t_sec)}s.jpg"
                out_path = frames_dir / name
                h, w = frame.shape[:2]
                if w > FRAME_MAX_WIDTH:
                    scale = FRAME_MAX_WIDTH / w
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                results.append((out_path, t_sec))
                saved += 1
            frame_index += 1
        cap.release()

    return results


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


# Strict prompt: no face-only, only complete content; reduces false positives
FRAME_IMPORTANCE_SYSTEM = """You decide if a video frame is worth keeping for study notes.

Mark as NOT important:
- Mainly a person's face or talking head with no diagram/slide/code visible.
- Partial or incomplete content (diagram being drawn, slide with one bullet, unfinished equation).
- Blank screen, duplicate of previous, intro/outro, or no educational content.

Mark as IMPORTANT only when the frame clearly shows:
- A complete diagram, flowchart, or architecture diagram.
- A full slide with bullet points or key concepts.
- A complete equation, whiteboard, or chart/table.
- Full code snippet or terminal output that explains something.

Reply with exactly one line: IMPORTANT or NOT. If IMPORTANT, add a second line with a short caption (e.g. "Training loop flowchart", "Gradient descent equation")."""


def _is_important_frame_response(text: str) -> tuple[bool, str]:
    text = (text or "").strip().upper()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False, ""
    first = lines[0].upper()
    if "NOT" in first or first.startswith("NO"):
        return False, ""
    if "IMPORTANT" in first or "YES" in first:
        caption = " ".join(ln for ln in lines[1:] if not ln.upper().startswith("IMPORTANT")).strip()
        return True, caption or "Key frame"
    return False, ""


def select_important_frames(
    candidate_frames: Sequence[tuple[Path, float]],
    vision_model: str = VISION_MODEL,
    max_important: int = MAX_IMPORTANT_FRAMES,
) -> list[dict]:
    """
    First pass: vision LLM marks each frame IMPORTANT/NOT. Then deduplicate by time cluster
    (keep last frame in each cluster). If still more than max_important, second pass: text-only
    "pick top N" to keep only the most useful. Returns list of {"path", "timestamp_sec", "caption"}.
    """
    if not candidate_frames:
        return []

    llm = ChatOpenAI(model=vision_model, temperature=0.0, max_tokens=120)
    important: list[dict] = []

    for frame_path, timestamp_sec in candidate_frames:
        if not frame_path.exists():
            continue
        b64 = _encode_image(frame_path)
        data_url = f"data:image/jpeg;base64,{b64}"
        user_content = [
            {"type": "text", "text": f"Timestamp: {timestamp_sec:.0f}s. Is this frame IMPORTANT for study notes? Mark NOT if mainly a face with no content, or partial/incomplete. Mark IMPORTANT only if it shows complete educational content (full diagram/slide/equation). Reply IMPORTANT or NOT; if IMPORTANT add one short caption on next line."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        messages = [SystemMessage(content=FRAME_IMPORTANCE_SYSTEM), HumanMessage(content=user_content)]
        try:
            response = llm.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            is_imp, caption = _is_important_frame_response(text)
            if is_imp:
                important.append({
                    "path": str(frame_path.resolve()),
                    "timestamp_sec": timestamp_sec,
                    "caption": caption or f"Frame at {timestamp_sec:.0f}s",
                })
        except Exception:
            continue

    # Deduplicate: within 50s keep only the last (complete slide vs build-up)
    important = _cluster_keep_last(important, window_sec=50.0)

    # Hard cap: if still too many, ask LLM to pick top N by caption/timestamp (text-only)
    if len(important) > max_important:
        important = _pick_top_n(important, max_n=max_important)
    return important


def _cluster_keep_last(frames: list[dict], window_sec: float) -> list[dict]:
    if len(frames) <= 1:
        return frames
    sorted_f = sorted(frames, key=lambda x: x["timestamp_sec"])
    out: list[dict] = []
    cluster_start = sorted_f[0]["timestamp_sec"]
    best = sorted_f[0]
    for f in sorted_f[1:]:
        t = f["timestamp_sec"]
        if t <= cluster_start + window_sec:
            best = f
        else:
            out.append(best)
            cluster_start = t
            best = f
    out.append(best)
    return out


def _pick_top_n(frames: list[dict], max_n: int) -> list[dict]:
    """Use text-only LLM to pick the top max_n most useful frames by caption/timestamp."""
    if len(frames) <= max_n:
        return frames
    from prompts.notes import FRAME_TOP_N_SYSTEM
    lines = [f"{i+1}. [{int(f['timestamp_sec'])}s] {f.get('caption', '')}" for i, f in enumerate(frames)]
    prompt = f"""Candidates (number, timestamp, caption):
{chr(10).join(lines)}

Pick exactly the {max_n} most useful for study notes. Reply with comma-separated numbers only, e.g. 1,3,5,7,8,10."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=80)
    try:
        response = llm.invoke([SystemMessage(content=FRAME_TOP_N_SYSTEM), HumanMessage(content=prompt)])
        text = (response.content if hasattr(response, "content") else str(response)) or ""
        numbers = [int(x.strip()) for x in re.findall(r"\d+", text) if 1 <= int(x.strip()) <= len(frames)]
        seen = set()
        indices = []
        for n in numbers:
            if n not in seen and len(indices) < max_n:
                seen.add(n)
                indices.append(n)
        if indices:
            return [frames[i - 1] for i in indices]
    except Exception:
        pass
    return frames[:max_n]


def fetch_important_frames(
    video_id: str,
    output_dir: Path | str,
    max_important: int = MAX_IMPORTANT_FRAMES,
    vision_model: str = VISION_MODEL,
) -> tuple[list[dict], str]:
    """
    Download video, extract frames at scene changes (with interval fallback), run vision filter,
    cluster dedup, then top-N cap. Returns (list of frames, error_string).
    """
    output_dir = Path(output_dir)
    try:
        video_path = download_video(video_id, output_dir)
    except Exception as e:
        return [], f"Video download failed: {e}"
    try:
        candidates = extract_frames_at_scene_changes(video_path, max_frames=MAX_CANDIDATE_FRAMES)
    except Exception as e:
        return [], f"Frame extraction failed: {e}"
    if not candidates:
        return [], ""
    try:
        important = select_important_frames(candidates, vision_model=vision_model, max_important=max_important)
    except Exception as e:
        return [], f"Frame selection failed: {e}"
    return important, ""
