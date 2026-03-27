"""
Production-grade frame extraction from YouTube videos.

Pipeline:
1. Download video via yt-dlp
2. Adaptive scene-change detection using SSIM + motion debouncing
3. Batched vision-model filtering (GPT-4o scores frames 1-10 for information density)
4. Deduplication built into the judge prompt
5. Top-N cap via text-only LLM if needed

Async API calls via asyncio for concurrent vision batches.
Generator pattern for long videos to bound memory.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Generator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VISION_MODEL = "gpt-4o"
FRAME_MAX_WIDTH = 768

# SSIM-based scene change thresholds
SSIM_CHANGE_THRESHOLD = 0.80       # Below this SSIM → significant visual shift
MIN_SCENE_GAP_SEC = 3.0            # Min gap between candidate captures
MOTION_STASIS_WINDOW = 8           # Frames to confirm visual stasis after a change
MOTION_STASIS_SSIM = 0.95          # Consecutive frames must exceed this to be "stable"

# Fallback for very static videos (e.g. single camera, no slides)
FALLBACK_INTERVAL_SEC = 40.0
MIN_CANDIDATES_BEFORE_FALLBACK = 5

# Budget
MAX_CANDIDATE_FRAMES = 30
VISION_BATCH_SIZE = 6              # Frames per GPT-4o call
RELEVANCE_THRESHOLD = 6            # Min score (1-10) to keep a frame


# ---------------------------------------------------------------------------
# 1. Video download
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. SSIM-based adaptive scene-change detection with motion debouncing
# ---------------------------------------------------------------------------
def _compute_ssim(img_a, img_b) -> float:
    """Compute simplified SSIM between two grayscale images (same shape).

    Uses the standard SSIM formula with default constants.
    Returns value in [-1, 1]; 1 = identical.
    """
    import numpy as np

    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a_sq = a.var()
    sigma_b_sq = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()

    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(num / den)


def _frame_generator(
    video_path: Path,
    sample_fps: float = 2.0,
) -> Generator[tuple[int, float, "np.ndarray"], None, None]:
    """Yield (frame_index, timestamp_sec, bgr_frame) sampled at ~sample_fps from the video.

    Uses a generator to avoid loading the entire video into memory.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # How many source frames to skip between samples
    step = max(1, int(round(fps / sample_fps)))
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % step == 0:
            t_sec = frame_index / fps
            yield frame_index, t_sec, frame
        frame_index += 1

    cap.release()


def extract_candidate_frames(
    video_path: Path,
    ssim_threshold: float = SSIM_CHANGE_THRESHOLD,
    min_gap_sec: float = MIN_SCENE_GAP_SEC,
    stasis_window: int = MOTION_STASIS_WINDOW,
    stasis_ssim: float = MOTION_STASIS_SSIM,
    max_frames: int = MAX_CANDIDATE_FRAMES,
    fallback_interval_sec: float = FALLBACK_INTERVAL_SEC,
) -> list[tuple[Path, float]]:
    """
    Adaptive scene-change detection:
    1. Sample frames at ~2 FPS
    2. Compute SSIM between consecutive samples
    3. When SSIM drops below threshold → scene change detected
    4. Motion debouncing: after detecting change, wait for visual stasis
       (consecutive high-SSIM frames) before capturing the *stable* frame
    5. Fallback: if too few candidates, switch to interval-based sampling

    Returns list of (frame_path, timestamp_sec).
    """
    import cv2
    import numpy as np

    frames_dir = video_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Path, float]] = []
    prev_gray = None
    last_saved_sec = -min_gap_sec - 1.0

    # State for motion debouncing
    change_detected = False
    stasis_count = 0
    pending_frame = None       # (bgr_frame, t_sec) — candidate waiting for stasis confirmation
    pending_prev_gray = None

    def _save_frame(bgr_frame, t_sec: float) -> None:
        nonlocal last_saved_sec
        if len(results) >= max_frames:
            return
        name = f"frame_{len(results) + 1:03d}_{int(t_sec)}s.jpg"
        out_path = frames_dir / name
        h, w = bgr_frame.shape[:2]
        if w > FRAME_MAX_WIDTH:
            scale = FRAME_MAX_WIDTH / w
            bgr_frame = cv2.resize(bgr_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results.append((out_path, t_sec))
        last_saved_sec = t_sec

    for _idx, t_sec, bgr_frame in _frame_generator(video_path, sample_fps=2.0):
        if len(results) >= max_frames:
            break

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))  # Higher res than before for better SSIM

        if prev_gray is None:
            # Always save the first frame
            _save_frame(bgr_frame, t_sec)
            prev_gray = gray
            continue

        ssim = _compute_ssim(prev_gray, gray)

        if change_detected:
            # Waiting for stasis: consecutive frames with high SSIM
            if ssim >= stasis_ssim:
                stasis_count += 1
                pending_frame = (bgr_frame, t_sec)
                pending_prev_gray = gray
            else:
                # Still changing — reset stasis counter, update pending
                stasis_count = 0
                pending_frame = (bgr_frame, t_sec)
                pending_prev_gray = gray

            if stasis_count >= stasis_window:
                # Stable state reached — save the final stable frame
                if pending_frame is not None:
                    pf, pt = pending_frame
                    if (pt - last_saved_sec) >= min_gap_sec:
                        _save_frame(pf, pt)
                change_detected = False
                stasis_count = 0
                pending_frame = None
                if pending_prev_gray is not None:
                    prev_gray = pending_prev_gray
                    pending_prev_gray = None
                continue
        else:
            if ssim < ssim_threshold:
                # Scene change detected — start debouncing
                change_detected = True
                stasis_count = 0
                pending_frame = (bgr_frame, t_sec)
                pending_prev_gray = gray

        prev_gray = gray

    # Flush: if we ended while waiting for stasis, save the pending frame
    if change_detected and pending_frame is not None:
        pf, pt = pending_frame
        if (pt - last_saved_sec) >= min_gap_sec and len(results) < max_frames:
            _save_frame(pf, pt)

    # Fallback: if scene detection yielded very few candidates (e.g. static lecture)
    if len(results) < MIN_CANDIDATES_BEFORE_FALLBACK:
        import cv2 as cv2_fb

        cap = cv2_fb.VideoCapture(str(video_path))
        fps = cap.get(cv2_fb.CAP_PROP_FPS) or 25.0
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
                    frame = cv2_fb.resize(
                        frame, None, fx=FRAME_MAX_WIDTH / w, fy=FRAME_MAX_WIDTH / w,
                        interpolation=cv2_fb.INTER_AREA,
                    )
                cv2_fb.imwrite(str(out_path), frame, [cv2_fb.IMWRITE_JPEG_QUALITY, 85])
                results.append((out_path, t_sec))
                saved += 1
            frame_index += 1
        cap.release()

    return results


# ---------------------------------------------------------------------------
# 3. Batched vision-model filtering with information-density scoring
# ---------------------------------------------------------------------------
def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


BATCH_JUDGE_SYSTEM = """You are a vision-based frame evaluator for educational video notes.

You will receive a batch of numbered video frames with their timestamps.
For EACH frame, evaluate its **Information Density** on a scale of 1–10:

Score 8–10 (HIGH — must keep):
- Complete diagram, flowchart, or architecture diagram clearly visible
- Full slide with multiple bullet points or key concepts
- Complete equation or derivation on whiteboard/screen
- Full code snippet or terminal output that teaches something
- Clear chart, table, or comparison visible

Score 5–7 (MEDIUM — keep if unique):
- Partial but still useful slide content (most text readable)
- Equation being built but mostly complete
- Code with some parts visible

Score 1–4 (LOW — discard):
- Talking head / face with no educational content behind
- Blank or nearly blank screen
- Intro/outro slides, channel logos, subscribe prompts
- Extremely blurry or unreadable content
- Frame where a hand/body is covering most of the content

DEDUPLICATION: If multiple frames show the same slide or content, give the highest score ONLY to the one where the content is MOST COMPLETE. Score the others lower (1-3) even if individually they would be medium/high.

Reply with ONLY a JSON array. Each element must be:
{"frame": <number>, "score": <int 1-10>, "content_type": "<code|diagram|slide|equation|chart|other>", "caption": "<short description>"}

Example:
[{"frame": 1, "score": 9, "content_type": "diagram", "caption": "Neural network architecture diagram"}, {"frame": 2, "score": 2, "content_type": "other", "caption": "Talking head, no content"}]"""


async def _judge_batch_async(
    llm: ChatOpenAI,
    batch: list[tuple[int, Path, float]],
) -> list[dict]:
    """Send a batch of frames to the vision model and parse scored results."""
    image_content = []
    image_content.append({
        "type": "text",
        "text": (
            f"Evaluate these {len(batch)} frames. "
            "For each, return a JSON object with frame number, score (1-10), content_type, and caption. "
            "Apply deduplication: if frames show the same content, only the most complete one gets a high score. "
            "Reply with ONLY a JSON array."
        ),
    })

    for seq_num, frame_path, ts in batch:
        if not frame_path.exists():
            continue
        b64 = _encode_image(frame_path)
        image_content.append({
            "type": "text",
            "text": f"--- Frame {seq_num} (timestamp: {ts:.0f}s) ---",
        })
        image_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    messages = [
        SystemMessage(content=BATCH_JUDGE_SYSTEM),
        HumanMessage(content=image_content),
    ]

    try:
        response = await llm.ainvoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        # Extract JSON array from response (handle markdown code fences)
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


async def _run_batched_vision_filter(
    candidate_frames: list[tuple[Path, float]],
    vision_model: str,
    batch_size: int = VISION_BATCH_SIZE,
    relevance_threshold: int = RELEVANCE_THRESHOLD,
) -> list[dict]:
    """Score all candidate frames via batched async calls to the vision model.

    Returns list of {"path", "timestamp_sec", "caption", "content_type", "relevance_score"}
    sorted by relevance_score descending.
    """
    if not candidate_frames:
        return []

    llm = ChatOpenAI(model=vision_model, temperature=0.0, max_tokens=1200)

    # Build numbered batches
    batches: list[list[tuple[int, Path, float]]] = []
    current_batch: list[tuple[int, Path, float]] = []
    for i, (path, ts) in enumerate(candidate_frames):
        current_batch.append((i + 1, path, ts))
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    # Run all batches concurrently
    tasks = [_judge_batch_async(llm, batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    scored: list[dict] = []
    frame_lookup = {i + 1: (path, ts) for i, (path, ts) in enumerate(candidate_frames)}

    for br in batch_results:
        if isinstance(br, Exception):
            continue
        for item in br:
            frame_num = item.get("frame")
            score = item.get("score", 0)
            if frame_num not in frame_lookup:
                continue
            if score < relevance_threshold:
                continue
            path, ts = frame_lookup[frame_num]
            scored.append({
                "path": str(path.resolve()),
                "timestamp_sec": ts,
                "caption": item.get("caption", f"Frame at {ts:.0f}s"),
                "content_type": item.get("content_type", "other"),
                "relevance_score": score,
            })

    # Sort by score descending, then by timestamp ascending for tie-breaking
    scored.sort(key=lambda x: (-x["relevance_score"], x["timestamp_sec"]))
    return scored


# ---------------------------------------------------------------------------
# 4. Top-N selection (text-only LLM fallback if still too many)
# ---------------------------------------------------------------------------
def _pick_top_n(frames: list[dict], max_n: int) -> list[dict]:
    """Use text-only LLM to pick the top max_n frames by caption/score/timestamp."""
    if len(frames) <= max_n:
        return frames
    from prompts.notes import FRAME_TOP_N_SYSTEM

    lines = [
        f"{i+1}. [{int(f['timestamp_sec'])}s] score={f.get('relevance_score', '?')} "
        f"type={f.get('content_type', '?')} — {f.get('caption', '')}"
        for i, f in enumerate(frames)
    ]
    prompt = (
        f"Candidates (number, timestamp, score, type, caption):\n"
        f"{chr(10).join(lines)}\n\n"
        f"Pick exactly the {max_n} most useful for study notes. "
        f"Prefer diversity of content types and high scores. "
        f"Reply with comma-separated numbers only, e.g. 1,3,5,7,8,10."
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=80)
    try:
        response = llm.invoke([SystemMessage(content=FRAME_TOP_N_SYSTEM), HumanMessage(content=prompt)])
        text = (response.content if hasattr(response, "content") else str(response)) or ""
        numbers = [int(x.strip()) for x in re.findall(r"\d+", text) if 1 <= int(x.strip()) <= len(frames)]
        seen: set[int] = set()
        indices: list[int] = []
        for n in numbers:
            if n not in seen and len(indices) < max_n:
                seen.add(n)
                indices.append(n)
        if indices:
            return [frames[i - 1] for i in indices]
    except Exception:
        pass
    return frames[:max_n]


# ---------------------------------------------------------------------------
# 5. Public API — orchestrates the full pipeline
# ---------------------------------------------------------------------------
def select_important_frames(
    candidate_frames: list[tuple[Path, float]],
    vision_model: str = VISION_MODEL,
) -> list[dict]:
    """
    Batched vision-model scoring → filter by relevance threshold.
    Keeps ALL frames that score >= RELEVANCE_THRESHOLD (dedup handled by vision judge).
    Returns list of {"path", "timestamp_sec", "caption", "content_type", "relevance_score"}.
    """
    if not candidate_frames:
        return []

    # Run async batched vision filter
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (e.g. Jupyter) — use nest_asyncio or thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            scored = pool.submit(
                asyncio.run,
                _run_batched_vision_filter(candidate_frames, vision_model),
            ).result()
    else:
        scored = asyncio.run(
            _run_batched_vision_filter(candidate_frames, vision_model)
        )

    if not scored:
        return []

    # Final sort by timestamp for chronological order in notes
    scored.sort(key=lambda x: x["timestamp_sec"])
    return scored


def _cleanup_video(video_path: Path) -> None:
    """Delete the downloaded video file to free disk space."""
    try:
        if video_path.exists():
            video_path.unlink()
    except Exception:
        pass  # Non-critical — don't crash if cleanup fails


def fetch_important_frames(
    video_id: str,
    output_dir: Path | str,
    vision_model: str = VISION_MODEL,
) -> tuple[list[dict], str]:
    """
    Full pipeline: download → SSIM scene detection with motion debouncing →
    batched vision scoring → auto-delete video.
    Returns (list of frame dicts, error_string).
    """
    output_dir = Path(output_dir)
    video_path: Path | None = None
    try:
        video_path = download_video(video_id, output_dir)
    except Exception as e:
        return [], f"Video download failed: {e}"
    try:
        candidates = extract_candidate_frames(video_path, max_frames=MAX_CANDIDATE_FRAMES)
    except Exception as e:
        if video_path:
            _cleanup_video(video_path)
        return [], f"Frame extraction failed: {e}"

    # Done reading video — delete it to free disk space
    if video_path:
        _cleanup_video(video_path)

    if not candidates:
        return [], "No candidate frames extracted"
    try:
        important = select_important_frames(
            candidates, vision_model=vision_model,
        )
    except Exception as e:
        return [], f"Frame selection failed: {e}"
    return important, ""
