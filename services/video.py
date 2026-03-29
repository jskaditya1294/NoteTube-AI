# # """YouTube video ID extraction and transcript fetching."""


import logging
import re
from urllib.parse import parse_qs, urlparse
from typing import Sequence

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)

VIDEO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")
logger = logging.getLogger(__name__)


def extract_video_id(input_str: str) -> str:
    """
    Extract YouTube video ID from URL or return input if it's already an ID.

    Supports:
    - youtube.com/watch?v=VIDEO_ID
    - youtu.be/VIDEO_ID
    - plain 11-char VIDEO_ID
    - youtube.com/shorts/VIDEO_ID
    - youtube.com/embed/VIDEO_ID
    """
    if not input_str or not input_str.strip():
        raise ValueError("Video ID or URL cannot be empty")

    s = input_str.strip()

    # Plain 11-char ID
    if VIDEO_ID_PATTERN.fullmatch(s):
        return s

    # Short URL: youtu.be/VIDEO_ID
    m = re.match(
        r"^(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})(?:[?/].*)?$",
        s,
    )
    if m:
        return m.group(1)

    parsed = urlparse(s)
    valid_hosts = {"youtube.com", "www.youtube.com", "m.youtube.com"}

    # Long URL: youtube.com/watch?v=VIDEO_ID
    if parsed.netloc in valid_hosts:
        if parsed.path == "/watch":
            q = parse_qs(parsed.query)
            v = q.get("v")
            if v:
                candidate = v[0]
                if VIDEO_ID_PATTERN.fullmatch(candidate):
                    return candidate

        # Support /shorts/VIDEO_ID and /embed/VIDEO_ID
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) == 2 and path_parts[0] in {"shorts", "embed"}:
            candidate = path_parts[1]
            if VIDEO_ID_PATTERN.fullmatch(candidate):
                return candidate

    raise ValueError(f"Could not extract video ID from: {s}")




"""
You give it a YouTube video ID (like "uCrevbBh0zM").
It asks the YouTube Transcript API library to fetch the transcript (captions/subtitles).
It tries multiple languages (either the ones you pass, or a default fallback list).
It then extracts only the text from each transcript snippet and joins everything into one long plain string (good for summarization/notes).

api.fetch() returns a FetchedTranscript object, which behaves like a list of snippet objects. Each snippet has .text and timing fields (start/duration). 
Example shape (from docs):

FetchedTranscript(snippets=[FetchedTranscriptSnippet(text="Hey there", start=0.0, duration=...), ...])
"""
def get_transcript(video_id: str, languages: Sequence[str] | None = None) -> str:
    api = YouTubeTranscriptApi()

    try:
        fetched_transcript = api.fetch(
            video_id,
            languages=list(languages) if languages else [
                "en", "hi", "es", "fr", "de", "ja", "ko", "pt", "zh-Hans"
            ]
        )
    except (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
        RequestBlocked,
        IpBlocked,
    ):
        raise

    # #7: Preserve timestamps — enables future "link to video time" features.
    # Format: [MM:SS] text (lightweight, stripped by chunker's normalize_transcript)
    parts: list[str] = []
    for snippet in fetched_transcript:
        ts = int(getattr(snippet, "start", 0))
        mm, ss = divmod(ts, 60)
        parts.append(f"[{mm:02d}:{ss:02d}] {snippet.text}")
    return "\n".join(parts)


def get_chapters(video_id: str) -> list[dict]:
    """Fetch YouTube chapter markers via yt-dlp metadata (no download).

    Returns list of ``{"title": str, "start_time": float}`` sorted by start_time.
    Returns empty list if the video has no chapters or on any error.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Only need metadata — no formats, thumbnails, etc.
        "extract_flat": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            raw_chapters = info.get("chapters") or []
            chapters = [
                {"title": ch["title"].strip(), "start_time": float(ch["start_time"])}
                for ch in raw_chapters
                if ch.get("title")
            ]
            chapters.sort(key=lambda c: c["start_time"])
            logger.info("Found %d YouTube chapters for %s", len(chapters), video_id)
            return chapters
    except Exception as e:
        logger.warning("Could not fetch chapters for %s: %s", video_id, e)
        return []