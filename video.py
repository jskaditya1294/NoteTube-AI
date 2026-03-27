# # """YouTube video ID extraction and transcript fetching."""


import re
from urllib.parse import parse_qs, urlparse
from typing import Sequence

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)

VIDEO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")


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

    return " ".join(snippet.text for snippet in fetched_transcript)