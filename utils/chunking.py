import re
from typing import List

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Marker format injected into transcript before chunking.
# normalize_transcript preserves these; the chunker uses them as preferred split points.
CHAPTER_MARKER_RE = re.compile(r"^\[SECTION:\s*(.+?)\]\s*$", re.MULTILINE)


def inject_chapter_markers(
    transcript_with_timestamps: str,
    chapters: list[dict],
) -> str:
    """Insert ``[SECTION: Title]`` markers into a timestamped transcript.

    *chapters* is a list of ``{"title": str, "start_time": float}`` (seconds).
    The transcript must have ``[MM:SS] text`` lines (as produced by ``get_transcript``).
    Markers are inserted on their own line just before the closest transcript line.
    """
    if not chapters or not transcript_with_timestamps:
        return transcript_with_timestamps

    lines = transcript_with_timestamps.split("\n")
    ts_pattern = re.compile(r"^\[(\d{2}):(\d{2})\]")

    # Build (line_index, seconds) for every timestamped line
    line_times: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        m = ts_pattern.match(line)
        if m:
            line_times.append((i, int(m.group(1)) * 60 + int(m.group(2))))

    if not line_times:
        return transcript_with_timestamps

    # Map each chapter → best matching line index
    insertions: dict[int, str] = {}
    for ch in chapters:
        target = ch["start_time"]
        best_idx, best_diff = line_times[0][0], abs(line_times[0][1] - target)
        for li, ts in line_times:
            diff = abs(ts - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = li
        insertions[best_idx] = ch["title"]

    # Rebuild with markers
    result: list[str] = []
    for i, line in enumerate(lines):
        if i in insertions:
            result.append(f"\n[SECTION: {insertions[i]}]\n")
        result.append(line)

    return "\n".join(result)


def normalize_transcript(text: str) -> str:
    """Light cleanup without changing meaning. Strips timestamp prefixes if present.

    Preserves ``[SECTION: ...]`` chapter markers injected by inject_chapter_markers.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip [MM:SS] timestamp prefixes (added by get_transcript)
    # but NOT [SECTION: ...] markers
    text = re.sub(r"^\[(\d{2}):(\d{2})\]\s*", "", text, flags=re.MULTILINE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def estimate_tokens(text: str, model_name: str = "gpt-5.1") -> int:
    """
    Count tokens if tiktoken is available, otherwise fallback to rough estimate.
    Rough fallback: ~4 chars/token for English-like text.
    """
    if not text:
        return 0

    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    return max(1, len(text) // 4)


def split_into_sentences(text: str) -> List[str]:
    """
    Multilingual sentence splitter for Hindi and English transcripts.

    Handles:
    - English punctuation (.!?)
    - Hindi purna viram (।) and double danda (॥)
    - Single-newline caption breaks (common in YouTube transcripts)
    - Unpunctuated auto-captions via word-count fallback
    """
    if not text.strip():
        return []

    # Split by paragraph boundaries first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: List[str] = []

    # Sentence terminators: . ! ? । ॥
    # Lookahead: uppercase Latin, Devanagari (U+0900-U+097F), digits, quotes
    sentence_end = re.compile(
        r'(?<=[.!?।॥])\s+(?=[A-Z0-9\u0900-\u097F"\'])'
    )

    for para in paragraphs:
        # Single newlines often mark caption breaks in transcripts — honour them
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]

        for line in lines:
            parts = sentence_end.split(line)
            parts = [p.strip() for p in parts if p.strip()]

            for part in (parts if parts else [line]):
                # Long unpunctuated fragment → break further
                if len(part.split()) > 40:
                    sentences.extend(_split_long_segment(part))
                else:
                    sentences.append(part)

    return sentences


def _split_long_segment(segment: str, target_words: int = 25) -> List[str]:
    """
    Break a long unpunctuated segment into ~target_words pieces.
    Tries comma / semicolon pauses first, then falls back to word-count splits.
    """
    # Try splitting at commas / semicolons (natural pauses in speech)
    fragments = re.split(r'(?<=[,;])\s+', segment)

    if len(fragments) > 1:
        merged: List[str] = []
        current: List[str] = []
        current_wc = 0
        for frag in fragments:
            frag = frag.strip()
            if not frag:
                continue
            wc = len(frag.split())
            if current_wc + wc > target_words and current:
                merged.append(" ".join(current))
                current = [frag]
                current_wc = wc
            else:
                current.append(frag)
                current_wc += wc
        if current:
            merged.append(" ".join(current))
        return merged if merged else [segment]

    # Last resort: split by word count
    words = segment.split()
    pieces: List[str] = []
    for i in range(0, len(words), target_words):
        piece = " ".join(words[i : i + target_words]).strip()
        if piece:
            pieces.append(piece)
    return pieces if pieces else [segment]



def build_overlapping_chunks(
    text: str,
    target_tokens: int = 1400,
    max_tokens: int = 1800,
    overlap_tokens: int = 220,
    model_name: str = "gpt-4o",
    chapter_flush_min_tokens: int = 400,
) -> List[str]:
    """
    Sentence-aware, chapter-aware token-budgeted chunker.

    Strategy:
    - Split transcript into sentences
    - Pack sentences until near target/max token budget
    - Carry ~overlap_tokens worth of trailing sentences into next chunk
    - When a ``[SECTION: ...]`` chapter marker is encountered and the current
      chunk already has ``chapter_flush_min_tokens``, flush before starting
      the new section — this keeps topics together.
    """
    text = normalize_transcript(text)
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    def sentence_tokens(s: str) -> int:
        return estimate_tokens(s, model_name=model_name)

    def _is_chapter_marker(s: str) -> bool:
        return bool(CHAPTER_MARKER_RE.match(s.strip()))

    def flush_with_overlap():
        nonlocal current_sentences, current_tokens

        if not current_sentences:
            return

        # Save current chunk
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Build overlap from tail
        overlap_sentences: List[str] = []
        overlap_count = 0
        for tail_sentence in reversed(current_sentences):
            t = sentence_tokens(tail_sentence)
            if overlap_count + t > overlap_tokens and overlap_sentences:
                break
            overlap_sentences.insert(0, tail_sentence)
            overlap_count += t

        current_sentences = overlap_sentences[:]
        current_tokens = overlap_count

    i = 0
    while i < len(sentences):
        s = sentences[i]
        s_tokens = sentence_tokens(s)

        # Chapter marker → prefer starting a new chunk here if current chunk
        # already has meaningful content.  This keeps each section together.
        if _is_chapter_marker(s) and current_tokens >= chapter_flush_min_tokens:
            flush_with_overlap()
            # Don't skip the marker — include it in the new chunk so the LLM
            # sees the section title.
            current_sentences.append(s)
            current_tokens += s_tokens
            i += 1
            continue

        # If one sentence itself exceeds max_tokens, force it into its own chunk
        if s_tokens > max_tokens:
            flush_with_overlap()
            chunks.append(s.strip())
            i += 1
            continue

        # If adding this sentence would exceed max_tokens, flush current chunk first
        if current_tokens + s_tokens > max_tokens:
            flush_with_overlap()
            continue

        # Otherwise, add sentence
        current_sentences.append(s)
        current_tokens += s_tokens
        i += 1

        # If we have reached target size, flush near target
        if current_tokens >= target_tokens:
            flush_with_overlap()

    # Final leftover chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks