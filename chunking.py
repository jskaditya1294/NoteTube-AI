import re
from typing import List

try:
    import tiktoken
except ImportError:
    tiktoken = None


def normalize_transcript(text: str) -> str:
    """Light cleanup without changing meaning."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def estimate_tokens(text: str, model_name: str = "gpt-4o") -> int:
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
    Basic sentence splitter.
    Good enough for transcripts; avoids extra dependency.
    """
    if not text.strip():
        return []

    # First split by paragraph boundaries
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: List[str] = []

    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')

    for para in paragraphs:
        parts = sentence_pattern.split(para)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            sentences.extend(parts)
        else:
            sentences.append(para)

    return sentences



def build_overlapping_chunks(
    text: str,
    target_tokens: int = 1400,
    max_tokens: int = 1800,
    overlap_tokens: int = 220,
    model_name: str = "gpt-4o",
) -> List[str]:
    """
    Sentence-aware token-budgeted chunker.

    Strategy:
    - Split transcript into sentences
    - Pack sentences until near target/max token budget
    - Carry ~overlap_tokens worth of trailing sentences into next chunk
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