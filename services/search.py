"""Web search + full-page extract for interview harvester. TAVILY_API_KEY in .env."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MAX_CHARS_PER_URL = 8000
EXTRACT_URL_CAP = 32
SEARCH_RESULTS_PER_QUERY = 10


def _tavily_search(client, query: str, max_results: int) -> list[dict]:
    kwargs = {"query": query, "max_results": max_results}
    try:
        kwargs["search_depth"] = "advanced"
        kwargs["include_raw_content"] = True
        return client.search(**kwargs).get("results") or []
    except TypeError:
        try:
            kwargs.pop("include_raw_content", None)
            return client.search(**kwargs).get("results") or []
        except Exception:
            return client.search(query=query, max_results=max_results).get("results") or []


def _tavily_extract_batch(client, urls: List[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(urls), 8):
        chunk = urls[i : i + 8]
        try:
            resp = client.extract(urls=chunk, extract_depth="advanced")
        except TypeError:
            try:
                resp = client.extract(urls=chunk)
            except Exception:
                continue
        except Exception:
            continue
        for item in resp.get("results") or []:
            u = (item.get("url") or "").strip()
            raw = (item.get("raw_content") or item.get("content") or "").strip()
            if u and raw:
                out[u] = raw[:MAX_CHARS_PER_URL]
    return out


def gather_interview_source_blob(
    queries: List[str],
    max_queries: int = 28,
    max_results_per_query: int = SEARCH_RESULTS_PER_QUERY,
) -> Tuple[List[dict], str]:
    """
    Deduped queries → Tavily search → full-page extract for up to EXTRACT_URL_CAP URLs.
    Returns (meta list, mega_blob). Strongly recommend TAVILY_API_KEY for non-empty results.
    """
    seen_q: set[str] = set()
    flat_q = []
    for q in queries:
        q = (q or "").strip()
        if q and q not in seen_q:
            seen_q.add(q)
            flat_q.append(q)
    flat_q = flat_q[:max_queries]

    results: List[Tuple[str, str, str]] = []
    seen_urls: set[str] = set()
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()

    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            for q in flat_q:
                try:
                    for item in _tavily_search(client, q, max_results_per_query):
                        url = (item.get("url") or "").strip()
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        title = item.get("title") or ""
                        raw = (item.get("raw_content") or item.get("content") or "")[:MAX_CHARS_PER_URL]
                        results.append((url, title, raw))
                except Exception:
                    continue

            urls_ordered = [u for u, _, _ in results][:EXTRACT_URL_CAP]
            extracted = _tavily_extract_batch(client, urls_ordered)

            lines: List[str] = []
            meta: List[dict] = []
            for url, title, body in results:
                full = extracted.get(url) or body
                lines.append(
                    f"=== SOURCE_URL: {url}\nTITLE: {title}\nCONTENT:\n{full}\n---END_BLOCK---\n"
                )
                meta.append({"url": url, "title": title})
            return meta, "\n".join(lines)[:200000]

        except ImportError:
            pass

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [], ""

    with DDGS() as ddgs:
        for q in flat_q:
            try:
                for item in ddgs.text(q, max_results=max_results_per_query):
                    url = (item.get("href") or item.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    body = (item.get("body") or "")[:4500]
                    results.append((url, item.get("title") or "", body))
            except Exception:
                continue
    meta = [{"url": u, "title": t} for u, t, _ in results]
    lines = [
        f"=== SOURCE_URL: {url}\nTITLE: {title}\nCONTENT:\n{body}\n---END_BLOCK---\n"
        for url, title, body in results
    ]
    return meta, "\n".join(lines)[:200000]


def build_search_queries_for_topic(topic: str, aliases: List[str], extra: List[str] | None = None) -> List[str]:
    t = topic.strip()
    queries = [
        f"{t} machine learning interview question Google Meta Amazon Microsoft",
        f'"{t}" data scientist interview asked at company',
        f"{t} MLE interview experience reddit",
        f"glassdoor {t} machine learning interview",
        f"{t} AI engineer FAANG interview question",
        f"{t} onsite machine learning interview",
        f"leetcode discuss {t} interview",
    ]
    if aliases:
        for a in aliases[:2]:
            if a and a.lower() != t.lower():
                queries.append(f'"{a}" data scientist interview question')
    if extra:
        queries.extend(x for x in extra[:8] if x and x.strip())
    out, s = [], set()
    for q in queries:
        q = q.strip()
        if q and q not in s:
            s.add(q)
            out.append(q)
    return out[:10]


def build_global_queries_from_topics(topics: List[dict]) -> List[str]:
    """Build broad cross-topic search queries dynamically from actual topic names."""
    names = [t.get("topic") or "" for t in topics[:6] if t.get("topic")]
    if not names:
        return []

    # Build topic-aware queries instead of hardcoded ones
    blob_short = " ".join(names[:3])[:80].strip()
    blob_long = " ".join(names[:5])[:120].strip()
    queries = [
        f"{blob_short} machine learning interview questions Google Meta Amazon",
        f"{blob_long} data scientist interview questions FAANG",
        f"{names[0]} ML engineer interview questions companies",
    ]
    if len(names) > 1:
        queries.append(f"{names[1]} AI engineer interview question experience")
    if len(names) > 2:
        queries.append(f"{names[2]} deep learning interview asked at company")
    return queries
