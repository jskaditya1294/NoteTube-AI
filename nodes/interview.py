"""Nodes 1–3: Topic Miner, Question Harvester, Critic + format export."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.state import NotesWorkflowState, MAX_INTERVIEW_LOOPS
from prompts.interview import (
    NODE1_TOPIC_MINER_SYSTEM,
    NODE2_QUESTION_HARVESTER_SYSTEM,
    NODE2_QUESTION_HARVESTER_RETRY,
    NODE3_CRITIC_SYSTEM,
)
from services.search import (
    gather_interview_source_blob,
    build_search_queries_for_topic,
    build_global_queries_from_topics,
)

_llm = ChatOpenAI(model="gpt-5.1", temperature=0.1)
_llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def interview_topic_miner(state: NotesWorkflowState) -> dict[str, Any]:
    transcript = (state.get("transcript") or "")[:120000]
    notes = (state.get("notes") or "")[:80000]
    feedback = state.get("feedback_node1") or ""
    user = f"""Transcript (excerpt if long):
---
{transcript[:60000]}
---

Notes:
---
{notes}
---

Prior critic feedback for topics (follow if non-empty):
{feedback}

Output STRICT JSON only per schema in system message."""
    r = _llm.invoke([SystemMessage(content=NODE1_TOPIC_MINER_SYSTEM), HumanMessage(content=user)])
    text = r.content if hasattr(r, "content") else str(r)
    data = _extract_json_object(text)
    topics = data.get("topics") or []
    if not topics:
        data = {
            "topics": [],
            "topic_coverage_notes": data.get("topic_coverage_notes") or "No topics extracted.",
        }
    out: dict[str, Any] = {"topics_data": data}
    if (state.get("feedback_node1") or "").strip():
        out["question_bank"] = []
        out["excluded_unattributed"] = []
        out["search_coverage"] = []
    return out


def interview_question_harvester(state: NotesWorkflowState) -> dict[str, Any]:
    topics_data = state.get("topics_data") or {}
    topics = topics_data.get("topics") or []
    extra_queries = list(state.get("node2_extra_queries") or [])

    def pri(p):
        return {"High": 0, "Medium": 1, "Low": 2}.get(p, 1)

    topics_sorted = sorted(topics, key=lambda x: pri(x.get("priority", "Medium")))[:12]

    all_queries: list[str] = []
    all_queries.extend(build_global_queries_from_topics(topics_sorted))
    for t in topics_sorted[:8]:
        name = t.get("topic") or ""
        if not name:
            continue
        all_queries.extend(
            build_search_queries_for_topic(name, t.get("aliases") or [], extra_queries)[:5]
        )
    all_queries.extend(q for q in extra_queries if q and q.strip())

    seen_q: set[str] = set()
    deduped: list[str] = []
    for q in all_queries:
        q = q.strip()
        if q and q not in seen_q:
            seen_q.add(q)
            deduped.append(q)

    meta, blob = gather_interview_source_blob(deduped, max_queries=30)
    doms = []
    for m in meta:
        u = m.get("url") or ""
        if "://" in u:
            try:
                doms.append(u.split("/")[2])
            except IndexError:
                pass
    coverage_meta = [{
        "topic": "aggregated_search",
        "sources_used": list(dict.fromkeys(doms))[:30],
        "notes": f"{len(meta)} pages fetched, {len(blob)} chars (use TAVILY_API_KEY + extract for best results)",
    }]

    if not blob.strip():
        blob = (
            "(No web results. Add TAVILY_API_KEY to .env — DuckDuckGo alone often misses company-attributed Q&A.)"
        )

    topics_json = json.dumps({"topics": topics_sorted}, ensure_ascii=False)[:20000]
    user = f"""Topics JSON:
{topics_json}

WEB SOURCES (each block: SOURCE_URL, TITLE, CONTENT — extract from CONTENT/TITLE only):
{blob[:195000]}

Output STRICT JSON."""

    r = _llm.invoke([SystemMessage(content=NODE2_QUESTION_HARVESTER_SYSTEM), HumanMessage(content=user)])
    text = r.content if hasattr(r, "content") else str(r)
    data = _extract_json_object(text)
    qb = data.get("question_bank") or []

    if len(qb) < 4 and len(blob) > 4000:
        r2 = _llm.invoke([
            SystemMessage(
                content=NODE2_QUESTION_HARVESTER_SYSTEM + "\n\n" + NODE2_QUESTION_HARVESTER_RETRY
            ),
            HumanMessage(content=user),
        ])
        data2 = _extract_json_object(r2.content if hasattr(r2, "content") else str(r2))
        qb2 = data2.get("question_bank") or []
        if len(qb2) > len(qb):
            qb = qb2
            data = data2
        elif not qb and qb2:
            qb = qb2
            data = data2

    if len(qb) < 3 and len(blob) > 3000:
        emergency = [
            "FAANG machine learning interview questions 2024",
            "Google Meta Amazon data scientist interview questions list",
            "MLE onsite technical questions company experience",
        ] + deduped[:8]
        _, blob_em = gather_interview_source_blob(emergency, max_queries=14)
        if len(blob_em) > 2500:
            r3 = _llm.invoke([
                SystemMessage(content=NODE2_QUESTION_HARVESTER_SYSTEM),
                HumanMessage(
                    content=f"Topics JSON:\n{topics_json}\n\nADDITIONAL WEB SOURCES:\n{blob_em[:120000]}\n\nOutput STRICT JSON."
                ),
            ])
            qb3 = _extract_json_object(r3.content if hasattr(r3, "content") else str(r3)).get(
                "question_bank"
            ) or []
            for q in qb3:
                k = (
                    str(q.get("question_text", "")).strip().lower(),
                    str(q.get("company", "")).strip().lower(),
                )
                if k[0] and not any(
                    str(x.get("question_text", "")).strip().lower() == k[0]
                    and str(x.get("company", "")).strip().lower() == k[1]
                    for x in qb
                ):
                    qb.append(q)
    excl = data.get("excluded_unattributed") or []
    sc = data.get("search_coverage") or coverage_meta

    old_qb = state.get("question_bank") or []
    if old_qb:
        seen = {
            (str(q.get("question_text", "")).strip().lower(), str(q.get("company", "")).strip().lower())
            for q in old_qb
        }
        for q in qb:
            k = (str(q.get("question_text", "")).strip().lower(), str(q.get("company", "")).strip().lower())
            if k not in seen and k[0]:
                seen.add(k)
                old_qb.append(q)
        qb = old_qb
    old_ex = state.get("excluded_unattributed") or []
    excl = (old_ex + excl)[-80:]

    return {
        "question_bank": qb,
        "excluded_unattributed": excl,
        "search_coverage": sc,
        "raw_search_blob": blob[:5000],
    }


def interview_critic(state: NotesWorkflowState) -> dict[str, Any]:
    topics_data = state.get("topics_data") or {}
    qb = state.get("question_bank") or []
    loop = state.get("interview_loop") or 0

    summary = {
        "num_topics": len(topics_data.get("topics") or []),
        "num_questions": len(qb),
        "companies": list({q.get("company") for q in qb if q.get("company")})[:30],
        "sample_questions": qb[:5],
    }
    user = f"""Loop iteration: {loop} / {MAX_INTERVIEW_LOOPS}

Topics JSON (abbreviated):
{json.dumps(topics_data, ensure_ascii=False)[:15000]}

Question bank count: {len(qb)}
Question bank sample (full list truncated if huge):
{json.dumps(qb[:40], ensure_ascii=False)[:25000]}

Decide: APPROVE_AND_EXPORT, REVISE_TOPICS, or RESEARCH_MORE.
If iteration >= {MAX_INTERVIEW_LOOPS - 1}, prefer APPROVE unless data is empty.

Output STRICT JSON only."""
    r = _llm_fast.invoke([SystemMessage(content=NODE3_CRITIC_SYSTEM), HumanMessage(content=user)])
    text = r.content if hasattr(r, "content") else str(r)
    data = _extract_json_object(text)
    decision = (data.get("decision") or "APPROVE_AND_EXPORT").upper()
    if "REVISE" in decision:
        decision = "REVISE_TOPICS"
    elif "RESEARCH" in decision:
        decision = "RESEARCH_MORE"
    else:
        decision = "APPROVE_AND_EXPORT"

    li = data.get("loop_instructions") or {}
    feedback_n1 = ""
    extra_q = state.get("node2_extra_queries") or []
    if decision == "REVISE_TOPICS":
        for iss in data.get("issues") or []:
            feedback_n1 += (iss.get("fix_instructions") or "") + "\n"
        if li.get("priority_topics"):
            feedback_n1 += "Focus topics: " + ", ".join(str(x) for x in li["priority_topics"])
    new_extra = list(extra_q)
    if decision == "RESEARCH_MORE":
        new_extra.extend(li.get("search_queries_to_try") or [])
        for iss in data.get("issues") or []:
            new_extra.append(iss.get("fix_instructions") or "")

    new_loop = loop + 1 if decision in ("REVISE_TOPICS", "RESEARCH_MORE") else loop
    cycles = (state.get("interview_cycles") or 0) + (
        1 if decision in ("REVISE_TOPICS", "RESEARCH_MORE") else 0
    )

    if decision == "REVISE_TOPICS":
        n2_queries: list[str] = []
    else:
        n2_queries = list(state.get("node2_extra_queries") or [])
    if decision == "RESEARCH_MORE":
        n2_queries.extend(li.get("search_queries_to_try") or [])
        for iss in data.get("issues") or []:
            fi = iss.get("fix_instructions") or ""
            if fi and len(fi) < 200:
                n2_queries.append(fi)
    n2_queries = list(dict.fromkeys(q for q in n2_queries if q))[:25]

    fn1 = (feedback_n1.strip() or state.get("feedback_node1") or "") if decision == "REVISE_TOPICS" else (
        state.get("feedback_node1") or ""
    )
    if decision == "REVISE_TOPICS" and feedback_n1.strip():
        fn1 = feedback_n1.strip()

    return {
        "critic_decision": decision,
        "critic_full": data,
        "interview_loop": new_loop,
        "interview_cycles": cycles,
        "feedback_node1": fn1 if decision == "REVISE_TOPICS" else state.get("feedback_node1", ""),
        "node2_extra_queries": n2_queries,
    }


def interview_format_export(state: NotesWorkflowState) -> dict[str, Any]:
    """Build final markdown: summary table, coverage, Topic → Company → Questions."""
    topics_data = state.get("topics_data") or {}
    qb = state.get("question_bank") or []
    sc = state.get("search_coverage") or []
    excl = state.get("excluded_unattributed") or []

    by_topic: dict[str, dict[str, list]] = {}
    for q in qb:
        t = q.get("topic") or "General"
        co = q.get("company") or "Unknown"
        by_topic.setdefault(t, {}).setdefault(co, []).append(q)

    lines = [
        "## Interview question bank",
        "",
        "*Target roles: ML Engineer / Data Scientist / AI Engineer. Questions are sourced from web pages only (not invented).*",
        "",
    ]
    if not qb:
        lines.extend([
            "> **No attributed questions were extracted.** For reliable results, set **`TAVILY_API_KEY`** in `.env` "
            "(Tavily search + full-page extract). DuckDuckGo-only mode often returns snippets too short to tie "
            "questions to a company.",
            "",
        ])
    lines.extend([
        "### Summary table",
        "",
        "| Topic | # Companies | # Questions | Top sources |",
        "|-------|-------------|-------------|-------------|",
    ])
    topic_list = topics_data.get("topics") or []
    for t in topic_list:
        name = t.get("topic") or ""
        companies = by_topic.get(name, {})
        n_co = len(companies)
        n_q = sum(len(v) for v in companies.values())
        src = ""
        for row in sc:
            if row.get("topic") == name:
                src = ", ".join((row.get("sources_used") or [])[:5])
                break
        lines.append(f"| {name} | {n_co} | {n_q} | {src[:80]} |")
    if qb:
        n_co_all = len({(q.get("company") or "").strip() for q in qb if (q.get("company") or "").strip()})
        lines.append(
            f"| **Total (all in bank)** | {n_co_all} | {len(qb)} | See sections below |"
        )

    lines.extend(["", "### Coverage report", "", f"- Extracted topics: {len(topic_list)}", f"- Questions with attribution: {len(qb)}", f"- Excluded (unattributed): {len(excl)}", ""])

    if topics_data.get("topic_coverage_notes"):
        lines.extend(["**Topic coverage notes:**", topics_data["topic_coverage_notes"], ""])

    lines.extend(["### Questions by topic → company", ""])
    for tname, companies in sorted(by_topic.items()):
        lines.append(f"#### {tname}")
        lines.append("")
        for co, items in sorted(companies.items()):
            lines.append(f"**{co}**")
            for q in items:
                qt = q.get("question_text") or ""
                rl = q.get("role_level") or ""
                url = q.get("source_url") or ""
                ev = q.get("evidence_snippet") or ""
                lines.append(f"- **Q:** {qt}")
                if rl:
                    lines.append(f"  - *Role/level:* {rl}")
                lines.append(f"  - *Source:* {url}")
                lines.append(f"  - *Evidence:* {ev[:500]}{'…' if len(ev) > 500 else ''}")
                lines.append("")
        lines.append("")

    if excl:
        lines.extend(["### Unattributed (excluded from main list)", ""])
        for e in excl[:30]:
            lines.append(f"- **Topic:** {e.get('topic')} — {e.get('reason')} — _{str(e.get('question_text', ''))[:200]}_")
        lines.append("")

    return {"interview_bank_markdown": "\n".join(lines)}
