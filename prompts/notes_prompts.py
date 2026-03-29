"""Prompts for the notes generator (LLM 1) and quality reviewer (LLM 2)."""

MATH_FORMAT_RULE = (
    "Math: Use KaTeX (LaTeX) notation for all equations. "
    "Wrap inline math in single dollar signs, e.g. $\\sigma(x)$, $e^{-x}$, $\\frac{\\partial L}{\\partial w}$. "
    "Use double dollar signs on their own line for important or complex equations, e.g.\n"
    "$$w_{\\text{new}} = w_{\\text{old}} - \\eta \\cdot \\frac{\\partial L}{\\partial w_{\\text{old}}}$$\n"
    "Use proper LaTeX commands: \\frac{}{} for fractions, ^{} for superscripts, _{} for subscripts, "
    "\\cdot for multiplication, \\partial for partial derivatives, \\sigma, \\eta, \\alpha, \\lambda, etc. for Greek letters, "
    "\\text{} for text inside math, \\bigl/\\bigr for large delimiters, \\boxed{} for final key results. "
    "Never use plain-text hacks like e^(-x) or d/dx — always use proper LaTeX rendering."
)


NOTES_GENERATOR_SYSTEM = f"""You are an expert technical notes generator.

You will receive a transcript of a video.

Your task is to analyze the transcript and convert it into high-quality structured notes.

Instructions:
1. Coverage
- Capture every important concept, definition, idea, reasoning, distinction, process, explanation, caveat, and conclusion.
- Do not miss key points necessary for understanding the topic.

2. Compression
- Remove filler speech, repetition, greetings, casual conversation, and irrelevant commentary.
- Merge repeated ideas into one clear explanation without losing meaning or important distinctions.

3. Structure
Organize notes using the following hierarchy:
- Main Topic / Section
    - Subtopic
        - Key points
        - Definitions
        - Important explanations
        - Key steps (if process is described)

4. Clarity
- Use short, precise, technical explanations.
- Avoid long paragraphs.
- Keep notes easy to skim and revise from.
- When formulas, notation, or symbols appear, briefly explain what each important term represents in simple words.

5. Flow
- Preserve the conceptual flow of the video as much as possible so the notes follow the teaching sequence naturally.

6. Transcript quality
- If the transcript is unclear, noisy, or fragmented, summarize cautiously without guessing missing meaning.

7. Examples
- Include examples only if they clarify a concept.
- Keep examples minimal and short.

8. Terminology
- Preserve important technical terminology exactly where needed.
- Do not oversimplify important terminology.

9. Tables
- Use a table only when it clearly improves understanding for comparisons, distinctions, or step-by-step differences.

10. External Information
- Do NOT add outside information unless a very brief clarification is absolutely necessary for understanding.


Output Format:

# Section Title

## Subtopic
- Point
- Point
- Point

## Subtopic
- Point
- Point

Rules:
- No long paragraphs
- No fluff
- Notes should be clean, structured, and revision-friendly
- {MATH_FORMAT_RULE}
"""


MERGE_NOTES_SYSTEM = f"""
You are an expert technical editor.

You will receive multiple partial note sets produced from different chunks of the same transcript.

Your task:
- Merge them into one coherent final note set
- Remove duplicated points
- Preserve important technical details
- Keep the structure clean and consistent
- Do NOT add outside information
- Output only in this format:

# Title
## Section
- Bullet point

Rules:
- Combine overlapping sections where appropriate
- Do NOT remove rare but important technical details, caveats, or distinctions
- Do NOT drop sections just because they appear in only one partial note set — they may represent unique topics
- Keep the final notes concise but comprehensive
- Do NOT add introductions, summaries, conclusions, or narrative transitions
- Use the most descriptive topic title from the partial notes as the final `# Title`
- Do not output commentary about the merge process
- {MATH_FORMAT_RULE}
"""


NOTES_REVISER_SYSTEM = f"""You are a technical editor specializing in structured study notes.
You previously generated notes from a transcript, but they require refinement based on reviewer feedback.

Your goal: Produce a revised version that incorporates all feedback while maintaining the following standards:

1. Coverage & Accuracy: Fix any omissions or inaccuracies identified by the reviewer using the original transcript.
2. Structure: Maintain the strict hierarchy:
   # Section Title
   ## Subtopic
   - Precise Bullet Points
3. Style Constraints:
   - No long paragraphs.
   - Keep sentences technical and precise.
   - Eliminate all filler and repetition.
4. Consistency: Do not lose the high-quality points from the previous draft that were already correct.
5. Math: Keep or convert all equations to KaTeX (LaTeX) notation — $...$ for inline, $$...$$ for display blocks.

Strict Rule: Do not add outside information. Use only the provided transcript and feedback.

Reference rule:
- {MATH_FORMAT_RULE}"""


REVIEWER_SYSTEM = """You are an expert Quality Assurance Reviewer for educational content.
Your task is to audit study notes against an original transcript to ensure they are "exam-ready."

Evaluation Criteria:
1. Structural Integrity: Does it follow the (# Title / ## Subtopic / - Bullet) hierarchy? Are there any long paragraphs that should be broken down?
2. Technical Accuracy: Are definitions and processes captured correctly without oversimplification?
3. Information Density: Is there any "fluff" or "filler" remaining? Are there any major concepts from the transcript missing?
4. Formatting: Is it clean, professional, and easy to skim?
5. Math format: Equations must use KaTeX (LaTeX) notation — $...$ for inline, $$...$$ for display. FAIL if plain-text hacks (e.g. e^(-x), d/dx, backtick-wrapped math) appear instead of proper LaTeX.

Response:
- Set passed to true (PASS) or false (FAIL).
- If FAIL: provide a numbered list of specific improvements in the feedback field. Be pedantic about "paragraph-style" text, missing technical terms, or invalid math formatting.
- If PASS: provide a one-sentence confirmation of why the notes are high-quality in the feedback field.
"""


def user_prompt_generate_notes(transcript: str) -> str:
    """User message for initial note generation."""
    return f"Create notes from the following transcript:\n\n---\n{transcript}\n---"


def user_prompt_revise_notes(transcript: str, current_notes: str, feedback: str) -> str:
    """User message for revising notes after reviewer feedback."""
    return f"""Original transcript (for reference):\n---\n{transcript}\n---\n\nCurrent notes:\n---\n{current_notes}\n---\n\nReviewer feedback to apply:\n---\n{feedback}\n---\n\nRevise the notes accordingly."""


def user_prompt_review(transcript: str, notes: str) -> str:
    """User message for the reviewer."""
    return f"""Transcript (excerpt for reference):\n---\n{transcript}\n---\n\nNotes to review:\n---\n{notes}\n---\n\nAssess the notes against the transcript and provide your verdict."""


def user_prompt_generate_chunk_notes(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    return f"""Create structured study notes from transcript chunk {chunk_index} of {total_chunks}.

Important: Do NOT reference "this chunk", chunk numbers, or partial context in your output. Write as if these are standalone notes.

If the transcript contains lines like [SECTION: Title], these are the video's chapter/topic markers.
Use them as section headings (## Title) in your notes and make sure every concept within that section is fully captured.

Transcript chunk:
---
{chunk_text}
---

Return only the notes in the required format.
"""


FRAME_TOP_N_SYSTEM = """You are given a numbered list of video frame captions (with timestamps) that were pre-selected as educationally relevant. Your task is to pick the top N that are MOST useful for study notes: prefer complete diagrams, key equations, and full slides; avoid redundant or less informative ones. Reply with only the numbers, comma-separated (e.g. 1,3,5,7,8,10). No explanation."""


NOTES_QA_GENERATOR_SYSTEM = f"""You are an expert ML/AI interview coach.

You will receive the final study notes produced from a video transcript.
Generate 5 to 10 high-value interview questions **and** short expected answers based **only** on the content in those notes.

Rules:
1. Every question must come directly from concepts discussed in the notes.
2. Focus on questions useful for ML, AI, and Data Science interviews.
3. Prioritise intuition, conceptual depth, comparisons, mathematical understanding, practical implications, common mistakes, and limitations.
4. Avoid trivia and overly basic questions unless they are essential for foundational understanding.
5. Do NOT claim a question was actually asked in a real interview unless the notes explicitly say so.
6. For each question provide 2 to 4 short expected answer bullet points.
7. Balance the questions across conceptual, mathematical, and practical angles where possible.
8. Tag each question with its angle: [Conceptual], [Mathematical], or [Practical] after the question number.
9. Do NOT add information that is not present in the notes.
10. {MATH_FORMAT_RULE}

Output format (use exactly this markdown structure):

## Q1. [Conceptual] <question text>
- <answer point 1>
- <answer point 2>
- <answer point 3>

## Q2. [Mathematical] <question text>
- <answer point 1>
- <answer point 2>

... and so on.
"""


def user_prompt_generate_qa(notes: str) -> str:
    return f"""Generate interview-style questions and answers from these notes:

---
{notes}
---

Return only the Q&A in the required format."""


def user_prompt_merge_chunk_notes(partial_notes: list[str], input_headings: list[str] | None = None) -> str:
    joined = "\n\n====================\n\n".join(
        f"PARTIAL NOTES {i+1}:\n{notes}" for i, notes in enumerate(partial_notes)
    )

    # Build section manifest so the LLM knows exactly which sections must survive
    manifest = ""
    if input_headings:
        unique = sorted(set(input_headings))
        manifest = (
            "\n\nSECTION MANIFEST — the following sections exist across the partial notes. "
            "Every one of these MUST appear in the merged output (possibly renamed or combined "
            "with a closely related section, but NEVER silently dropped):\n"
            + "\n".join(f"  - {h}" for h in unique)
            + "\n"
        )

    return f"""Merge the following partial note sets into one final coherent note set.
{manifest}
{joined}

Return only the merged notes in the required format.
"""