# YouTube Notes Workflow

A **single LangGraph pipeline** that turns a YouTube video into **study notes**, **key images** from the video, and an **interview question bank** (for ML Engineer / Data Scientist / AI Engineer roles). One run: transcript → notes (with quality review) → important frames → interview topics → web-sourced questions with company attribution → export to Markdown and Word.

---

## Project flow (high level)

1. **Notes pipeline**: Fetch transcript → extract key frames from video → generate structured notes from transcript (chunk → summarize → merge) → review quality → revise until pass or max iterations.
2. **Interview pipeline** (uses the same transcript + notes): Extract topics from transcript/notes → search the web for real interview questions per topic → critic checks quality and can loop back → format and export question bank.

Everything runs in **one graph**; the interview phase starts only after notes are finalized and reuses that data (no duplicate work).

---

## Workflow graph: every node

The graph runs in this order. Each node reads/writes the shared **state** (e.g. `transcript`, `notes`, `question_bank`).

### Phase 1: Notes pipeline

| Node | What it does |
|------|----------------|
| **fetch_transcript** | Gets the video ID from URL or input, then uses `youtube-transcript-api` to fetch the transcript. Writes `transcript` and `video_id` to state. On failure, sets `error` and the graph stops. |
| **fetch_important_frames** | Downloads the video with `yt-dlp`, extracts frames at **scene changes** (or by interval if few scenes), then uses a **vision LLM** (GPT-4o) to keep only educationally useful frames (diagrams, slides, equations). Applies a cap (e.g. 6 images) and writes `important_frames` to state. Optional: skip with no video download if you only want notes. |
| **generate_notes** | Splits the transcript into overlapping chunks, has the LLM write notes per chunk, then **merges** chunk notes into one final note set. Uses prompts in `prompts.py` (structure, math format, no fluff). Writes `notes`, `chunks`, `partial_notes`, `chunk_metadata`. |
| **review_quality** | A separate **reviewer** LLM compares notes to the transcript and returns PASS or FAIL plus feedback. Writes `review_passed` and `review_feedback`. |
| **revise_notes** | If the reviewer failed, this node takes the transcript, current notes, and feedback and produces a revised version. Writes updated `notes` and increments `iteration`. Then the graph goes back to **review_quality**. The loop stops when the reviewer passes or `MAX_ITERATIONS` (e.g. 3) is reached. |

After **review_quality**, the graph either:

- **pass** → go to **interview_topic_miner** (Phase 2), or  
- **pass_skip_interview** → end (if `--skip-interview`), or  
- **revise** → **revise_notes** → back to **review_quality**.

### Phase 2: Interview question mining

Uses **the same** `transcript` and `notes` from Phase 1 (no re-fetch or re-generation).

| Node | What it does |
|------|----------------|
| **interview_topic_miner** | LLM reads transcript + notes and outputs a **list of topics** (e.g. “Gradient Descent”, “Batch GD”) with aliases, category, evidence quote, and priority. Output is strict JSON (`topics_data`). If the critic had asked for topic changes, it uses `feedback_node1` and can clear the previous question bank. |
| **interview_question_harvester** | For each topic (up to a limit), builds search queries and runs **web search** (Tavily if `TAVILY_API_KEY` is set, else DuckDuckGo). Passes snippets + URLs to the LLM and asks it to extract **only** questions that appear in those snippets with **company name and source URL**. Writes `question_bank`, `excluded_unattributed`, `search_coverage`. On “research more” loops, it merges new results into the existing bank. |
| **interview_critic** | Evaluates topics + question bank (attribution, coverage, duplicates). Returns **APPROVE_AND_EXPORT**, **REVISE_TOPICS**, or **RESEARCH_MORE**. Writes `critic_decision`, `critic_full`, `interview_cycles`, and optional `feedback_node1` / `node2_extra_queries` for the next loop. After a max number of cycles (e.g. 3), it forces export. |
| **interview_format_export** | Turns `question_bank` and metadata into one markdown string: summary table, coverage report, and “Topic → Company → Questions” with source URL and evidence. Writes `interview_bank_markdown`. |

The critic can loop:

- **REVISE_TOPICS** → back to **interview_topic_miner**  
- **RESEARCH_MORE** → back to **interview_question_harvester**  
- **APPROVE_AND_EXPORT** (or max cycles) → **interview_format_export** → **END**

---

## Setup

- **Python**: 3.10+
- **Install**: `pip install -r requirements.txt`
- **Environment**: Copy `.env.example` to `.env` and set:
  - `OPENAI_API_KEY` — required for notes and interview nodes.
  - `TAVILY_API_KEY` — optional; used for interview web search (if unset, DuckDuckGo is used).

---

## Run

From the project root:

```bash
# Full run: notes + key images + interview question bank
python run.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Or with video ID only
python run.py VIDEO_ID

# Output directory (default: outputs)
python run.py VIDEO_ID --output-dir my_outputs

# Skip interview mining (notes + key images only)
python run.py VIDEO_ID --skip-interview

# Limit key images (default 6)
python run.py VIDEO_ID --max-images 8

# Export workflow as PNG or Mermaid
python run.py VIDEO_ID --save-graph-png --save-graph-mermaid
```

---

## Outputs

- **Notes + key images**:  
  `{output_dir}/{title}_notes.md` and `{title}_notes.docx`  
  Include metadata, notes section, and “Key images” (with captions and timestamps).

- **Interview question bank**:  
  - Appended at the end of the same `.md` and `.docx`.  
  - Standalone: `{output_dir}/interview_question_bank.md` and `interview_question_bank.docx`  
  Questions are sourced from the web with company name, role/level (if found), source URL, and evidence snippet; the pipeline does not invent questions.

---

## Project layout (main files)

| Path | Role |
|------|------|
| `run.py` | CLI: parse args, build initial state, invoke graph, save notes + append/save interview bank. |
| `graph.py` | Builds the single LangGraph (all nodes and edges, routing after review and after critic). |
| `state.py` | Shared state schema (`NotesWorkflowState`) and constants (`MAX_ITERATIONS`, `MAX_INTERVIEW_LOOPS`). |
| `nodes.py` | Notes pipeline nodes: fetch_transcript, fetch_important_frames_node, generate_notes, review_quality, revise_notes. |
| `frames.py` | Video download, scene-based frame extraction, vision-based “important frame” selection, top-N cap. |
| `interview_nodes.py` | Interview nodes: interview_topic_miner, interview_question_harvester, interview_critic, interview_format_export. |
| `interview_search.py` | Web search for harvester (Tavily from env, else DuckDuckGo); builds queries per topic. |
| `prompts.py` | System/user prompts for notes generation, merge, review, and revision. |
| `prompts_interview.py` | System prompts for topic miner, question harvester, and critic. |
| `chunking.py` | Splits transcript into overlapping chunks for generate_notes. |
| `video.py` | YouTube URL/ID parsing and transcript fetch. |
| `export_utils.py` | Save notes + key images to .md/.docx; append interview bank; save standalone interview_question_bank files. |

Together, these form one end-to-end flow: **YouTube URL → notes + key images + interview question bank**, with every node briefly described above.
