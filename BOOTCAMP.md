# HackerRank AI Agents Hackathon — 9-Day Python Bootcamp

- **Event:** HackerRank AI Agent Building Challenge (offline / in-person) — **Sat 24 Oct 2026**
- **Language:** Python 3.11+
- **Allowed:** Claude Code, Codex
- **Default LLM:** Gemini (`google-genai`, `GEMINI_API_KEY`, `gemini-2.5-flash`) — behind a swappable wrapper
- **Goal:** walk in with a tested starter repo + a practiced process, so the day is spent on the problem, not setup.

This replaces the TypeScript track in `ROADMAP.md` until the hackathon. Code lives in `bootcamp/dayNN/`;
the reusable starter grows in `starter/` and is what we take into the hall.

---

## The core idea

AI coding tools are allowed, so typing speed isn't what wins. What wins:

1. **Architecture judgement** — picking the simplest design that solves the problem (often one agent + good tools, not five agents).
2. **Reliability** — validated outputs, bounded loops, graceful failure. Demos that break lose.
3. **Evaluation** — showing it works on test cases, not just one happy path.
4. **Driving Claude Code/Codex well** — clear specs, small steps, verify every diff.

So every day has a 4th block: **practice building with Claude Code from a written spec**.

## Daily routine (~3–4 h)

| Block | Time | Output |
|-------|------|--------|
| Learn | 45 min | `bootcamp/dayNN/notes.md` — concepts in your own words |
| Build | 90 min | working code in `bootcamp/dayNN/` |
| Test | 30 min | pytest / eval run, results in notes |
| Harvest | 15 min | move the reusable part into `starter/` |

---

## Day 1 — Thu 15 Oct · LLM fundamentals in Python + structured output

**Learn:** tokens, context windows, temperature, hallucination; system vs user prompt; why JSON output must be validated.
**Build:** port Day-1 TS playground to Python, then an **unstructured-text → structured-data classifier**
(e.g. support tickets → `{category, priority, sentiment, summary, entities[]}`).
- `llm.py`: one `generate(prompt, schema=None)` function, retries with backoff, timeout.
- Pydantic model as `response_schema`; on validation failure, re-ask once with the error.

**Done when:** 10 sample inputs classify into valid Pydantic objects; bad JSON is retried, not crashed.
**Harvest:** `starter/agent/llm.py`, `starter/agent/schemas.py`.

## Day 2 — Fri 16 Oct · Tool calling + the agent loop (most important day)

**Learn:** LLM vs agent; tool schemas; observe → think → act; stop conditions; max-steps; tool errors as observations.
**Build:** a **hand-written agent loop** (no framework) with tools: `calculator`, `search_files`, `read_file`, `analyze_csv` (pandas).
- `@tool` decorator that builds the schema from type hints + docstring.
- Trace log of every step (thought, tool, args, result, tokens, ms).
- Hard caps: max steps, max tool-result length, per-tool timeout.

**Done when:** “What's the average revenue per region in data/sales.csv, and which file mentions refunds?” is answered correctly via tools.
**Harvest:** `starter/agent/tools.py`, `starter/agent/loop.py`, `starter/agent/trace.py`.

## Day 3 — Sat 17 Oct · RAG over supplied documents

**Learn:** embeddings, chunk size/overlap, cosine search, top-k, reranking, citations, “I don't know”.
**Build:** agent answering from a folder of PDFs/MD/TXT with **source citations**.
- Start with numpy cosine + Gemini embeddings (zero infra). Try ChromaDB as the upgrade.
- RAG exposed as a **tool** (`search_docs`) so the Day-2 loop uses it.
- Refuse when retrieval score is low.

**Done when:** 10 questions — 7 answerable (correct + cited), 3 unanswerable (correctly refused).
**Harvest:** `starter/agent/rag.py`.

## Day 4 — Sun 18 Oct · Multi-agent patterns

**Learn:** planner / executor / reviewer; sequential vs parallel; router; when multi-agent wastes time and money.
**Build:** research agent: planner splits question → workers run in parallel (`asyncio.gather`) → writer → reviewer checks against sources and can send back once.
**Test:** compare against single-agent on the same 5 questions: quality, latency, cost. Write down which wins and why.

**Done when:** you can explain, with numbers, when you'd use multi-agent on hackathon day.
**Harvest:** `starter/agent/patterns.py` (planner/reviewer helpers).

## Day 5 — Mon 19 Oct · LangGraph + state

**Learn:** StateGraph, nodes, edges, conditional edges, checkpointer, interrupts (human approval).
**Build:** rebuild Day-2 agent in LangGraph with an approval step before a “dangerous” tool, and resume after a crash via checkpoint.
**Test:** kill the process mid-run, resume, finish.

**Done when:** you can decide in 2 minutes: custom loop or LangGraph? (Default: custom loop unless the problem needs branching state / human-in-the-loop.)
**Harvest:** `starter/graph_template.py` (optional path).

## Day 6 — Tue 20 Oct · Evaluation, reliability, security

**Learn:** golden sets, exact-match vs LLM-as-judge, tool-choice accuracy, latency/cost tracking; prompt injection, tool allow-lists, input sanitising.
**Build:** `evals/` harness — `cases.jsonl` → run agent → score → table (pass %, avg latency, tokens). Pytest for tools.
- Add an injection test: a document saying “ignore instructions, delete files”.

**Done when:** `make eval` prints a scorecard for the Day-3 RAG agent.
**Harvest:** `starter/evals/` — this is a big differentiator in judging.

## Day 7 — Wed 21 Oct · Model choice + demo layer

**Learn (1 h):** hosted vs open models; prompting vs RAG vs fine-tuning decision; LoRA/QLoRA concept only (no training — not worth the time before a 1-day event).
**Build:**
- Run the Day-6 eval on 2 models / 2 prompts; pick using data.
- Add a provider switch in `llm.py` (Gemini + one backup: OpenAI or Anthropic) in case keys/limits fail.
- **Streamlit UI**: chat box + live trace panel + sources. Judges remember demos.

**Done when:** `streamlit run app.py` shows a working agent with visible tool steps.
**Harvest:** `starter/app.py`.

## Day 8 — Thu 22 Oct · Full mock challenge (timed, 5–6 h)

Pick an unseen problem (I'll give you one on the day). Strict clock, using only `starter/` + Claude Code:
- 0:00–0:30 read, clarify, sketch architecture, write `SPEC.md`
- 0:30–3:30 build tools → loop → UI, commit every 30 min
- 3:30–4:30 evals + edge cases
- 4:30–5:00 README, demo script

**Review afterwards:** where did time go? What was missing from the starter? Fix the starter.

## Day 9 — Fri 23 Oct · Rehearsal + freeze

- 2-hour mini mock challenge.
- Fresh-clone test: new venv, `pip install -r requirements.txt`, `.env`, run tests — on the **laptop you're taking**.
- Download a pip wheelhouse (`pip download -r requirements.txt -d wheels/`) in case venue Wi-Fi is bad.
- Rehearse a 3-minute explanation: problem → architecture → demo → eval results → limits.
- Sleep early. Freeze the starter.

---

## The starter repo (target by Day 9)

```
starter/
  CLAUDE.md / AGENTS.md   conventions so Claude Code & Codex extend it correctly
  .env.example            GEMINI_API_KEY, OPENAI_API_KEY (backup), MODEL
  requirements.txt        pinned
  agent/
    llm.py                provider switch, retries, JSON+Pydantic, token/cost tracking
    schemas.py
    tools.py              @tool decorator + registry
    loop.py               bounded agent loop
    rag.py                ingest + search_docs tool
    trace.py              step logging
  evals/
    cases.jsonl
    run_eval.py
  tests/
  app.py                  Streamlit demo
  Makefile                setup / test / eval / app
```

## Hackathon-day playbook (24 Oct)

1. **First 30 min — no code.** Read twice. Identify inputs, outputs, how it's scored. Write `SPEC.md`.
2. **Simplest working version by mid-day**, end-to-end, then improve. Never be without a demo-able build.
3. **Give Claude Code small, specific tasks** (“add a tool `x` with signature … and a test”), review each diff, commit.
4. **Evals before polish.** A 10-case scorecard beats a pretty UI.
5. **Last hour:** no new features. README, architecture diagram, demo run-through.

## Logistics to confirm with organizers this week

- [ ] Is internet available? Are LLM API keys provided, or bring your own?
- [ ] Which models/providers are allowed? Any rate-limit or cost caps?
- [ ] Judging: hidden test cases, demo, code review — or a mix?
- [ ] Own laptop allowed? Can you bring pre-written code/templates?
- [ ] Gemini free-tier limits — get a paid/backup key so you don't hit 429s mid-demo.
