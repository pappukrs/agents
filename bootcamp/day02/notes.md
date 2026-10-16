# Day 2 — Fri 16 Oct · Tool calling + the agent loop (most important day)

## Learn (45 min) — open `concepts.html` in a browser first, then write the answers in your own words

1. **LLM vs agent** — Day 1's classifier made one call. What makes today's program an *agent*? (Hint: who decides the next step?)
2. **Tool calling** — the model never runs your code. What does it actually return, and who executes the function?
3. **Tool schemas** — where does the schema the model sees come from in `tools.py`? Why is generating it from type hints better than writing JSON by hand?
4. **The loop** — write the loop in 5 lines of pseudocode from memory. What gets appended to `contents` after each tool call?
5. **Stop conditions** — name the 3 ways `run_agent` can stop. Why is `max_steps` not optional?
6. **Tool errors as observations** — `run_tool` never raises. Why is `{"error": "no column 'area'. Columns: [...]"}` better than crashing?
7. **Context growth** — every tool result is re-sent on every later step. What does `truncate()` protect against?
8. **Security** — the model chooses file paths. What stops `read_file("../.env")`? What stops `calculator("__import__('os')...")`?

## Build — what's here

| File | Purpose |
|------|---------|
| `tools.py` | `@tool` decorator: builds a JSON schema from type hints + docstring `Args:`. `run_tool()` validates args, enforces a timeout, truncates results, turns every failure into `{"error": ...}`. |
| `builtin_tools.py` | `calculator` (safe AST evaluator, no `eval`), `list_files`, `search_files`, `read_file`, `analyze_csv` (pandas). All file access sandboxed to `data/`. |
| `loop.py` | `run_agent()`: the hand-written loop. Model turn → run calls → append results → repeat, capped by `max_steps`. |
| `tracing.py` | `Trace` / `Step`: prints every step live (thought, tool, args, result, tokens, ms) and saves JSON to `traces/`. |
| `llm.py` | Day 1 wrapper + new `chat()` for multi-turn calls with tool declarations (automatic function calling off). |
| `agent.py` | CLI: one question, `--file questions.jsonl` (keyword check), or `--tools` to print the schemas. |
| `data/` | The agent's sandbox: `sales.csv`, `docs/*.md`, `notes.txt`. |
| `tests/` | 24 offline tests: schemas, sandbox, timeouts, and the loop driven by a scripted fake model. |

## Run

```bash
cd bootcamp/day02
cp .env.example .env                  # add GEMINI_API_KEY
source ../../.venv/bin/activate
pip install -r requirements.txt       # adds pandas
pytest -q                             # 24 offline tests
python agent.py --tools               # see exactly what the model sees
python agent.py "What's the average revenue per region in sales.csv, and which file mentions refunds?"
python agent.py --file questions.jsonl
```

## Exercises (Build + Test blocks)

- [ ] Run the target question. Paste the trace below. Count the steps: could it have been fewer?
- [ ] Run `--file questions.jsonl`. Record the scorecard.
- [ ] q5 asks for `../.env`. Read the trace: which layer refused it, the prompt or the code?
- [ ] Run with `--max-steps 2`. What happens? Is the message useful to a user?
- [ ] Delete the line "call analyze_csv with operation=describe first" from `SYSTEM`. Does it guess column names now? Do tool errors go up?
- [ ] Add a tool `top_n(path, column, n)` with a docstring. Don't touch the loop. Ask a question that needs it.
- [ ] **Claude Code practice:** spec a `write_note(filename, text)` tool that may only write to `data/out/`, with a test that `../` is refused. Give it to Claude Code, review the diff, run the tests.

## Target question trace

```
(paste here)
```

## Results

| Run | Model | Questions passed | Avg steps | Tool errors | Tokens |
|-----|-------|------------------|-----------|-------------|--------|
| 1 | gemini-2.5-flash | /5 | | | |

## What I learned

-

## Harvest → `starter/`

- [ ] `tools.py` → `starter/agent/tools.py`
- [ ] `loop.py` → `starter/agent/loop.py`
- [ ] `tracing.py` → `starter/agent/tracing.py` (not `trace.py`: that name shadows a stdlib module)
- [ ] `llm.py` (with `chat()`) → `starter/agent/llm.py`
