# Day 1 — Thu 15 Oct · LLM fundamentals in Python + structured output

## Learn (45 min) — open `concepts.html` in a browser first, then write the answers in your own words

1. **Tokens** — what is a token, roughly how many per English word, why do input and output tokens cost different amounts?
2. **Context window** — what happens when prompt + history + output exceed it? What fills it fastest in an agent? (Hint: tool results.)
3. **Temperature** — why `0.0` for classification/extraction and higher for brainstorming?
4. **Thinking tokens** — `gemini-2.5-flash` can "think" before answering. They're billed and count toward `max_output_tokens`. `classify.py` sets `thinking_budget=0`. When would you turn it back on?
5. **Hallucination** — name 3 ways to reduce it (grounding, "say I don't know", schema constraints, low temperature...).
6. **Why validate JSON?** Even with `response_schema` the model can return truncated or wrong-typed output. What does `generate_structured` do when that happens?
7. **Prompt injection** — ticket `t09` tries to override the instructions. How does the system prompt defend against it? Is that enough? (Day 6 goes deeper.)

## Build — what's here

| File | Purpose |
|------|---------|
| `llm.py` | `generate()` with retries on 429/5xx + backoff, timing, token counts. `generate_structured()` validates with Pydantic and re-asks once with the validation error. |
| `schemas.py` | `Ticket` Pydantic model (Literal enums = model can't invent categories). |
| `classify.py` | CLI: classify one message, or run `tickets.jsonl` and score category/priority. |
| `tickets.jsonl` | 10 inputs incl. Spanish, gibberish, legal threat, prompt injection, multi-issue. |
| `tests/test_llm.py` | Offline tests with a fake client — no key needed. |

## Run

```bash
cd bootcamp/day01
cp .env.example .env            # add GEMINI_API_KEY
source ../../.venv/bin/activate # venv is at repo root
pytest -q                       # offline tests
python classify.py "I was charged twice for order 123"
python classify.py --file tickets.jsonl
```

## Exercises (Build + Test blocks)

- [ ] Run the sample set. Record the scorecard below.
- [ ] Find the tickets it gets wrong. Is the model wrong or is the `expected` label debatable? Fix the prompt *or* the label — write down which and why.
- [ ] Set `thinking_budget=None` (default thinking). Compare accuracy, latency, tokens.
- [ ] Try `gemini-2.5-pro` via `GEMINI_MODEL`. Is it worth the latency?
- [ ] Add 5 tickets of your own, including one adversarial one.
- [ ] **Claude Code practice:** write a 5-line spec for a second classifier (e.g. resume → `{name, years_experience, skills[], seniority}`), give it to Claude Code, review the diff, run it.

## Results

| Run | Model | Thinking | Valid | Category | Priority | Avg ms | Tokens |
|-----|-------|----------|-------|----------|----------|--------|--------|
| 1 | gemini-2.5-flash | 0 | | | | | |

## What I learned

-

## Harvest → `starter/`

- [ ] `llm.py` → `starter/agent/llm.py` (once live run works)
