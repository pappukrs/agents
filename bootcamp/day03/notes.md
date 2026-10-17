# Day 3 — Sat 17 Oct · RAG over supplied documents

## Learn (45 min) — open `concepts.html` in a browser first, then write the answers in your own words

1. **Why RAG** — why not paste every document into the prompt? Give 3 reasons.
2. **Embeddings** — what is an embedding? Why must chunks and questions use the same model? What do `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` change?
3. **Cosine** — what does it measure? Why does `normalize()` at ingest turn it into `vectors @ q`?
4. **Chunking** — what goes wrong with chunks that are too big? Too small? What does overlap protect?
5. **Top-k vs threshold** — top-k always returns k results. How does the agent ever say "I don't know"?
6. **Reranking** — what is it, and when would you skip it on hackathon day?
7. **Citations** — the model cites `pricing.md#3`, but no search returned it. Which function catches that, and how?
8. **Prompt vs code** — list what the system prompt asks for, and what the code enforces regardless of the prompt.

## Build — what's here

| File | Purpose |
|------|---------|
| `docs/` | 8 policy docs for a made-up store (returns, refunds, warranty, repairs, shipping, payments, security, Care+ pricing). Drop your own `.md` / `.txt` / `.pdf` here. |
| `rag.py` | `chunk_text`, `normalize`, `ingest()` → `data/index.npz`, `search()` (cosine, top-k, `MIN_SCORE`), and the `search_docs` **tool**. CLI: `ingest`, `search`, `scores`. |
| `llm.py` | Day 2 wrapper + `embed()`: batches of 100, same retry/backoff as `generate`, 768 dims. |
| `ask.py` | Day 2's `run_agent` with `tools=["search_docs"]` and `RAG_SYSTEM`. `check_citations()` fails any cited id that wasn't retrieved. `--file` prints a scorecard. |
| `questions.jsonl` | 10 eval questions: 7 answerable (`expect` + `source` doc), 3 unanswerable (must refuse). |
| `tools.py`, `loop.py`, `tracing.py` | Day 2, unchanged. RAG is just one more tool. |
| `tests/` | 20 offline tests: a bag-of-words fake embedder + Day 2's scripted model. No key needed. |

## Run

```bash
cd bootcamp/day03
cp .env.example .env                      # add GEMINI_API_KEY
source ../../.venv/bin/activate
pip install -r requirements.txt           # adds numpy
pytest -q                                 # 20 offline tests
python rag.py ingest                      # docs/ -> data/index.npz (one batch of embeddings)
python rag.py search "return window"      # raw scores, keep/drop vs MIN_SCORE
python rag.py scores questions.jsonl      # best score per question, for picking the threshold
python ask.py "How many days do I have to return an item?"
python ask.py --file questions.jsonl      # scorecard
python ask.py --file questions.jsonl --min-score 0.65
```

`RAG_MIN_SCORE=0.65` in `.env` changes the default threshold without editing code.

## Exercises (Build + Test blocks)

- [ ] `python rag.py ingest`. How many chunks? Open one id with `rag.py search` and read its text.
- [ ] **Pick the threshold from data:** run `python rag.py scores questions.jsonl`. Write the lowest answerable and highest unanswerable score below, then set `MIN_SCORE` between them.
- [ ] Run `ask.py --file questions.jsonl`. Record the scorecard. Read every FAIL's trace in `traces/`.
- [ ] Set `--min-score 0.3`. Do the unanswerable questions now get "answered" with citations? That's a hallucination with a source attached.
- [ ] Set `--min-score 0.85`. Which answerable questions start refusing?
- [ ] Re-ingest with `--size 300 --overlap 50`, then `--size 2000`. Does the scorecard change?
- [ ] Add a 9th doc that contradicts `refunds.md` (e.g. "refunds take 10 days"). What does the agent say? How would you fix it? (Hint: dates, doc priority, or say both.)
- [ ] Stretch: replace the numpy `Index` with ChromaDB (`pip install chromadb`), same `search()` signature. The tests should still pass.
- [ ] Stretch: rerank. Take the top 8 cosine hits, score them with `generate_structured()` (0–10 relevance), keep the top 3. Does it fix any FAIL?
- [ ] **Claude Code practice:** spec "split chunks on Markdown headings (`## `) first, fall back to `chunk_text` for long sections; ids stay `file#n`; add tests". Give it to Claude Code, review the diff, re-run the scorecard.

## Why a real embedder, not keywords

With the bag-of-words fake from `tests/conftest.py`, "How many days do I have to return an item?" ranks `refunds.md#1` above `returns-policy.md#0`, and "How long is a password reset link valid?" ranks `returns-policy.md#0` first. Matching words is not the same as matching meaning. Compare with the real `rag.py scores` output.

## Threshold

| | Score |
|-|-------|
| Lowest answerable | |
| Highest unanswerable | |
| **MIN_SCORE chosen** | |

Why:

## Results

| Run | Embed model | MIN_SCORE | chunk size | Passed /10 | Avg searches | Tokens |
|-----|-------------|-----------|------------|------------|--------------|--------|
| 1 | gemini-embedding-001 (768) | 0.6 | 800/150 | | | |

## What I learned

-

## Harvest → `starter/`

- [ ] `rag.py` → `starter/agent/rag.py` (make `DOCS_DIR` / `INDEX_PATH` configurable)
- [ ] `embed()` + `_retrying()` → `starter/agent/llm.py`
- [ ] `check_citations()` → `starter/agent/rag.py` (or `starter/evals/` on Day 6)
