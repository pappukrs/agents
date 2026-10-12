# 30-Day AI Agent Engineering Roadmap

**Who:** Pappu — ~5 years coding, strong in JavaScript/TypeScript, knows Python, backend/API/DB experience.
**Goal:** Go from "knows how to code" to "builds, evaluates, fine-tunes and ships AI agents", with a hackathon-ready demo by Day 10.

No basic-programming days. Every day has **three blocks**: **Learn → Build → Test/Deliver**. Aim for 2.5–3 hours/day: ~1 h concepts, ~1 h coding, 30–60 min testing + writing notes.

**Stack:** TypeScript + Node.js agent backend · Python for ML/fine-tuning · Google Gemini via `@google/genai` (gemini-2.5-flash default, gemini-2.5-pro for hard reasoning; Ollama locally for cheap experiments) · PostgreSQL + pgvector · React/Next.js dashboard when a UI is needed.

---

## Phases

| Phase | Days | Theme | Outcome |
|-------|------|-------|---------|
| 1 — Hackathon sprint | 1–10 | LLM APIs, structured output, tool calling, agent loops, RAG, state, eval, deploy | Working demo + pitch |
| 2 — Advanced agent systems | 11–20 | Multi-agent orchestration, memory, vector DBs, advanced retrieval, tracing, guardrails, MCP | Smarter, observable, safer agent |
| 3 — Fine-tuning & production | 21–30 | Fine-tuning fundamentals, LoRA/QLoRA, datasets, model eval, serving, cost, capstone | Fine-tuned model + shipped capstone |

---

## Phase 1 — Hackathon sprint (Days 1–10)

Target: a working project with a live demo, not just a chatbot.

### Day 1 — How LLMs actually work
- **Learn:** Tokens, context windows, system prompts, temperature/top-p, inference, hallucination, embeddings vs generation, pricing by token.
- **Build:** TypeScript CLI/app that sends a prompt to an LLM and returns a response. Configurable system instructions, model, temperature.
- **Test:** Compare 2–3 prompts on 10 inputs; record quality, latency, token usage in a table.
- **Deliverable:** LLM playground with a test-case file.

### Day 2 — Structured outputs and prompting
- **Learn:** Zero-shot / few-shot, chain-of-thought vs direct answers, JSON schema, Zod validation, output constraints, retries on invalid output.
- **Build:** Text classifier that returns typed JSON: `category`, `confidence`, `reasoning`, `nextAction`. Validate with Zod.
- **Test:** Malformed inputs, missing fields, ambiguous requests. Every response must validate or be retried.
- **Deliverable:** Reliable structured-output API endpoint.

### Day 3 — Tool calling: your first agent
- **Learn:** LLM vs agent, tool schemas, tool execution, observation/action loop, error returns to the model.
- **Build:** Agent with 3 tools: search a local knowledge file, query a database, calculate a score.
- **Test:** Does the model pick the right tool? Does it recover when a tool throws?
- **Deliverable:** Agent that performs actions, not just text.

### Day 4 — Agent loops and workflows
- **Learn:** Planning, tool selection, stop conditions, retries, deterministic workflows vs autonomous agents, step budgets.
- **Build:** Bounded agent loop: max 5 tool steps, explicit completion condition, structured final answer.
- **Test:** Timeouts, repeated identical tool calls, impossible tasks — the loop must terminate cleanly.
- **Deliverable:** Controlled agent execution engine.

### Day 5 — RAG and embeddings
- **Learn:** Embeddings, chunking strategies, cosine similarity, top-k retrieval, context injection, citations, "I don't know" behavior.
- **Build:** Upload docs → chunk → embed → store in pgvector → retrieve → answer with sources.
- **Test:** 15 questions including 5 the docs cannot answer. Require source references; no fabricated citations.
- **Deliverable:** Document Q&A agent.

### Day 6 — Memory and state
- **Learn:** Conversation history, short-term vs long-term memory, checkpoints, session IDs, summarizing old context.
- **Build:** Persist conversations and agent state in Postgres. Resume a task after a process restart.
- **Test:** Two users' sessions stay separate; prior context restores correctly after restart.
- **Deliverable:** Stateful agent with persistence.

### Day 7 — Multi-agent systems
- **Learn:** Planner / researcher / executor / reviewer roles; supervisor vs peer patterns; when multi-agent is worth the cost.
- **Build:** Research agent → analysis agent → reviewer with explicit typed handoffs.
- **Test:** Compare against single-agent baseline on the same 10 tasks. Measure extra latency and cost.
- **Deliverable:** Multi-agent prototype with a measured reason to exist.

### Day 8 — Build the actual hackathon product
- **Learn:** API design, streaming responses (SSE), UI loading/error states, validation, secrets management.
- **Build:** React/Next.js UI, backend endpoints, DB schema, primary user workflow end to end.
- **Test:** Full workflow on realistic sample data; show intermediate agent steps in the UI.
- **Deliverable:** End-to-end working application in `project/`.

### Day 9 — Evaluation, safety, reliability
- **Learn:** Golden test sets, accuracy, groundedness, latency, cost, prompt injection, tool permissions, human-in-the-loop approval.
- **Build:** Test runner, structured logs, schema validation, bounded retries, approval step before consequential actions.
- **Test:** Run 20–30 cases; fix the top 3 most serious failures.
- **Deliverable:** Evidence the demo works beyond one happy path.

### Day 10 — Deploy and rehearse
- **Learn:** Env vars, deployment logs, API key security, rate limits, fallback behavior, demo mode.
- **Build:** Deploy backend + frontend, seed data, simple demo mode that cannot fail live.
- **Present:** Record a 2-minute demo: problem → architecture → agent decisions → evaluation → impact.
- **Deliverable:** Hackathon-ready demo, repo, README, pitch.

### The sprint project — AI Lead Intelligence Agent
Adaptable to any hackathon theme. Takes records/documents, researches evidence, classifies intent, scores 1–100, explains each score, recommends next action.

```
React / Next.js dashboard      upload data · view scores · inspect evidence
        │
TypeScript agent orchestrator  classify → retrieve → analyze → score → review
        │
PostgreSQL + pgvector          records · run history · evidence chunks + citations
```

Swap "lead scoring" for the hackathon use case without throwing away the agent infrastructure.

---

## Phase 2 — Advanced agent systems (Days 11–20)

### Day 11 — Hackathon retrospective + architecture cleanup
- **Learn:** Read your own logs. What broke, what was slow, what cost the most tokens. Agent architecture patterns (router, pipeline, loop, supervisor).
- **Build:** Refactor the project into clear modules: `llm/`, `tools/`, `agents/`, `memory/`, `eval/`. Add typed interfaces between them.
- **Test:** All Day 9 eval cases still pass after refactor.
- **Deliverable:** Clean, modular agent codebase + written retro in `days/day11/notes.md`.

### Day 12 — Observability and tracing
- **Learn:** Traces, spans, token accounting per step, OpenTelemetry concepts, tools like Langfuse / LangSmith / Braintrust (pick one, free tier).
- **Build:** Instrument every LLM call and tool call: inputs, outputs, latency, tokens, cost. Trace viewer per run.
- **Test:** Find the single most expensive step across 20 runs and cut its cost 30%.
- **Deliverable:** Traced agent with a cost/latency dashboard.

### Day 13 — Advanced retrieval
- **Learn:** Hybrid search (BM25 + vectors), reranking, query rewriting, HyDE, metadata filtering, parent-child chunking.
- **Build:** Add hybrid search + a reranker to the Day 5 RAG. Query rewriting before retrieval.
- **Test:** Build a 30-question retrieval eval; measure recall@5 before vs after.
- **Deliverable:** RAG v2 with measured recall improvement.

### Day 14 — Long-term memory
- **Learn:** Episodic vs semantic memory, memory extraction, memory consolidation, forgetting, user profiles.
- **Build:** Agent that extracts facts from conversations, stores them as embeddings + metadata, and recalls them in later sessions.
- **Test:** 5 multi-session scenarios; memory must recall the right fact and not hallucinate stored facts.
- **Deliverable:** Agent with persistent cross-session memory.

### Day 15 — MCP: standard tool interfaces
- **Learn:** Model Context Protocol — servers, tools, resources, prompts; why standard tool interfaces matter.
- **Build:** Write an MCP server exposing your project's tools (DB query, scoring, retrieval). Connect it to Claude Code / an MCP client.
- **Test:** Call every tool through the MCP client; verify schemas and error handling.
- **Deliverable:** Your agent's tools usable from any MCP host.

### Day 16 — Multi-agent orchestration, properly
- **Learn:** Supervisor pattern, handoff protocols, shared state vs message passing, parallel fan-out/fan-in, failure isolation.
- **Build:** Supervisor that dispatches parallel sub-agents, merges results, retries failed branches.
- **Test:** Kill one sub-agent mid-run; the system must still produce a partial, honest result.
- **Deliverable:** Fault-tolerant multi-agent orchestrator.

### Day 17 — Guardrails and security
- **Learn:** Prompt injection (direct and via retrieved docs), tool permission scopes, output filtering, PII handling, sandboxing.
- **Build:** Input/output guards, a tool allowlist per agent role, injection detection on retrieved content, red-team test file.
- **Test:** 15 attack prompts (injection, exfiltration, jailbreak); log which get through, fix the critical ones.
- **Deliverable:** Hardened agent + security test suite.

### Day 18 — Agent evaluation at scale
- **Learn:** LLM-as-judge, pairwise comparison, rubric scoring, trajectory evaluation (did it take the right steps?), regression suites.
- **Build:** Eval harness: dataset → run agent → judge → report. Run in CI on every commit.
- **Test:** Judge agreement with your own manual labels on 30 cases; tune the judge prompt until agreement >85%.
- **Deliverable:** Automated agent eval pipeline with a scorecard.

### Day 19 — Streaming, UX, and human-in-the-loop
- **Learn:** Streaming tool calls, partial results, interruption, approval flows, showing agent reasoning safely.
- **Build:** UI that streams agent steps live, lets the user approve/reject a consequential action, and lets them edit the plan mid-run.
- **Test:** Approve, reject, and edit flows all work; rejected actions never execute.
- **Deliverable:** Interactive agent UI.

### Day 20 — Phase 2 integration day
- **Learn:** Review Days 11–19 notes; identify the 3 weakest parts of the system.
- **Build:** Integrate tracing, RAG v2, memory, guardrails, and eval into the main project. Fix the 3 weak spots.
- **Test:** Full eval suite + security suite green. Record cost per run.
- **Deliverable:** Agent v2 — observable, safer, measured. Tag release `v2.0`.

---

## Phase 3 — Fine-tuning and production (Days 21–30)

Python-heavy. Use Google Colab (free T4) or Kaggle if no local GPU.

### Day 21 — Fine-tuning fundamentals
- **Learn:** Pretraining vs SFT vs RLHF/DPO, when fine-tuning beats prompting/RAG, full fine-tune vs PEFT, loss curves, overfitting.
- **Build:** Python env with `transformers`, `peft`, `trl`, `datasets`. Load a small open model (e.g. Llama 3.x 1B/3B or Qwen 2.5 1.5B), run inference.
- **Test:** Baseline the small model on 20 of your project's classification cases; record accuracy.
- **Deliverable:** Fine-tuning environment + baseline numbers.

### Day 22 — Dataset preparation
- **Learn:** Instruction format (chat templates), data quality > quantity, dedup, decontamination, train/val/test splits, synthetic data from a stronger model.
- **Build:** Export your project's logged runs → clean → convert to chat-format JSONL. Generate extra synthetic examples with a frontier model, then filter them.
- **Test:** Inspect 50 random examples by hand; reject bad ones. Target 500–2000 clean examples.
- **Deliverable:** Versioned training dataset with a data card.

### Day 23 — LoRA / QLoRA fine-tune
- **Learn:** LoRA math (rank, alpha, target modules), 4-bit quantization (QLoRA), learning rate, epochs, gradient accumulation.
- **Build:** QLoRA fine-tune the small model on your dataset with `trl` SFTTrainer. Log to Weights & Biases or TensorBoard.
- **Test:** Watch train/val loss; stop at the best val checkpoint. Compare accuracy to the Day 21 baseline.
- **Deliverable:** First fine-tuned adapter + training report.

### Day 24 — Model evaluation
- **Learn:** Held-out eval, exact match vs judge scoring, calibration, regressions on general ability, comparing to the frontier model.
- **Build:** Eval script: fine-tuned model vs base model vs frontier API on the same 100 test cases. Cost and latency per 1k requests.
- **Test:** Produce a comparison table. Decide honestly whether the fine-tune is worth it.
- **Deliverable:** Model comparison report with a go/no-go decision.

### Day 25 — Preference tuning (DPO) and iteration
- **Learn:** DPO / ORPO basics, building preference pairs, reward hacking, when SFT is enough.
- **Build:** Create 200 preference pairs (chosen/rejected) from judge outputs. Run DPO on top of the SFT adapter.
- **Test:** Re-run Day 24 eval. Did output quality (judge score) improve without hurting accuracy?
- **Deliverable:** SFT+DPO adapter + updated comparison.

### Day 26 — Serving a fine-tuned model
- **Learn:** Merging adapters, GGUF conversion, quantized inference, vLLM / Ollama / llama.cpp, OpenAI-compatible endpoints, batching.
- **Build:** Merge adapter → export GGUF → serve locally with Ollama (or vLLM on a GPU). Point your TypeScript agent at it as a provider.
- **Test:** Agent runs end-to-end on the local model; measure tokens/sec and p95 latency.
- **Deliverable:** Self-hosted model behind your agent.

### Day 27 — Model routing and cost optimization
- **Learn:** Routing by task difficulty, cascading (small model first, escalate on low confidence), prompt caching, batching, caching retrievals.
- **Build:** Router: fine-tuned small model for classification/scoring, frontier model for reasoning-heavy steps. Add response caching.
- **Test:** Compare cost and quality of router vs frontier-only across 100 runs.
- **Deliverable:** Hybrid routing with a cost report.

### Day 28 — Production deployment
- **Learn:** Dockerizing agent + model server, health checks, secrets, rate limiting, queues for long runs, retries/idempotency, monitoring + alerts.
- **Build:** Docker Compose (or k8s manifests) for API + worker + Postgres + model server. Deploy to a cloud VM / Fly.io / Railway.
- **Test:** Load test 50 concurrent runs; no dropped jobs, alerts fire on failure.
- **Deliverable:** Production deployment with monitoring.

### Day 29 — Capstone build
- **Learn:** Write the architecture doc first: problem, users, agents, tools, data flow, eval, costs, risks.
- **Build:** Finish the capstone — the Phase 1 project upgraded with Phase 2 systems and the Phase 3 fine-tuned model.
- **Test:** Full eval + security + load suites. Everything green or documented.
- **Deliverable:** Capstone repo, architecture doc, demo video.

### Day 30 — Ship, write, reflect
- **Learn:** How to present technical work: README, blog post, demo video, talking through tradeoffs.
- **Build:** Polish README, publish a blog post / LinkedIn writeup, record final demo, tag `v3.0`.
- **Test:** A friend clones the repo and runs it from the README alone.
- **Deliverable:** Public portfolio project + written reflection on what to learn next (e.g. RL for agents, computer-use agents, voice agents).

---

## Tracking

Mark each day in this table as you go.

| Day | Done | Notes |
|-----|------|-------|
| 1 | ☐ | |
| 2 | ☐ | |
| 3 | ☐ | |
| 4 | ☐ | |
| 5 | ☐ | |
| 6 | ☐ | |
| 7 | ☐ | |
| 8 | ☐ | |
| 9 | ☐ | |
| 10 | ☐ | Hackathon |
| 11 | ☐ | |
| 12 | ☐ | |
| 13 | ☐ | |
| 14 | ☐ | |
| 15 | ☐ | |
| 16 | ☐ | |
| 17 | ☐ | |
| 18 | ☐ | |
| 19 | ☐ | |
| 20 | ☐ | v2.0 |
| 21 | ☐ | |
| 22 | ☐ | |
| 23 | ☐ | |
| 24 | ☐ | |
| 25 | ☐ | |
| 26 | ☐ | |
| 27 | ☐ | |
| 28 | ☐ | |
| 29 | ☐ | |
| 30 | ☐ | v3.0 |

## References
- Vercel AI SDK — tool calling and structured generation in TypeScript
- Anthropic / OpenAI API docs — tool use, structured outputs, prompt caching
- Model Context Protocol — https://modelcontextprotocol.io
- Hugging Face `transformers`, `peft`, `trl` docs — fine-tuning
- pgvector — https://github.com/pgvector/pgvector
