# Day 1 — How LLMs actually work

## Learn
Concepts to cover (write 2-3 lines each in your own words):
- Tokens — subword units; you pay per token in and out. ~4 chars ≈ 1 token in English.
- Context window — max input + output tokens per call. Gemini 2.5 Flash: 1M in.
- System prompt — standing instructions; separate from user turns.
- Temperature / top-p — randomness in sampling. 0 = near-deterministic.
- Inference — one forward pass per generated token; latency grows with output length.
- Hallucination — fluent but false output; cases c05, c08, c10 are designed to trigger it.
- Embeddings vs generation — embeddings turn text into vectors for similarity; generation produces text.

## Build
```
cd days/day01
npm install
cp .env.example .env   # add GEMINI_API_KEY
npm run ask -- "What is a token?" --system "Answer in one sentence." --temp 0.2
```

## Test / Deliver
```
npm run bench            # 3 prompts x 10 cases -> results/bench-*.md
TEMP=1.0 npm run bench   # repeat at high temperature, compare
```
Then open the results file and fill the `quality (1-5)` column by hand.
Questions to answer in Notes below:
1. Which system prompt hallucinated least on c05/c08/c10?
2. How much did the structured prompt increase output tokens?
3. Latency vs output length — roughly linear?

## Notes
