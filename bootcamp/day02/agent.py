"""CLI for the Day 2 agent.

Usage:
    python agent.py "What's the average revenue per region in sales.csv?"
    python agent.py --file questions.jsonl          # run all questions, check expected keywords
    python agent.py --tools                         # print the schemas the model sees
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import builtin_tools  # noqa: F401 - importing registers the tools
from loop import run_agent
from tools import declarations


def ask(question: str, max_steps: int) -> None:
    result = run_agent(question, max_steps=max_steps, thinking_budget=0)
    print(f"\nANSWER ({result.stopped_because}):\n{result.answer}")
    print(f"\n{result.trace.totals()}")
    print(f"trace -> {result.trace.save()}")


def run_file(path: Path, max_steps: int) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    passed = 0
    for row in rows:
        print(f"\n=== {row['id']}: {row['question']}")
        result = run_agent(row["question"], max_steps=max_steps, thinking_budget=0, verbose=False)
        answer = result.answer.replace(",", "")
        missing = [k for k in row.get("expect", []) if k.lower() not in answer.lower()]
        leaked = [k for k in row.get("must_not_contain", []) if k in answer]
        ok = result.stopped_because == "final" and not missing and not leaked
        passed += ok
        t = result.trace.totals()
        print(f"{'PASS' if ok else 'FAIL'}  steps={t['llm_calls']} tools={t['tool_calls']} "
              f"errors={t['tool_errors']} tokens={t['input_tokens'] + t['output_tokens']} ms={t['wall_ms']}")
        if missing or leaked:
            print(f"  missing={missing} leaked={leaked}")
        print(f"  {result.answer.strip()[:300]}")
        result.trace.save()
    print(f"\n{passed}/{len(rows)} passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--file", type=Path)
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()
    if args.tools:
        print(json.dumps(declarations(), indent=2))
    elif args.file:
        run_file(args.file, args.max_steps)
    elif args.question:
        ask(args.question, args.max_steps)
    else:
        ap.error("give a question, --file or --tools")


if __name__ == "__main__":
    main()
