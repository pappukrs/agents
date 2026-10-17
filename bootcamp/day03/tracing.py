"""Step-by-step trace of an agent run: what it thought, which tool, args, result, tokens, ms."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Step:
    n: int
    kind: str  # "llm" | "tool" | "final" | "stop"
    text: str = ""
    tool: str | None = None
    args: dict | None = None
    result: object = None
    ok: bool = True
    ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Trace:
    question: str
    steps: list[Step] = field(default_factory=list)
    verbose: bool = True
    started: float = field(default_factory=time.time)

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        if self.verbose:
            print(self.format(step), flush=True)
        return step

    @staticmethod
    def format(s: Step) -> str:
        if s.kind == "llm":
            say = f" thought: {s.text.strip()[:160]}" if s.text.strip() else ""
            return f"[{s.n}] LLM   {s.ms:>5}ms  in={s.input_tokens} out={s.output_tokens}{say}"
        if s.kind == "tool":
            mark = "ok " if s.ok else "ERR"
            res = json.dumps(s.result, default=str)
            return f"[{s.n}] TOOL  {s.ms:>5}ms  {mark} {s.tool}({json.dumps(s.args)}) -> {res[:160]}"
        return f"[{s.n}] {s.kind.upper():<5} {s.text.strip()[:300]}"

    def totals(self) -> dict:
        llm = [s for s in self.steps if s.kind == "llm"]
        return {
            "llm_calls": len(llm),
            "tool_calls": sum(s.kind == "tool" for s in self.steps),
            "tool_errors": sum(s.kind == "tool" and not s.ok for s in self.steps),
            "input_tokens": sum(s.input_tokens for s in llm),
            "output_tokens": sum(s.output_tokens for s in llm),
            "wall_ms": round((time.time() - self.started) * 1000),
        }

    def save(self, folder: str | Path = "traces") -> Path:
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        path = folder / f"run-{time.strftime('%Y%m%d-%H%M%S')}.json"
        data = {"question": self.question, "totals": self.totals(), "steps": [asdict(s) for s in self.steps]}
        path.write_text(json.dumps(data, indent=2, default=str))
        return path
