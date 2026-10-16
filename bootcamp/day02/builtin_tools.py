"""The Day 2 tools. Importing this module registers them.

All file access is sandboxed to WORKSPACE: the model picks paths, so paths are untrusted input.
"""

from __future__ import annotations

import ast
import math
import operator
from pathlib import Path
from typing import Literal

import pandas as pd

from tools import tool

WORKSPACE = Path(__file__).parent / "data"
TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".jsonl", ".log"}


def safe_path(path: str) -> Path:
    """Resolve `path` inside WORKSPACE, refusing anything that escapes it (../, absolute paths, symlinks)."""
    root = WORKSPACE.resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root):
        raise PermissionError(f"'{path}' is outside the workspace")
    return p


def safe_glob(pattern: str) -> list[Path]:
    """Glob inside WORKSPACE; drops anything a '..' in the pattern would reach outside it."""
    root = WORKSPACE.resolve()
    return sorted(p for p in root.glob(pattern) if p.is_file() and p.resolve().is_relative_to(root))


# ---- calculator: a tiny safe evaluator. Never eval() model output. ----

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max, "log": math.log}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")


@tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression exactly. Use this for any math instead of doing it in your head.

    Args:
        expression: e.g. "(1200 + 950) / 2" or "sqrt(16) * 3". Supports + - * / // % ** and sqrt, round, abs, min, max, log.
    """
    return _eval(ast.parse(expression, mode="eval"))


@tool
def list_files(pattern: str = "**/*") -> list[str]:
    """List files in the workspace. Call this first when you don't know which files exist.

    Args:
        pattern: glob pattern relative to the workspace, e.g. "**/*.csv" or "docs/*".
    """
    root = WORKSPACE.resolve()
    return [str(p.relative_to(root)) for p in safe_glob(pattern)][:200]


@tool
def search_files(query: str, pattern: str = "**/*") -> list[dict]:
    """Case-insensitive text search across workspace files. Returns matching lines with file and line number.

    Args:
        query: text to look for, e.g. "refund".
        pattern: glob to restrict which files are searched, e.g. "docs/*.md".
    """
    root = WORKSPACE.resolve()
    hits = []
    for p in safe_glob(pattern):
        if p.suffix not in TEXT_SUFFIXES:
            continue
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if query.lower() in line.lower():
                hits.append({"file": str(p.relative_to(root)), "line": n, "text": line.strip()[:200]})
                if len(hits) >= 50:
                    return hits
    return hits


@tool
def read_file(path: str, max_chars: int = 3000) -> str:
    """Read a text file from the workspace.

    Args:
        path: path relative to the workspace, e.g. "docs/q3-review.md".
        max_chars: stop after this many characters.
    """
    p = safe_path(path)
    if p.suffix not in TEXT_SUFFIXES:
        raise ValueError(f"not a text file: {path}")
    return p.read_text(errors="replace")[:max_chars]


@tool(timeout_s=20)
def analyze_csv(
    path: str,
    operation: Literal["describe", "head", "group_by"] = "describe",
    group_by: str | None = None,
    column: str | None = None,
    agg: Literal["mean", "sum", "count", "min", "max", "median"] = "mean",
) -> dict:
    """Analyze a CSV file with pandas. Use 'describe' first to learn the columns, then 'group_by' to aggregate.

    Args:
        path: CSV path relative to the workspace, e.g. "sales.csv".
        operation: describe = columns, dtypes, row count, numeric stats; head = first 5 rows; group_by = aggregate column per group.
        group_by: column to group by (required for group_by).
        column: numeric column to aggregate (required for group_by).
        agg: aggregation to apply to the column.
    """
    df = pd.read_csv(safe_path(path))
    if operation == "head":
        return {"rows": df.head(5).to_dict(orient="records")}
    if operation == "describe":
        return {
            "rows": len(df),
            "columns": {c: str(t) for c, t in df.dtypes.items()},
            "numeric_stats": df.describe().round(2).to_dict(),
        }
    for name, value in (("group_by", group_by), ("column", column)):
        if not value:
            raise ValueError(f"'{name}' is required for group_by")
        if value not in df.columns:
            raise ValueError(f"no column '{value}'. Columns: {list(df.columns)}")
    result = df.groupby(group_by)[column].agg(agg).round(2)
    return {"group_by": group_by, "column": column, "agg": agg, "result": result.to_dict()}
