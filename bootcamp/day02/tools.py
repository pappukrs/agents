"""@tool decorator + registry: turn plain Python functions into LLM tools.

The schema the model sees is built from the function's type hints and docstring,
so a tool is written once and can't drift out of sync with its declaration.
"""

from __future__ import annotations

import inspect
import json
import types
import typing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Literal

MAX_RESULT_CHARS = 4000
DEFAULT_TIMEOUT_S = 10.0

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    timeout_s: float = DEFAULT_TIMEOUT_S

    def declaration(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


REGISTRY: dict[str, Tool] = {}


def _json_type(annotation) -> dict:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is Literal:
        return {"type": _PY_TO_JSON[type(args[0])], "enum": list(args)}
    if origin in (list, tuple):
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin in (typing.Union, types.UnionType):  # Optional[x] / x | None
        non_none = [a for a in args if a is not type(None)]
        return _json_type(non_none[0])
    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}
    raise TypeError(f"Unsupported tool parameter type: {annotation!r}")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Summary = text before 'Args:'. Each 'name: text' line under 'Args:' describes a parameter."""
    summary, _, args_block = inspect.cleandoc(doc or "").partition("Args:")
    params = {}
    for line in args_block.splitlines():
        name, sep, text = line.strip().partition(":")
        if sep and name.isidentifier():
            params[name] = text.strip()
    return " ".join(summary.split()), params


def build_schema(fn: Callable) -> tuple[str, dict]:
    description, param_docs = _parse_docstring(fn.__doc__)
    hints = typing.get_type_hints(fn)
    properties, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        if name not in hints:
            raise TypeError(f"{fn.__name__}: parameter '{name}' needs a type hint")
        prop = _json_type(hints[name])
        if name in param_docs:
            prop["description"] = param_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default
        properties[name] = prop
    return description, {"type": "object", "properties": properties, "required": required}


def tool(fn: Callable | None = None, *, name: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S):
    """Register a function as a tool. Use as @tool or @tool(timeout_s=30)."""

    def register(f: Callable) -> Callable:
        description, parameters = build_schema(f)
        if not description:
            raise ValueError(f"{f.__name__}: the docstring is the tool description, write one")
        t = Tool(name or f.__name__, description, parameters, f, timeout_s)
        REGISTRY[t.name] = t
        return f

    return register(fn) if fn else register


def declarations(names: list[str] | None = None) -> list[dict]:
    return [t.declaration() for n, t in REGISTRY.items() if names is None or n in names]


_pool = ThreadPoolExecutor(max_workers=4)


def run_tool(name: str, args: dict | None) -> dict:
    """Execute a tool call from the model. Never raises: every failure becomes an observation
    the model can read and react to ({"error": ...}), so one bad call doesn't kill the run."""
    t = REGISTRY.get(name)
    if t is None:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(REGISTRY)}"}
    args = dict(args or {})
    unknown = set(args) - set(t.parameters["properties"])
    missing = [p for p in t.parameters["required"] if p not in args]
    if unknown or missing:
        return {"error": f"Bad arguments for {name}: unknown={sorted(unknown)} missing={missing}"}

    future = _pool.submit(t.fn, **args)
    try:
        value = future.result(timeout=t.timeout_s)
    except FutureTimeout:
        # The thread keeps running in the background; we just stop waiting for it.
        return {"error": f"{name} timed out after {t.timeout_s}s"}
    except Exception as e:  # noqa: BLE001 - tool errors are data for the model
        return {"error": f"{type(e).__name__}: {e}"}
    return {"result": truncate(value)}


def truncate(value: Any, limit: int = MAX_RESULT_CHARS) -> Any:
    """Cap what goes back into the context window. Big tool results are what overflow it."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
