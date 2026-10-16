"""Tool layer tests: schema generation, sandboxing, errors as observations. No LLM involved."""

import time
from typing import Literal

import pytest

import builtin_tools  # noqa: F401
from tools import REGISTRY, build_schema, run_tool, tool, truncate


def test_schema_from_type_hints_and_docstring():
    def f(city: str, days: int = 3, unit: Literal["c", "f"] = "c", tags: list[str] | None = None):
        """Get the weather.

        Args:
            city: city name.
            days: how many days.
        """

    desc, schema = build_schema(f)
    assert desc == "Get the weather."
    props = schema["properties"]
    assert props["city"] == {"type": "string", "description": "city name."}
    assert props["days"]["type"] == "integer" and props["days"]["default"] == 3
    assert props["unit"]["enum"] == ["c", "f"]
    assert props["tags"]["type"] == "array"
    assert schema["required"] == ["city"]


def test_missing_type_hint_is_rejected():
    def f(x):
        """Doc."""

    with pytest.raises(TypeError, match="type hint"):
        build_schema(f)


def test_calculator():
    assert run_tool("calculator", {"expression": "(1200 + 950) / 2"}) == {"result": 1075.0}
    assert run_tool("calculator", {"expression": "sqrt(16) * 3"}) == {"result": 12.0}


def test_calculator_refuses_code():
    out = run_tool("calculator", {"expression": "__import__('os').system('ls')"})
    assert "error" in out


def test_unknown_tool_and_bad_args_become_observations():
    assert "Unknown tool" in run_tool("delete_everything", {})["error"]
    assert "missing=['expression']" in run_tool("calculator", {})["error"]
    assert "unknown=['foo']" in run_tool("calculator", {"expression": "1", "foo": 2})["error"]


def test_read_file_cannot_escape_workspace():
    out = run_tool("read_file", {"path": "../llm.py"})
    assert "outside the workspace" in out["error"]
    out = run_tool("read_file", {"path": "/etc/passwd"})
    assert "error" in out


def test_glob_cannot_escape_workspace():
    files = run_tool("list_files", {"pattern": "../*"})["result"]
    assert files == []


def test_search_finds_refunds():
    hits = run_tool("search_files", {"query": "refund"})["result"]
    assert any(h["file"] == "docs/q3-review.md" for h in hits)


def test_analyze_csv_group_by():
    out = run_tool("analyze_csv", {"path": "sales.csv", "operation": "group_by", "group_by": "region",
                                   "column": "revenue", "agg": "mean"})
    assert out["result"]["result"] == {"East": 1182.0, "North": 1217.0, "South": 1150.0, "West": 1457.0}


def test_analyze_csv_bad_column_explains_itself():
    out = run_tool("analyze_csv", {"path": "sales.csv", "operation": "group_by", "group_by": "area", "column": "revenue"})
    assert "Columns:" in out["error"]


def test_slow_tool_times_out():
    @tool(timeout_s=0.1)
    def slow_tool() -> str:
        """Sleeps."""
        time.sleep(1)
        return "done"

    try:
        assert "timed out" in run_tool("slow_tool", {})["error"]
    finally:
        REGISTRY.pop("slow_tool")


def test_truncate():
    assert truncate("short") == "short"
    out = truncate("x" * 5000, limit=100)
    assert out.startswith("x" * 100) and "truncated 4900 chars" in out
