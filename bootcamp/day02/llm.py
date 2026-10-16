"""Thin Gemini wrapper: one place for retries, timing, token counts and validated JSON.

Day 2 adds `chat()`: one turn of a multi-turn conversation with tool declarations,
returning the raw response so the agent loop can read function calls.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RETRYABLE_CODES = {429, 500, 502, 503, 504}

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing. Copy .env.example to .env and add your key.")
        _client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
    return _client


def set_client(client) -> None:
    """Swap in a fake client for tests."""
    global _client
    _client = client


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    attempts: int = 1


class LLMError(RuntimeError):
    pass


def generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_output_tokens: int = 2048,
    thinking_budget: int | None = None,
    response_schema: type[BaseModel] | None = None,
    max_retries: int = 3,
) -> LLMResult:
    """One single-turn LLM call."""
    model = model or DEFAULT_MODEL
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if thinking_budget is not None:
        config.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema

    res, latency_ms, attempts = _call(model, prompt, config, max_retries)
    return _result(res, model, latency_ms, attempts)


def _call(model: str, contents, config: types.GenerateContentConfig, max_retries: int):
    """Send one request. Retries transient API errors (429/5xx) with exponential backoff + jitter."""
    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            res = get_client().models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == max_retries:
                raise LLMError(f"Gemini API error {e.code}: {e}") from e
            time.sleep(min(2 ** attempt, 20) + random.random())
            continue
        return res, round((time.perf_counter() - started) * 1000), attempt
    raise LLMError("unreachable")


def _result(res, model: str, latency_ms: int, attempts: int) -> LLMResult:
    usage = res.usage_metadata
    return LLMResult(
        text=_text(res),
        model=model,
        latency_ms=latency_ms,
        input_tokens=(usage and usage.prompt_token_count) or 0,
        output_tokens=(usage and usage.candidates_token_count) or 0,
        thinking_tokens=(usage and usage.thoughts_token_count) or 0,
        attempts=attempts,
    )


def _text(res) -> str:
    # res.text warns when the reply also has function calls; read the text parts directly.
    try:
        parts = res.candidates[0].content.parts or []
        return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
    except (AttributeError, IndexError, TypeError):
        return getattr(res, "text", "") or ""


def chat(
    contents: list,
    *,
    tools: list[dict] | None = None,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 4096,
    thinking_budget: int | None = None,
    max_retries: int = 3,
):
    """One turn of a tool-using conversation.

    `contents` is the whole history (the model is stateless). `tools` are JSON-schema
    function declarations: {"name", "description", "parameters"}. Automatic function
    calling is off: the agent loop decides what runs. Returns (response, LLMResult).
    """
    model = model or DEFAULT_MODEL
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if tools:
        config.tools = [types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["parameters"])
            for t in tools
        ])]
    if thinking_budget is not None:
        config.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
    res, latency_ms, attempts = _call(model, contents, config, max_retries)
    return res, _result(res, model, latency_ms, attempts)


def generate_structured(
    prompt: str,
    schema: type[T],
    *,
    max_repairs: int = 1,
    temperature: float = 0.0,
    **kwargs,
) -> tuple[T, LLMResult]:
    """Ask for JSON matching `schema`, validate with Pydantic, and re-ask with the error if invalid."""
    current = prompt
    last_error: ValidationError | None = None
    for _ in range(max_repairs + 1):
        result = generate(current, response_schema=schema, temperature=temperature, **kwargs)
        try:
            return schema.model_validate_json(result.text), result
        except ValidationError as e:
            last_error = e
            current = (
                f"{prompt}\n\nYour previous answer was invalid.\n"
                f"Previous answer:\n{result.text}\n\nValidation errors:\n{e}\n\n"
                "Return corrected JSON only."
            )
    raise LLMError(f"Output failed validation after {max_repairs + 1} attempts: {last_error}")
