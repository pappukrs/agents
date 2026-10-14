"""Offline tests: a fake client stands in for Gemini, so these run without an API key."""

from types import SimpleNamespace

import pytest
from google.genai import errors

import llm
from schemas import Ticket

GOOD = (
    '{"category": "billing", "priority": "high", "sentiment": "negative", "language": "en",'
    ' "summary": "Charged twice.", "entities": [{"type": "order_id", "value": "A-1"}], "needs_human": false}'
)


class FakeClient:
    """Returns queued responses; an Exception in the queue is raised instead."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.prompts = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=0)
        return SimpleNamespace(text=item, usage_metadata=usage)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def use(*responses) -> FakeClient:
    client = FakeClient(*responses)
    llm.set_client(client)
    return client


def test_structured_valid_first_try():
    use(GOOD)
    ticket, res = llm.generate_structured("x", Ticket)
    assert ticket.category == "billing"
    assert res.input_tokens == 10


def test_invalid_json_is_repaired_with_error_feedback():
    client = use('{"category": "billing"}', GOOD)
    ticket, _ = llm.generate_structured("x", Ticket)
    assert ticket.priority == "high"
    assert "Validation errors" in client.prompts[1]


def test_gives_up_after_max_repairs():
    use("not json", "still not json")
    with pytest.raises(llm.LLMError, match="failed validation"):
        llm.generate_structured("x", Ticket, max_repairs=1)


def test_retries_rate_limit_then_succeeds():
    use(errors.APIError(429, {"error": {"message": "slow down"}}), "hello")
    res = llm.generate("x")
    assert res.text == "hello"
    assert res.attempts == 2


def test_non_retryable_error_fails_fast():
    client = use(errors.APIError(400, {"error": {"message": "bad request"}}), "never")
    with pytest.raises(llm.LLMError, match="400"):
        llm.generate("x")
    assert len(client.queue) == 1
