"""Agent loop tests with a scripted fake model: tool calls, error recovery, step cap."""

from types import SimpleNamespace

from google.genai import types

import builtin_tools  # noqa: F401
import llm
from loop import run_agent


def call(name, **args):
    return types.FunctionCall(name=name, args=args)


class ScriptedClient:
    """Each queued item is one model turn: a list of FunctionCalls, or a final text string."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append(list(contents))
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            parts = [types.Part.from_text(text=turn)]
        else:
            parts = [types.Part(function_call=c) for c in turn]
        content = types.Content(role="model", parts=parts)
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=10, thoughts_token_count=0)
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=content)],
            function_calls=[p.function_call for p in parts if p.function_call] or None,
            usage_metadata=usage,
        )


def use(*turns):
    client = ScriptedClient(*turns)
    llm.set_client(client)
    return client


def test_final_answer_without_tools():
    use("Hello!")
    r = run_agent("hi", verbose=False)
    assert r.answer == "Hello!" and r.stopped_because == "final"
    assert r.trace.totals()["tool_calls"] == 0


def test_tool_result_is_fed_back_to_the_model():
    client = use([call("calculator", expression="2 + 3")], "The answer is 5.")
    r = run_agent("what is 2+3", verbose=False)
    assert r.answer == "The answer is 5."
    # second request = question, model's call, our function response
    second = client.requests[1]
    assert len(second) == 3
    fr = second[2].parts[0].function_response
    assert fr.name == "calculator" and fr.response == {"result": 5}


def test_parallel_calls_answered_in_one_turn():
    client = use([call("calculator", expression="1+1"), call("search_files", query="refund")], "done")
    r = run_agent("q", verbose=False)
    assert r.trace.totals()["tool_calls"] == 2
    assert len(client.requests[1][2].parts) == 2


def test_tool_error_is_observation_and_model_recovers():
    client = use(
        [call("analyze_csv", path="sales.csv", operation="group_by", group_by="area", column="revenue")],
        [call("analyze_csv", path="sales.csv", operation="group_by", group_by="region", column="revenue")],
        "West is highest.",
    )
    r = run_agent("q", verbose=False)
    assert r.stopped_because == "final"
    totals = r.trace.totals()
    assert totals["tool_errors"] == 1 and totals["tool_calls"] == 2
    assert "Columns:" in client.requests[1][2].parts[0].function_response.response["error"]


def test_max_steps_stops_a_looping_agent():
    use(*[[call("list_files")] for _ in range(10)])
    r = run_agent("q", max_steps=3, verbose=False)
    assert r.stopped_because == "max_steps"
    assert r.trace.totals()["llm_calls"] == 3


def test_tokens_are_summed_across_steps():
    use([call("list_files")], "ok")
    r = run_agent("q", verbose=False)
    assert r.trace.totals()["input_tokens"] == 200
