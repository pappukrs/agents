"""The RAG agent end to end with a scripted model + fake embedder: citations, refusals, grading."""

import llm
import rag
from ask import ask, check_citations, cited_ids, grade, is_refusal
from conftest import ScriptedClient, call


def use(*turns):
    client = ScriptedClient(*turns)
    llm.set_client(client)
    return client


def test_cited_ids_handles_lists_and_ignores_other_brackets():
    text = "30 days [returns.md#0]. Also [a.md#1, b/c.md#2] and [see above]."
    assert cited_ids(text) == ["returns.md#0", "a.md#1", "b/c.md#2"]


def test_answer_is_cited_from_retrieved_chunks(index):
    client = use([call("search_docs", query="return items days")], "Within 30 days [returns.md#0].")
    r = ask("How many days do I have to return an item?", verbose=False)
    assert check_citations(r.answer, r.trace) == {"cited": ["returns.md#0"], "made_up": [], "ok": True}
    sent = client.requests[1][2].parts[0].function_response.response["result"]
    assert sent["hits"][0]["id"] == "returns.md#0"


def test_made_up_citation_is_caught(index):
    use([call("search_docs", query="return items days")], "30 days [returns.md#0] [pricing.md#3].")
    r = ask("q", verbose=False)
    assert check_citations(r.answer, r.trace)["made_up"] == ["pricing.md#3"]
    ok, why = grade({"answerable": True, "expect": ["30"]}, r)
    assert not ok and "made-up" in why


def test_no_hits_then_refusal_passes_for_unanswerable(index):
    client = use([call("search_docs", query="CEO salary")], "The documents don't cover this. Salaries aren't mentioned.")
    r = ask("What is the CEO's salary?", verbose=False)
    assert client.requests[1][2].parts[0].function_response.response["result"]["hits"] == []
    assert grade({"answerable": False}, r) == (True, "refused")


def test_rephrased_second_search_is_allowed(index):
    use([call("search_docs", query="CEO salary")], [call("search_docs", query="express delivery cost")],
        "149 rupees [shipping.md#0].")
    r = ask("q", verbose=False)
    assert r.trace.totals()["tool_calls"] == 2
    assert grade({"answerable": True, "expect": ["149"], "source": "shipping.md"}, r)[0]


def test_grading_rules(index):
    use([call("search_docs", query="return items days")], "Within 30 days.")
    r = ask("q", verbose=False)
    assert grade({"answerable": True, "expect": ["30"]}, r) == (False, "no citations")
    assert grade({"answerable": False}, r) == (False, "should have refused")


def test_refusal_detection_accepts_curly_apostrophe():
    assert is_refusal("The documents don’t cover this.")
    assert not is_refusal("You can return it within 30 days.")


def test_embed_batches_and_passes_task_type():
    seen = []

    class FakeEmbedClient:
        models = None

        def embed_content(self, model, contents, config):
            seen.append((len(contents), config.task_type, config.output_dimensionality))
            return type("R", (), {"embeddings": [type("E", (), {"values": [0.1, 0.2]}) for _ in contents]})

    c = FakeEmbedClient()
    c.models = c
    llm.set_client(c)
    vecs = llm.embed([f"t{i}" for i in range(250)], task_type="RETRIEVAL_QUERY")
    assert len(vecs) == 250
    assert seen == [(100, "RETRIEVAL_QUERY", 768), (100, "RETRIEVAL_QUERY", 768), (50, "RETRIEVAL_QUERY", 768)]
