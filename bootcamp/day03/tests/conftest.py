"""Shared fakes: an embedder that needs no API key, and a scripted model (same idea as Day 2)."""

import re
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import types

import llm
import rag

DIM = 256
STOP = {"the", "a", "an", "of", "to", "is", "in", "and", "or", "do", "i", "my", "you", "for", "it", "are", "be", "how", "what"}


def fake_embed(texts, task_type="RETRIEVAL_DOCUMENT", **_):
    """Bag of words hashed into DIM buckets: shared words -> similar vectors. Not normalised, like Gemini at 768."""
    out = []
    for t in texts:
        v = np.zeros(DIM)
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            if w not in STOP:
                v[zlib.crc32(w.encode()) % DIM] += 1
        out.append((v * 3).tolist())
    return out


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "returns.md").write_text("Items can be returned within 30 days of delivery in original packaging.")
    (folder / "shipping.md").write_text("Express delivery costs 149 rupees and takes 1 to 2 business days.")
    (folder / "security.md").write_text("Password reset links expire after 30 minutes. Turn on two-factor authentication.")
    return folder


@pytest.fixture
def index(docs, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "embed", fake_embed)
    monkeypatch.setattr(rag, "MIN_SCORE", 0.3)  # bag-of-words scores run lower than Gemini's
    idx = rag.ingest(docs, tmp_path / "index.npz")
    rag.set_index(idx)
    yield idx
    rag.set_index(None)


class ScriptedClient:
    """Each queued item is one model turn: a list of FunctionCalls, or a final text string."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append(list(contents))
        turn = self.turns.pop(0)
        parts = [types.Part.from_text(text=turn)] if isinstance(turn, str) else [types.Part(function_call=c) for c in turn]
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=10, thoughts_token_count=0)
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=types.Content(role="model", parts=parts))],
            function_calls=[p.function_call for p in parts if p.function_call] or None,
            usage_metadata=usage,
        )


def call(name, **args):
    return types.FunctionCall(name=name, args=args)
