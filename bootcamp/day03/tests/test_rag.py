"""Chunking, cosine search, the threshold and the search_docs tool, all offline."""

import numpy as np
import pytest

import rag
from tools import MAX_RESULT_CHARS, REGISTRY, run_tool


def test_chunks_overlap_and_cover_the_text():
    text = "".join(str(i % 10) for i in range(2000))
    chunks = rag.chunk_text(text, size=800, overlap=150)
    assert len(chunks) == 3
    assert all(len(c) <= 800 for c in chunks)
    assert chunks[0][-150:] == chunks[1][:150]
    assert chunks[-1].endswith(text[-50:])


def test_short_text_is_one_chunk_and_blank_is_none():
    assert rag.chunk_text("hello") == ["hello"]
    assert rag.chunk_text("   ") == []


def test_bad_overlap_is_rejected():
    with pytest.raises(ValueError):
        rag.chunk_text("abc", size=100, overlap=100)


def test_normalize_gives_unit_rows_and_survives_zero():
    m = rag.normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(m[0], [0.6, 0.8])
    assert np.allclose(m[1], [0, 0])


def test_ingest_ids_and_saved_index(index, tmp_path):
    assert index.ids == ["returns.md#0", "security.md#0", "shipping.md#0"]
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1)
    loaded = rag.Index.load(tmp_path / "index.npz")
    assert loaded.ids == index.ids and loaded.vectors.shape == index.vectors.shape


def test_missing_index_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="rag.py ingest"):
        rag.Index.load(tmp_path / "nope.npz")


def test_search_ranks_the_right_chunk_first(index):
    hits = rag.search("how many days to return items", k=3, min_score=0)
    assert hits[0].id == "returns.md#0"
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_threshold_drops_weak_matches(index):
    assert rag.search("CEO salary", k=3, min_score=0.3) == []
    assert len(rag.search("CEO salary", k=3, min_score=-1)) == 3  # top-k alone always returns k


def test_search_docs_returns_ids_and_scores(index):
    out = rag.search_docs("express delivery cost")
    assert out["hits"][0]["id"] == "shipping.md#0"
    assert set(out["hits"][0]) == {"id", "score", "text"}


def test_search_docs_empty_hits_carry_a_note(index):
    out = rag.search_docs("CEO salary")
    assert out["hits"] == [] and "No passage" in out["note"]


def test_search_docs_is_a_registered_tool_with_a_schema():
    t = REGISTRY["search_docs"]
    assert t.parameters["required"] == ["query"]
    assert "cite" in t.description


def test_big_hits_are_trimmed_not_cut_mid_json(monkeypatch):
    big = rag.Index(rag.normalize(np.ones((8, 4), dtype=np.float32)), [f"d.md#{i}" for i in range(8)], ["x" * 800] * 8)
    rag.set_index(big)
    monkeypatch.setattr(rag.llm, "embed", lambda texts, **_: [[1.0, 1.0, 1.0, 1.0]])
    try:
        out = run_tool("search_docs", {"query": "x", "k": 8})
    finally:
        rag.set_index(None)
    assert isinstance(out["result"], dict) and len(out["result"]["hits"]) == 8
    assert len(str(out["result"])) < MAX_RESULT_CHARS
