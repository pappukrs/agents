"""RAG over a folder of documents: chunk -> embed -> save once, then cosine search on every question.

    python rag.py ingest                       # docs/ -> data/index.npz
    python rag.py search "return window"       # see raw scores, no LLM chat
    python rag.py scores questions.jsonl       # best score per question: pick MIN_SCORE from this

The agent never calls search() directly: it calls the `search_docs` tool, registered on import.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import llm
from tools import MAX_RESULT_CHARS, tool

HERE = Path(__file__).parent
DOCS_DIR = HERE / "docs"
INDEX_PATH = HERE / "data" / "index.npz"
DOC_SUFFIXES = {".md", ".txt", ".pdf"}
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.6"))


@dataclass
class Hit:
    id: str  # "file.md#2": what the model cites
    score: float
    text: str


@dataclass
class Index:
    vectors: np.ndarray  # (n_chunks, dim) float32, every row length 1
    ids: list[str]
    texts: list[str]

    def save(self, path: Path = INDEX_PATH) -> None:
        path.parent.mkdir(exist_ok=True)
        np.savez(path, vectors=self.vectors, ids=np.array(self.ids), texts=np.array(self.texts))

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> Index:
        if not path.exists():
            raise FileNotFoundError(f"No index at {path}. Run: python rag.py ingest")
        data = np.load(path)
        return cls(data["vectors"], data["ids"].tolist(), data["texts"].tolist())


# ---- ingest ----

def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Fixed-size character windows; each one repeats the last `overlap` chars of the previous."""
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
    return [c for c in chunks if c.strip()]


def normalize(m: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so cosine similarity becomes a plain dot product."""
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def read_doc(p: Path) -> str:
    if p.suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("PDFs need pypdf: pip install pypdf") from e
        return "\n".join(page.extract_text() or "" for page in PdfReader(p).pages)
    return p.read_text(encoding="utf-8", errors="replace")


def ingest(folder: Path = DOCS_DIR, out: Path = INDEX_PATH, size: int = 800, overlap: int = 150) -> Index:
    files = sorted(p for p in folder.glob("**/*") if p.is_file() and p.suffix.lower() in DOC_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No {sorted(DOC_SUFFIXES)} files in {folder}")
    ids, texts, chars = [], [], 0
    for p in files:
        text = read_doc(p)
        chars += len(text)
        name = p.relative_to(folder).as_posix()
        for i, chunk in enumerate(chunk_text(text, size, overlap)):
            ids.append(f"{name}#{i}")
            texts.append(chunk)
    print(f"read {len(files)} files, {chars:,} chars -> {len(texts)} chunks (size={size}, overlap={overlap})")

    started = time.perf_counter()
    vectors = normalize(np.array(llm.embed(texts, task_type="RETRIEVAL_DOCUMENT"), dtype=np.float32))
    print(f"embedded {len(texts)} chunks in {time.perf_counter() - started:.1f}s")

    index = Index(vectors, ids, texts)
    index.save(out)
    print(f"saved {out}  vectors={vectors.shape} ids={len(ids)}")
    return index


# ---- search ----

_index: Index | None = None


def get_index() -> Index:
    """Load the index once per process; questions never re-embed the documents."""
    global _index
    if _index is None:
        _index = Index.load()
    return _index


def set_index(index: Index | None) -> None:
    """Swap in an in-memory index (tests), or None to reload from disk."""
    global _index
    _index = index


def search(query: str, k: int = 4, min_score: float | None = None, index: Index | None = None) -> list[Hit]:
    """Top-k chunks by cosine similarity, dropping any below `min_score`."""
    index = index or get_index()
    min_score = MIN_SCORE if min_score is None else min_score
    q = normalize(np.array(llm.embed([query], task_type="RETRIEVAL_QUERY")[0], dtype=np.float32))
    scores = index.vectors @ q  # every cosine at once: shape (n_chunks,)
    top = np.argsort(scores)[::-1][:k]
    return [Hit(index.ids[i], float(scores[i]), index.texts[i]) for i in top if scores[i] >= min_score]


@tool(timeout_s=30)
def search_docs(query: str, k: int = 4) -> dict:
    """Search the provided documents. Returns the most relevant passages with ids to cite.
    If 'hits' is empty, the documents do not answer the question: say so, don't guess.

    Args:
        query: what to look for, in your own words. Rephrase and search again if results are weak.
        k: how many passages to return (1-8).
    """
    hits = search(query, k=max(1, min(k, 8)))
    if not hits:
        return {"hits": [], "note": f"No passage scored above {MIN_SCORE}."}
    # Trim passages so the JSON stays under run_tool's cap and is never cut mid-way.
    room = (MAX_RESULT_CHARS - 600) // len(hits)
    return {"hits": [{"id": h.id, "score": round(h.score, 2), "text": h.text[:room]} for h in hits]}


# ---- CLI ----

def _scores(path: Path) -> None:
    """For threshold tuning: the best score for each question, answerable vs not."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        hits = search(row["question"], k=1, min_score=-1)
        kind = "answerable  " if row.get("answerable", True) else "UNANSWERABLE"
        best = hits[0] if hits else Hit("-", 0.0, "")
        print(f"{best.score:.3f}  {kind}  {best.id:<22} {row['question']}")
    print(f"\ncurrent MIN_SCORE = {MIN_SCORE}: put it between the lowest answerable and highest unanswerable score")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=150)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=6)
    p = sub.add_parser("scores")
    p.add_argument("file", type=Path)
    args = ap.parse_args()

    if args.cmd == "ingest":
        ingest(size=args.size, overlap=args.overlap)
    elif args.cmd == "search":
        for h in search(args.query, k=args.k, min_score=-1):
            mark = "keep" if h.score >= MIN_SCORE else "drop"
            print(f"{h.score:.3f} {mark}  {h.id:<22} {h.text[:90].replace(chr(10), ' ')}")
    else:
        _scores(args.file)


if __name__ == "__main__":
    main()
