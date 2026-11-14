"""Offline tests: a fake embedder and fake LLM replace the real models, so these
run in seconds with no downloads and no API key.   Run:  pytest rag/tests
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carbonscope_rag.chunker import split_documents  # noqa: E402
from carbonscope_rag.loader import load_all_pdfs  # noqa: E402
from carbonscope_rag.pipeline import NO_ANSWER, RAGPipeline  # noqa: E402
from carbonscope_rag.retriever import Retriever  # noqa: E402
from carbonscope_rag.vector_store import VectorStore  # noqa: E402


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors."""

    DIM = 128

    def _vec(self, text):
        v = np.zeros(self.DIM, dtype="float32")
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts, show_progress=False):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return type("R", (), {"content": "fake answer [1]"})()

    def stream(self, messages):
        for t in ["fake ", "answer"]:
            yield type("P", (), {"content": t})()


def _docs():
    return [
        Document(page_content="Project registration happens after validation by the ACVA.",
                 metadata={"source_file": "a.pdf", "page": 1, "total_pages": 2}),
        Document(page_content="Monitoring of project implementation requires data collection.",
                 metadata={"source_file": "a.pdf", "page": 2, "total_pages": 2}),
    ]


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_directory=tmp_path / "vs", collection_name="test")


def _ingest(store, embedder):
    chunks = split_documents(_docs())
    store.upsert(chunks, embedder.embed_documents([c.page_content for c in chunks]))
    return chunks


def test_chunk_ids_are_stable():
    ids1 = [c.metadata["chunk_id"] for c in split_documents(_docs())]
    ids2 = [c.metadata["chunk_id"] for c in split_documents(_docs())]
    assert ids1 == ids2 and len(set(ids1)) == len(ids1)


def test_reingest_creates_no_duplicates(store):
    emb = FakeEmbedder()
    _ingest(store, emb)
    n = store.count()
    _ingest(store, emb)
    assert store.count() == n


def test_delete_source_removes_chunks(store):
    _ingest(store, FakeEmbedder())
    store.delete_source("a.pdf")
    assert store.count() == 0


def test_retrieval_finds_relevant_chunk(store):
    emb = FakeEmbedder()
    _ingest(store, emb)
    hits = Retriever(store, emb).retrieve("project registration validation", top_k=2, min_score=0.1)
    assert hits and hits[0]["metadata"]["page"] == 1


def test_irrelevant_question_skips_llm(store):
    emb = FakeEmbedder()
    _ingest(store, emb)
    llm = FakeLLM()
    rag = RAGPipeline(retriever=Retriever(store, emb), llm=llm)
    result = rag.query("zzz qqq xxx", min_score=0.5)
    assert result["answer"] == NO_ANSWER and result["grounded"] is False and llm.calls == 0


def test_query_returns_answer_and_sources(store):
    emb = FakeEmbedder()
    _ingest(store, emb)
    rag = RAGPipeline(retriever=Retriever(store, emb), llm=FakeLLM())
    result = rag.query("project registration validation", min_score=0.1)
    assert result["grounded"] and result["sources"][0]["n"] == 1
    assert result["sources"][0]["page"] == 1


def test_stream_event_order(store):
    emb = FakeEmbedder()
    _ingest(store, emb)
    rag = RAGPipeline(retriever=Retriever(store, emb), llm=FakeLLM())
    events = list(rag.stream("project registration validation", min_score=0.1))
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "fake answer"


def test_empty_question_rejected(store):
    rag = RAGPipeline(retriever=Retriever(store, FakeEmbedder()), llm=FakeLLM())
    with pytest.raises(ValueError):
        rag.query("   ")


def test_strict_loader_raises_on_bad_pdf(tmp_path):
    """A file containing garbage bytes named bad.pdf must:
    - raise when strict=True
    - return an empty list when strict=False (default)
    """
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"\x00\x01\x02\x03NOT_A_PDF")

    with pytest.raises(Exception):
        load_all_pdfs(tmp_path, strict=True)

    result = load_all_pdfs(tmp_path, strict=False)
    assert result == []

