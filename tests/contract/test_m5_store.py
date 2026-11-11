"""Milestone 5 — Vector store (tested with simple hand-made vectors)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.vectordb.store import get_vector_store
from tests.fakes import HashEmbedder, make_chunk

pytestmark = pytest.mark.contract
TEXTS = ["annual leave days", "expense receipts", "on-call stipend", "object storage price"]


def _fill(path: Path):  # type: ignore[no-untyped-def]
    store = get_vector_store(persist_dir=path)
    chunks = [make_chunk(t, position=i) for i, t in enumerate(TEXTS)]
    store.add(chunks, HashEmbedder().embed_documents(TEXTS))
    return store, chunks


def test_add_search_count(tmp_path: Path) -> None:
    store, chunks = _fill(tmp_path)
    assert store.count() == len(TEXTS)
    hits = store.search(HashEmbedder().embed_query("expense receipts"), k=2)
    assert len(hits) == 2
    assert hits[0].chunk.id == chunks[1].id
    assert hits[0].score >= hits[1].score


def test_upsert_persist_delete(tmp_path: Path) -> None:
    store, chunks = _fill(tmp_path)
    store.add(chunks[:1], HashEmbedder().embed_documents(TEXTS[:1]))
    assert store.count() == len(TEXTS), "re-adding the same id must not duplicate"
    reopened = get_vector_store(persist_dir=tmp_path)
    assert reopened.count() == len(TEXTS), "data must survive reopening"
    assert reopened.delete_document("doc-fake") == len(TEXTS)
    assert reopened.count() == 0
    assert reopened.search(HashEmbedder().embed_query("anything"), k=3) == []
