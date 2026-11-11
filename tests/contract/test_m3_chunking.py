"""Milestone 3 — Chunking rules (work for ANY chunking strategy)."""

from __future__ import annotations

import re

import pytest

from app.models import Chunk, Document
from app.vectordb.chunking import get_chunker
from tests.conftest import SAMPLE_DOCS

pytestmark = pytest.mark.contract
SIZE = 300


def _docs() -> list[Document]:
    return [
        Document(id=f"doc-{p.stem}", text=p.read_text(encoding="utf-8"), source=p.name)
        for p in sorted(SAMPLE_DOCS.glob("*"))
        if p.suffix in {".md", ".txt"}
    ]


def _words(text: str) -> set[str]:
    return set(re.findall(r"\S+", text))


def test_empty_document_gives_no_chunks() -> None:
    assert get_chunker(chunk_size=SIZE, chunk_overlap=0).chunk(Document(id="e", text="  \n ", source="e.md")) == []


@pytest.mark.parametrize("doc", _docs(), ids=lambda d: d.source)
def test_chunk_rules(doc: Document) -> None:
    chunks = get_chunker(chunk_size=SIZE, chunk_overlap=50).chunk(doc)
    assert chunks and all(isinstance(c, Chunk) for c in chunks)
    assert all(c.text.strip() for c in chunks), "no empty chunks"
    assert all(len(c.text) <= SIZE for c in chunks), "respect chunk_size"
    assert [c.position for c in chunks] == list(range(len(chunks))), "positions 0..n-1"
    assert len({c.id for c in chunks}) == len(chunks), "unique ids"
    assert all(c.doc_id == doc.id and c.source == doc.source for c in chunks)
    lost = _words(doc.text) - set().union(*(_words(c.text) for c in chunks))
    assert not lost, f"words lost: {sorted(lost)[:5]}"


def test_deterministic_and_size_matters() -> None:
    doc = _docs()[0]
    first = get_chunker(chunk_size=SIZE, chunk_overlap=0).chunk(doc)
    again = get_chunker(chunk_size=SIZE, chunk_overlap=0).chunk(doc)
    assert [c.text for c in first] == [c.text for c in again]
    big = get_chunker(chunk_size=SIZE * 3, chunk_overlap=0).chunk(doc)
    assert len(big) < len(first)
