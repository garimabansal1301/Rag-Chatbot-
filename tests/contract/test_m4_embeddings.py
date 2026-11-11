"""Milestone 4 — Embeddings (uses YOUR configured model; may need an API key in .env)."""

from __future__ import annotations

import math

import pytest

from app.vectordb.embeddings import get_embedder

pytestmark = pytest.mark.contract


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_shapes() -> None:
    embedder = get_embedder()
    vectors = embedder.embed_documents(["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(v) == embedder.dimension for v in vectors)
    assert embedder.embed_documents([]) == []


def test_query_matches_document_and_meaning() -> None:
    embedder = get_embedder()
    text = "How many days of annual leave do I get?"
    assert _cos(embedder.embed_query(text), embedder.embed_documents([text])[0]) > 0.95
    related = _cos(embedder.embed_query("vacation days allowance"), embedder.embed_query(text))
    unrelated = _cos(embedder.embed_query("object storage encryption"), embedder.embed_query(text))
    assert related > unrelated
