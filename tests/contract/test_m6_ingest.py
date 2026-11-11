"""Milestone 6 — Ingestion (uses YOUR chunker + store, fake embedder)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest.pipeline import ingest_directory, ingest_file, load_document
from app.vectordb.chunking import get_chunker
from tests.conftest import SAMPLE_DOCS
from tests.fakes import HashEmbedder, ListStore

pytestmark = pytest.mark.contract


def test_load_document() -> None:
    doc = load_document(SAMPLE_DOCS / "leave_policy.md")
    assert doc.source == "leave_policy.md" and "annual leave" in doc.text.lower()
    assert load_document(SAMPLE_DOCS / "leave_policy.md").id == doc.id, "id must be stable"


def test_rejects_unsupported(tmp_path: Path) -> None:
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    with pytest.raises(ValueError):
        load_document(bad)


def test_ingest_is_idempotent() -> None:
    store = ListStore()
    parts = {"chunker": get_chunker(chunk_size=300, chunk_overlap=0), "embedder": HashEmbedder(), "store": store}
    added = ingest_file(SAMPLE_DOCS / "leave_policy.md", **parts)
    assert added > 0 and store.count() == added
    ingest_file(SAMPLE_DOCS / "leave_policy.md", **parts)
    assert store.count() == added, "ingesting twice must not duplicate"
    total = ingest_directory(SAMPLE_DOCS, **parts)
    assert total == store.count()
