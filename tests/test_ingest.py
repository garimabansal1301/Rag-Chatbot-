"""Ingestion guarantees the M6 contract tests do not check. Real chunker and NumPy store in temp dirs, fake embedder."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chat.engine import PlaceholderEngine
from app.config import Settings
from app.deps import Components, get_components
from app.ingest.pipeline import document_id, ingest_directory, ingest_file, load_document
from app.main import create_app
from app.vectordb.chunking import get_chunker
from app.vectordb.store import VectorStore, get_vector_store
from tests.conftest import SAMPLE_DOCS
from tests.fakes import HashEmbedder

LEAVE = SAMPLE_DOCS / "leave_policy.md"


class FailingEmbedder(HashEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model unavailable")


class ShortEmbedder(HashEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(texts)[:-1]


def _ingest(path: Path, store: VectorStore, embedder: HashEmbedder | None = None) -> int:
    chunker = get_chunker("fixed", chunk_size=300, chunk_overlap=50)
    return ingest_file(path, chunker=chunker, embedder=embedder or HashEmbedder(), store=store)


# ---------------------------------------------------------------- loading
def test_load_strips_bom_and_accepts_upper_case_suffix(tmp_path: Path) -> None:
    path = tmp_path / "Notes.MD"
    path.write_bytes(b"\xef\xbb\xbf# Title\nbody")
    doc = load_document(path)
    assert doc.text == "# Title\nbody" and doc.source == "Notes.MD"
    assert doc.metadata["suffix"] == ".md" and len(doc.metadata["content_sha256"]) == 64


@pytest.mark.parametrize(("name", "data"), [("bad.txt", "caf\xe9".encode("latin-1")), ("bin.txt", b"PNG\x00\x01")])
def test_load_rejects_unreadable_content(tmp_path: Path, name: str, data: bytes) -> None:
    (tmp_path / name).write_bytes(data)
    with pytest.raises(ValueError, match=name):
        load_document(tmp_path / name)


def test_load_missing_file_raises_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.md")


def test_document_id_depends_on_file_name_only(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.md").write_text("one", encoding="utf-8")
    (tmp_path / "x.md").write_text("two", encoding="utf-8")
    assert load_document(tmp_path / "a" / "x.md").id == load_document(tmp_path / "x.md").id == document_id("x.md")
    assert document_id("x.md") != document_id("x.txt")


# ---------------------------------------------------------------- ingest_file
def test_ids_sources_positions_and_metadata_survive_in_the_store(tmp_path: Path) -> None:
    added = _ingest(LEAVE, get_vector_store(tmp_path / "v"))
    hits = get_vector_store(tmp_path / "v").search(HashEmbedder().embed_query("annual leave"), k=added)
    doc = load_document(LEAVE)
    assert sorted(h.chunk.position for h in hits) == list(range(added))
    for hit in hits:
        assert hit.chunk.id == f"{doc.id}:{hit.chunk.position}"
        assert hit.chunk.doc_id == doc.id and hit.chunk.source == "leave_policy.md"
        assert hit.chunk.metadata == doc.metadata


def test_reingesting_a_shorter_file_removes_stale_chunks(tmp_path: Path) -> None:
    path, store = tmp_path / "doc.md", get_vector_store(tmp_path / "v")
    path.write_text(LEAVE.read_text(encoding="utf-8"), encoding="utf-8")
    assert _ingest(path, store) > 1
    path.write_text("Only one short paragraph now.", encoding="utf-8")
    assert _ingest(path, store) == 1
    assert store.count() == 1


def test_emptied_file_removes_its_old_chunks(tmp_path: Path) -> None:
    path, store = tmp_path / "doc.md", get_vector_store(tmp_path / "v")
    path.write_text("Some text.", encoding="utf-8")
    _ingest(path, store)
    path.write_text("  \n", encoding="utf-8")
    assert _ingest(path, store) == 0
    assert store.count() == 0


@pytest.mark.parametrize(("embedder", "error"), [(FailingEmbedder(), RuntimeError), (ShortEmbedder(), ValueError)])
def test_embedding_failure_keeps_the_previous_copy(tmp_path: Path, embedder: HashEmbedder, error: type) -> None:
    store = get_vector_store(tmp_path / "v")
    before = _ingest(LEAVE, store)
    with pytest.raises(error):
        _ingest(LEAVE, store, embedder)
    assert store.count() == before
    assert get_vector_store(tmp_path / "v").count() == before


# ---------------------------------------------------------------- ingest_directory
def test_directory_skips_bad_and_unsupported_files(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    (folder / "nested").mkdir(parents=True)
    (folder / "good.md").write_text(LEAVE.read_text(encoding="utf-8"), encoding="utf-8")
    (folder / "bad.txt").write_bytes("caf\xe9".encode("latin-1"))
    (folder / "image.png").write_bytes(b"\x89PNG")
    (folder / "nested" / "deep.md").write_text("not read: only the top level is ingested", encoding="utf-8")
    store = get_vector_store(tmp_path / "v")
    chunker = get_chunker("fixed", chunk_size=300, chunk_overlap=50)
    total = ingest_directory(folder, chunker=chunker, embedder=HashEmbedder(), store=store)
    assert total == store.count() == _ingest(folder / "good.md", get_vector_store(tmp_path / "v2"))


# ---------------------------------------------------------------- upload endpoint
@pytest.fixture()
def indexing_client(settings: Settings) -> Iterator[tuple[TestClient, VectorStore]]:
    store = get_vector_store(settings.vector_dir)
    chunker = get_chunker("fixed", chunk_size=300, chunk_overlap=50)
    parts = Components(engine=PlaceholderEngine(), chunker=chunker, embedder=HashEmbedder(), store=store)
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: parts
    with TestClient(app) as client:
        yield client, store


def test_upload_indexes_and_reupload_does_not_duplicate(indexing_client: tuple[TestClient, VectorStore]) -> None:
    client, store = indexing_client
    first = client.post("/api/documents", files={"file": ("leave.md", LEAVE.read_bytes(), "text/markdown")}).json()
    assert first["indexed"] and first["chunks_indexed"] == store.count() > 0
    client.post("/api/documents", files={"file": ("leave.md", LEAVE.read_bytes(), "text/markdown")})
    assert store.count() == first["chunks_indexed"]


def test_upload_of_unreadable_text_is_rejected(
    indexing_client: tuple[TestClient, VectorStore], settings: Settings
) -> None:
    client, store = indexing_client
    response = client.post("/api/documents", files={"file": ("menu.txt", "caf\xe9".encode("latin-1"), "text/plain")})
    assert response.status_code == 400 and "UTF-8" in response.json()["detail"]
    assert store.count() == 0 and not (settings.documents_dir / "menu.txt").exists()
