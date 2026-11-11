"""Vector store guarantees the M5 contract tests do not check. Every test uses its own temp folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models import Chunk
from app.vectordb.store import INDEX_FILE, get_vector_store
from tests.fakes import make_chunk


def _chunk(cid: str, doc_id: str = "d1", **metadata: object) -> Chunk:
    return Chunk(id=cid, doc_id=doc_id, text=f"text {cid}", source=f"{doc_id}.md", position=0, metadata=metadata)


def test_chunk_fields_and_metadata_survive_reopen(tmp_path: Path) -> None:
    chunk = _chunk("d1:0", page=3, tags=["hr", "leave"], nested={"a": 1.5})
    get_vector_store(tmp_path).add([chunk], [[1.0, 0.0, 0.0]])
    hit = get_vector_store(tmp_path).search([1.0, 0.0, 0.0], k=1)[0]
    assert hit.chunk == chunk
    assert hit.score == pytest.approx(1.0)


def test_duplicate_ids_in_one_batch_keep_the_last(tmp_path: Path) -> None:
    store = get_vector_store(tmp_path)
    store.add([make_chunk("first", position=0), make_chunk("second", position=0)], [[1.0, 0.0], [0.0, 1.0]])
    assert store.count() == 1
    hit = store.search([0.0, 1.0], k=1)[0]
    assert hit.chunk.text == "second" and hit.score == pytest.approx(1.0)


def test_delete_document_persists_and_counts_only_that_document(tmp_path: Path) -> None:
    store = get_vector_store(tmp_path)
    store.add([_chunk("a:0", "a"), _chunk("a:1", "a"), _chunk("b:0", "b")], [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert store.delete_document("a") == 2
    assert store.delete_document("missing") == 0
    reopened = get_vector_store(tmp_path)
    assert reopened.count() == 1
    assert [h.chunk.id for h in reopened.search([1.0, 0.0], k=5)] == ["b:0"]


@pytest.mark.parametrize(
    ("chunks", "vectors"),
    [
        ([_chunk("x")], []),  # length mismatch
        ([_chunk("x"), _chunk("y")], [[1.0, 0.0], [1.0]]),  # ragged
        ([_chunk("x")], [[float("nan"), 1.0]]),  # not finite
        ([_chunk("x")], [[]]),  # empty vector
    ],
)
def test_invalid_embeddings_are_rejected(tmp_path: Path, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        get_vector_store(tmp_path).add(chunks, vectors)


def test_dimension_must_match_the_store(tmp_path: Path) -> None:
    store = get_vector_store(tmp_path)
    store.add([_chunk("x")], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        store.add([_chunk("y")], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0], k=1)


def test_failed_add_leaves_store_unchanged(tmp_path: Path) -> None:
    store = get_vector_store(tmp_path)
    store.add([_chunk("x")], [[1.0, 0.0]])
    with pytest.raises(ValueError):  # metadata that cannot be saved as JSON
        store.add([_chunk("y", bad=object())], [[0.0, 1.0]])
    assert store.count() == 1
    assert get_vector_store(tmp_path).count() == 1
    assert len(list(tmp_path.glob("vectors-*.npy"))) == 1, "a failed add must not leave files behind"


def test_empty_cases_and_k(tmp_path: Path) -> None:
    folder = tmp_path / "not-created-yet"
    store = get_vector_store(folder)
    assert store.count() == 0 and store.search([1.0], k=3) == []
    store.add([], [])
    assert not folder.exists(), "opening or a no-op add must not create anything"
    store.add([_chunk("x")], [[1.0, 0.0]])
    assert store.search([1.0, 0.0], k=0) == []


def test_only_the_current_vector_file_is_kept(tmp_path: Path) -> None:
    store = get_vector_store(tmp_path)
    for i in range(3):
        store.add([_chunk(f"c{i}")], [[1.0, float(i)]])
    assert len(list(tmp_path.glob("vectors-*.npy"))) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_bad_locations_and_corrupt_index_raise_value_error(tmp_path: Path) -> None:
    a_file = tmp_path / "file.txt"
    a_file.write_text("x")
    with pytest.raises(ValueError):
        get_vector_store(a_file)
    (tmp_path / INDEX_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        get_vector_store(tmp_path)
