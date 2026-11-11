"""MILESTONE 5 — Vector store.

Store chunk vectors on disk and find the nearest ones to a query vector.

Use any backend: Chroma, FAISS, LanceDB, Qdrant (local mode) ... or your own
NumPy + JSON file. Add the package with `uv add <pkg>`. Data must live under
`persist_dir` (default: settings.vector_dir) — never a hard-coded path.

Rules the evaluator checks:
  - `search(v, k)` returns at most k hits, best first (scores descending).
  - Searching with a stored chunk's own vector returns that chunk first.
  - `add` with an existing chunk id REPLACES it (no duplicates).
  - A new store opened on the same `persist_dir` sees the same data.
  - `delete_document(doc_id)` removes all chunks of that document.
  - Searching an empty store returns [].
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import IO, Protocol

import numpy as np
from numpy.typing import NDArray

from app.config import get_settings
from app.models import Chunk, RetrievedChunk

INDEX_FILE = "index.json"
FORMAT_VERSION = 1
Matrix = NDArray[np.float32]


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """IN: chunks and their vectors (same length and order). Upsert by chunk.id."""
        ...

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        """IN: a query vector and k.  OUT: up to k RetrievedChunk, highest score first."""
        ...

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk whose doc_id matches. OUT: how many were removed."""
        ...

    def count(self) -> int:
        """How many chunks are stored."""
        ...


class NumpyVectorStore:
    """Cosine-similarity store: a float32 NumPy matrix plus a JSON list of chunks, both under `persist_dir`.

    Layout: `index.json` (chunks in row order, dimension, name of the vector file) and
    `vectors-<id>.npy`. A save writes a new vector file, then atomically replaces the index
    that points to it, so a crash never leaves the two out of step. Every change is
    written to disk before it is applied in memory.
    """

    def __init__(self, persist_dir: Path) -> None:
        if persist_dir.exists() and not persist_dir.is_dir():
            raise ValueError(f"Vector store path is not a directory: {persist_dir}")
        self.persist_dir = persist_dir
        self._lock = threading.Lock()
        self._chunks, self._vectors = self._load()

    # ------------------------------------------------------------------ API
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        new_vectors = _normalise(_as_matrix(embeddings))
        with self._lock:
            dimension = new_vectors.shape[1]
            if self._chunks and dimension != self._vectors.shape[1]:
                raise ValueError(f"Embedding dimension {dimension} != store dimension {self._vectors.shape[1]}")
            stored_chunks, stored_rows = list(self._chunks), list(self._vectors)
            rows = {chunk.id: row for row, chunk in enumerate(stored_chunks)}
            for chunk, vector in zip(chunks, new_vectors, strict=True):
                row = rows.setdefault(chunk.id, len(stored_chunks))  # upsert by id; last one wins
                if row == len(stored_chunks):
                    stored_chunks.append(chunk)
                    stored_rows.append(vector)
                else:
                    stored_chunks[row] = chunk
                    stored_rows[row] = vector
            self._commit(stored_chunks, np.vstack(stored_rows))

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        with self._lock:
            if k <= 0 or not self._chunks:
                return []
            query = _normalise(_as_matrix([query_embedding]))[0]
            if query.shape[0] != self._vectors.shape[1]:
                raise ValueError(f"Query dimension {query.shape[0]} != store dimension {self._vectors.shape[1]}")
            scores = self._vectors @ query
            best = np.argsort(-scores, kind="stable")[:k]  # stable: ties keep insertion order
            return [RetrievedChunk(chunk=self._chunks[i].model_copy(deep=True), score=float(scores[i])) for i in best]

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            keep = [row for row, chunk in enumerate(self._chunks) if chunk.doc_id != doc_id]
            removed = len(self._chunks) - len(keep)
            if removed:
                self._commit([self._chunks[row] for row in keep], self._vectors[keep])
            return removed

    def count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------ persistence
    def _commit(self, chunks: list[Chunk], vectors: Matrix) -> None:
        """Write the new state to disk, then make it the in-memory state."""
        vectors_file = f"vectors-{uuid.uuid4().hex}.npy" if chunks else None
        index = {
            "version": FORMAT_VERSION,
            "dimension": int(vectors.shape[1]) if chunks else None,
            "vectors_file": vectors_file,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        payload = json.dumps(index, ensure_ascii=False, indent=1).encode("utf-8")  # fails before touching disk
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if vectors_file:
            _atomic_write(self.persist_dir / vectors_file, lambda fh: np.save(fh, vectors, allow_pickle=False))
        _atomic_write(self.persist_dir / INDEX_FILE, lambda fh: fh.write(payload))
        self._chunks, self._vectors = chunks, vectors
        for old in self.persist_dir.glob("vectors-*.npy"):
            if old.name != vectors_file:
                old.unlink(missing_ok=True)

    def _load(self) -> tuple[list[Chunk], Matrix]:
        index_path = self.persist_dir / INDEX_FILE
        if not index_path.exists():
            return [], _empty()
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            chunks = [Chunk.model_validate(record) for record in index["chunks"]]
            if not chunks:
                return [], _empty()
            vectors = np.load(self.persist_dir / index["vectors_file"], allow_pickle=False).astype(np.float32)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Vector store in {self.persist_dir} is unreadable: {exc}") from exc
        if index.get("version") != FORMAT_VERSION or vectors.shape != (len(chunks), index["dimension"]):
            raise ValueError(f"Vector store in {self.persist_dir} is inconsistent (index and vectors do not match)")
        return chunks, vectors


def _empty() -> Matrix:
    return np.zeros((0, 0), dtype=np.float32)


def _as_matrix(embeddings: list[list[float]]) -> Matrix:
    try:
        matrix = np.asarray(embeddings, dtype=np.float32)
    except ValueError as exc:  # ragged lists
        raise ValueError("All embeddings must have the same dimension") from exc
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Embeddings must be non-empty lists of numbers")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    return matrix


def _normalise(matrix: Matrix) -> Matrix:
    """Unit-length rows, so a dot product is cosine similarity. All-zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _atomic_write(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Write via a temp file in the same folder, then replace `path` in one step."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def get_vector_store(persist_dir: str | Path | None = None) -> VectorStore:
    """Open the vector store saved in `persist_dir` (default: settings.vector_dir).

    A missing folder gives an empty store; nothing is created until the first write.
    Raises ValueError if the path is not a folder or the saved store is unreadable.
    """
    path = Path(persist_dir) if persist_dir is not None else get_settings().vector_dir
    return NumpyVectorStore(path)
