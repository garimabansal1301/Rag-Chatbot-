"""MILESTONE 4 — Embeddings.

Turn text into vectors so similar meaning -> nearby vectors.

Pick any embedding model: a hosted API (OpenAI, Voyage, Cohere ...) or a local
one (sentence-transformers, fastembed ...). Add its package with `uv add <pkg>`
and read the model name from `EMBEDDING_MODEL` in `.env`.

Rules the evaluator checks:
  - `embed_documents(texts)` returns one vector per text, in the same order.
  - Every vector has length `dimension`.
  - `embed_documents([])` returns `[]`.
  - The same text always gives (almost) the same vector.
  - `embed_query(t)` is close to `embed_documents([t])[0]`.
  - Related sentences are closer (cosine) than unrelated ones.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from app.config import get_settings

if TYPE_CHECKING:
    from fastembed import TextEmbedding

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        """Length of every vector this embedder returns."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """IN: list of texts.  OUT: list of vectors (same length and order)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """IN: one search query.  OUT: one vector."""
        ...


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (no API key).

    The model is loaded on the first embedding call, not at construction, and is
    downloaded into `cache_dir` the first time it is needed on this machine.
    """

    def __init__(self, model_name: str, cache_dir: Path, batch_size: int = BATCH_SIZE) -> None:
        from fastembed import TextEmbedding

        supported = {m["model"] for m in TextEmbedding.list_supported_models()}
        if model_name not in supported:
            raise ValueError(f"EMBEDDING_MODEL {model_name!r} is not a fastembed text model")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._dimension = TextEmbedding.get_embedding_size(model_name)
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> TextEmbedding:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(self.cache_dir))
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._get_model().embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._get_model().query_embed(text)))
        return vector.tolist()


def get_embedder() -> Embedder:
    """Return the configured embedder: EMBEDDING_MODEL (default BAAI/bge-small-en-v1.5), cached in DATA_DIR/models.

    Raises ValueError for a model fastembed does not provide.
    """
    settings = get_settings()
    model_name = settings.embedding_model.strip() or DEFAULT_EMBEDDING_MODEL
    return FastEmbedEmbedder(model_name, cache_dir=settings.data_dir / "models")
