"""MILESTONE 7 — Retriever.

Glue between a question and the vector store: embed the query, search, and
(optionally) post-process — score threshold, de-duplication, MMR, re-ranking.

Rules the evaluator checks:
  - `retrieve(query)` returns at most `k` RetrievedChunk, best first.
  - `retrieve(query, k=2)` overrides the default k.
  - It uses the embedder and store it was GIVEN (the evaluator passes its own).
  - It shows up in LangSmith as a run with run_type="retriever"
    (hint: `from langsmith import traceable` and `@traceable(run_type="retriever")`).
"""

from __future__ import annotations

from typing import Any, Protocol

from langsmith import traceable

from app.config import get_settings
from app.models import RetrievedChunk
from app.vectordb.embeddings import Embedder, get_embedder
from app.vectordb.store import VectorStore, get_vector_store


class Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """IN: a user question, optional k.  OUT: the most relevant chunks."""
        ...


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key != "self"}


def _trace_outputs(hits: list[RetrievedChunk] | None) -> dict[str, Any]:
    """LangSmith shows retriever outputs as documents when they have page_content and metadata.

    `hits` is None when retrieve raised.
    """
    documents = [
        {
            "type": "Document",
            "page_content": hit.chunk.text,
            "metadata": {"source": hit.chunk.source, "chunk_id": hit.chunk.id, "score": hit.score},
        }
        for hit in hits or []
    ]
    return {"documents": documents}


def _check_k(k: int) -> int:
    if k < 1:
        raise ValueError(f"k (TOP_K) must be at least 1, got {k}")
    return k


class VectorRetriever:
    """Embeds the question with the given embedder and searches the given store. No score threshold."""

    def __init__(self, store: VectorStore, embedder: Embedder, k: int) -> None:
        self.store = store
        self.embedder = embedder
        self.k = _check_k(k)

    @traceable(
        run_type="retriever",
        name="retrieve",
        process_inputs=_trace_inputs,
        process_outputs=_trace_outputs,
    )
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        limit = self.k if k is None else _check_k(k)
        if not query.strip() or self.store.count() == 0:
            return []  # nothing to search for, or nothing to search in: skip the embedding call
        return self.store.search(self.embedder.embed_query(query), limit)[:limit]


def get_retriever(
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    k: int | None = None,
) -> Retriever:
    """Build a retriever.

    IN:  store / embedder to use (None -> build the defaults with
         get_vector_store() / get_embedder()), default k (None -> settings.top_k).
    OUT: an object with `.retrieve(query, k=None)`.
         Raises ValueError if k < 1.
    """
    return VectorRetriever(
        store=store if store is not None else get_vector_store(),
        embedder=embedder if embedder is not None else get_embedder(),
        k=k if k is not None else get_settings().top_k,
    )
