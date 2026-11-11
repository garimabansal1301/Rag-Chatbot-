"""The evaluation target: the chatbot's own RAG graph, run in isolation.

It builds a separate GraphEngine with the app's retriever, relevance threshold, prompts and LLM, but:
  - no tools (nothing external or consequential can run),
  - no review policy (answers are not held; the app's HUMAN_REVIEW setting and graph are untouched),
  - an in-memory checkpointer (never the app's SQLite checkpoints),
and calls `engine.invoke` directly, so no chat session, message history or review record is written.

The retriever is wrapped in a recorder that only observes: the engine still decides which chunks pass the
threshold (its `sources`), and the recorder supplies the full text of exactly those chunks.
"""

from __future__ import annotations

import re
import threading
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.chat.checkpoints import memory_checkpointer
from app.chat.engine import GraphEngine, get_engine
from app.models import RetrievedChunk
from app.vectordb.retriever import Retriever

EVAL_SESSION_PREFIX = "evaluation-"  # shown as session_id in traces, never a real chat session id
_CITATION = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


class RecordingRetriever:
    """Passes calls through to the real retriever and remembers the chunks it returned (per thread)."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self._local = threading.local()

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        hits = self.inner.retrieve(query, k)
        self._local.hits = list(hits)
        return hits

    def take(self) -> list[RetrievedChunk]:
        hits: list[RetrievedChunk] = getattr(self._local, "hits", [])
        self._local.hits = []
        return hits


def citations(answer: str, context_count: int) -> dict[str, Any]:
    """The [n] citations in an answer, split into valid (1..context_count) and invalid numbers."""
    cited = sorted({int(n) for group in _CITATION.findall(answer) for n in group.split(",")})
    return {
        "cited": [n for n in cited if 1 <= n <= context_count],
        "invalid": [n for n in cited if not 1 <= n <= context_count],
    }


class EvaluationTarget:
    """Callable target for LangSmith `evaluate` (and the local runner): {"question"} -> answer and contexts."""

    def __init__(self, llm: BaseChatModel, retriever: Retriever, min_score: float | None) -> None:
        self.recorder = RecordingRetriever(retriever)
        engine = get_engine(
            llm=llm,
            retriever=self.recorder,
            tools=[],
            min_score=min_score,
            review_policy=None,
            checkpointer=memory_checkpointer(),
        )
        assert isinstance(engine, GraphEngine)
        self.engine = engine

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs["question"])
        session_id = EVAL_SESSION_PREFIX + uuid.uuid4().hex[:12]
        result = self.engine.invoke(question, [], session_id)
        hits = {hit.chunk.id: hit for hit in self.recorder.take()}
        contexts = [
            {
                "number": number,
                "source": source.source,
                "chunk_id": source.chunk_id,
                "score": source.score,
                "text": hits[source.chunk_id].chunk.text if source.chunk_id in hits else source.snippet,
            }
            for number, source in enumerate(result.sources, start=1)
        ]
        return {
            "answer": result.answer,
            "contexts": contexts,
            "sources": sorted({c["source"] for c in contexts}),
            "citations": citations(result.answer, len(contexts)),
            "retrieval_empty": not contexts,
        }


def build_target(llm: BaseChatModel | None = None, retriever: Retriever | None = None) -> EvaluationTarget:
    """The target with the app's own components: the configured LLM (Gemini), the indexed documents (read
    only) and the configured relevance threshold. Building it makes no network call."""
    from app.chat.llm import get_llm
    from app.config import get_settings
    from app.vectordb.retriever import get_retriever

    settings = get_settings()
    return EvaluationTarget(
        llm=llm if llm is not None else get_llm(),
        retriever=retriever if retriever is not None else get_retriever(),
        min_score=settings.retrieval_min_score or None,
    )
