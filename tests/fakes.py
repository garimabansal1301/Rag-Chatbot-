"""Deterministic stand-ins used by the tests — no API keys, no network.

Use them in your own tests too: they let you test one module at a time.
The trainer's evaluator uses the same idea with its own (different) data.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from app.models import Chunk, RetrievedChunk

TOOL_TRIGGER = "[[use tool]]"


class FakeChatModel(BaseChatModel):
    """A chat model that answers by rule and records every prompt it receives.

    - Last message is a ToolMessage          -> answers f"{reply} | tool said: <tool output>"
    - A user message contains TOOL_TRIGGER and a tool is bound -> asks for that tool
    - Otherwise                               -> answers `reply`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reply: str = "FAKE-ANSWER"
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    bound_tools: list[str] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:  # type: ignore[override]
        names = [getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools]
        self.bound_tools.extend(n for n in names if n not in self.bound_tools)
        return self

    def prompt_text(self) -> str:
        """Everything the model has been shown, as one string (handy for assertions)."""
        return "\n".join(str(m.content) for call in self.calls for m in call)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        last = messages[-1] if messages else None
        wants_tool = any(TOOL_TRIGGER in str(m.content) for m in messages if m.type == "human")
        if isinstance(last, ToolMessage):
            message = AIMessage(content=f"{self.reply} | tool said: {last.content}")
        elif wants_tool and self.tool_name and self.tool_name in self.bound_tools:
            message = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": self.tool_args, "id": "call_1"}],
            )
        else:
            message = AIMessage(content=self.reply)
        return ChatResult(generations=[ChatGeneration(message=message)])


_WORD = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Bag-of-words vectors via hashing. Same text -> same vector, shared words -> similar."""

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class StaticRetriever:
    """Always returns the same chunks and records the queries it was asked."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        hits = [RetrievedChunk(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(self.chunks)]
        return hits[: k or len(hits)]


class ListStore:
    """Brute-force in-memory vector store (no persistence) for testing retrievers."""

    def __init__(self) -> None:
        self.items: dict[str, tuple[Chunk, list[float]]] = {}

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, embeddings, strict=True):
            self.items[chunk.id] = (chunk, vector)

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=c, score=sum(a * b for a, b in zip(query_embedding, v, strict=True)))
            for c, v in self.items.values()
        ]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:k]

    def delete_document(self, doc_id: str) -> int:
        ids = [cid for cid, (c, _) in self.items.items() if c.doc_id == doc_id]
        for cid in ids:
            del self.items[cid]
        return len(ids)

    def count(self) -> int:
        return len(self.items)


def make_chunk(text: str, source: str = "fake.md", position: int = 0, doc_id: str = "doc-fake") -> Chunk:
    return Chunk(id=f"{doc_id}:{position}", doc_id=doc_id, text=text, source=source, position=position)
