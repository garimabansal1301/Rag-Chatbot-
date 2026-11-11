"""Milestone 7 — Retriever and RAG inside the engine."""

from __future__ import annotations

import pytest

from app.chat.engine import get_engine
from app.vectordb.retriever import get_retriever
from tests.fakes import FakeChatModel, HashEmbedder, ListStore, StaticRetriever, make_chunk

pytestmark = pytest.mark.contract


def test_retriever_uses_given_store_and_k() -> None:
    texts = ["annual leave is 24 days", "meal allowance is 50 USD", "on-call stipend is 200 USD"]
    store, embedder = ListStore(), HashEmbedder()
    store.add([make_chunk(t, position=i) for i, t in enumerate(texts)], embedder.embed_documents(texts))
    retriever = get_retriever(store=store, embedder=embedder, k=2)
    hits = retriever.retrieve("how many annual leave days")
    assert len(hits) == 2
    assert hits[0].chunk.text == texts[0]
    assert len(retriever.retrieve("leave", k=1)) == 1


def test_engine_puts_context_in_prompt_and_returns_sources() -> None:
    chunk = make_chunk("The secret office plant is called Fernando.", source="plants.md")
    retriever = StaticRetriever([chunk])
    llm = FakeChatModel()
    result = get_engine(llm=llm, retriever=retriever, tools=[]).invoke(
        message="What is the plant called?", history=[], session_id="s1"
    )
    assert retriever.queries, "the retriever must be called"
    assert "Fernando" in llm.prompt_text(), "retrieved text must reach the LLM"
    assert [s.source for s in result.sources] == ["plants.md"]
    assert result.sources[0].chunk_id == chunk.id
