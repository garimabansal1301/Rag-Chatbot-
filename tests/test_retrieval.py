"""Retriever and RAG-engine guarantees the M7 contract tests do not check. Fakes only, temp folders only."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import SystemMessage
from langchain_core.tracers.context import collect_runs

from app.chat.engine import SNIPPET_CHARS, get_engine
from app.config import get_settings
from app.models import Chunk, Message
from app.vectordb.retriever import get_retriever
from app.vectordb.store import get_vector_store
from tests.fakes import FakeChatModel, HashEmbedder, ListStore, StaticRetriever, make_chunk

TEXTS = ["annual leave is 24 days", "meal allowance is 50 USD", "on-call stipend is 200 USD", "laptops are refreshed"]


class CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return super().embed_query(text)


class FailingEmbedder(HashEmbedder):
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embedding service down")


def _filled_store() -> ListStore:
    store = ListStore()
    store.add([make_chunk(t, position=i) for i, t in enumerate(TEXTS)], HashEmbedder().embed_documents(TEXTS))
    return store


# ---------------------------------------------------------------- retriever
def test_default_k_comes_from_top_k_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "3")
    get_settings.cache_clear()
    assert len(get_retriever(store=_filled_store(), embedder=HashEmbedder()).retrieve("leave")) == 3


@pytest.mark.parametrize("bad_k", [0, -1])
def test_invalid_k_is_rejected(bad_k: int) -> None:
    with pytest.raises(ValueError):
        get_retriever(store=_filled_store(), embedder=HashEmbedder(), k=bad_k)
    with pytest.raises(ValueError):
        get_retriever(store=_filled_store(), embedder=HashEmbedder(), k=2).retrieve("leave", k=bad_k)


def test_empty_question_and_empty_store_do_not_call_the_embedder() -> None:
    embedder = CountingEmbedder()
    assert get_retriever(store=_filled_store(), embedder=embedder, k=2).retrieve("   ") == []
    assert get_retriever(store=ListStore(), embedder=embedder, k=2).retrieve("annual leave") == []
    assert embedder.queries == []


def test_hits_keep_chunk_fields_and_scores(tmp_path: Path) -> None:
    store, embedder = get_vector_store(tmp_path), HashEmbedder()
    chunks = [
        Chunk(id=f"d:{i}", doc_id="d", text=t, source="hr.md", position=i, metadata={"suffix": ".md"})
        for i, t in enumerate(TEXTS)
    ]
    store.add(chunks, embedder.embed_documents(TEXTS))
    hits = get_retriever(store=store, embedder=embedder, k=4).retrieve("how many annual leave days")
    assert hits[0].chunk == chunks[0]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_embedding_errors_propagate() -> None:
    with pytest.raises(RuntimeError):
        get_retriever(store=_filled_store(), embedder=FailingEmbedder(), k=2).retrieve("leave")


# ---------------------------------------------------------------- engine
def _system_prompt(llm: FakeChatModel, call: int = -1) -> str:
    first = llm.calls[call][0]
    assert isinstance(first, SystemMessage)
    return str(first.content)


def test_retrieval_runs_on_every_turn_with_the_newest_message() -> None:
    retriever = StaticRetriever([make_chunk("Leave is 24 days.", source="leave.md")])
    llm = FakeChatModel(reply="ok")
    engine = get_engine(llm=llm, retriever=retriever, tools=[])
    engine.invoke(message="How much leave?", history=[], session_id="s1")
    history = [Message(role="user", content="How much leave?"), Message(role="assistant", content="ok")]
    engine.invoke(message="And for part-timers?", history=history, session_id="s1")
    assert retriever.queries == ["How much leave?", "And for part-timers?"]


def test_context_is_in_this_turns_prompt_only() -> None:
    chunk = make_chunk("The office plant is called Fernando.", source="plants.md")
    llm = FakeChatModel(reply="Fernando")
    get_engine(llm=llm, retriever=StaticRetriever([chunk]), tools=[]).invoke(
        message="Plant name?",
        history=[Message(role="user", content="hi"), Message(role="assistant", content="hello")],
        session_id="s1",
    )
    assert "[1] plants.md" in _system_prompt(llm) and "Fernando" in _system_prompt(llm)
    assert all("Fernando" not in str(m.content) for m in llm.calls[-1][1:]), "context must not enter the history"


def test_sources_are_numbered_in_order_with_scores_and_snippets() -> None:
    long_text = "word " * 100
    chunks = [make_chunk("short text", source="a.md", position=0), make_chunk(long_text, source="b.md", position=1)]
    llm = FakeChatModel()
    result = get_engine(llm=llm, retriever=StaticRetriever(chunks), tools=[]).invoke("q", [], "s1")
    assert [(s.source, s.chunk_id) for s in result.sources] == [("a.md", chunks[0].id), ("b.md", chunks[1].id)]
    assert [s.score for s in result.sources] == [1.0, 0.9]
    assert result.sources[0].snippet == "short text"
    assert result.sources[1].snippet.endswith("…") and len(result.sources[1].snippet) <= SNIPPET_CHARS + 1
    assert "[1] a.md" in _system_prompt(llm) and "[2] b.md" in _system_prompt(llm)


def test_no_hits_means_no_documents_block_and_no_sources() -> None:
    llm = FakeChatModel()
    result = get_engine(llm=llm, retriever=StaticRetriever([]), tools=[]).invoke("q", [], "s1")
    assert result.sources == [] and "<documents>" not in _system_prompt(llm)


def test_default_retriever_uses_the_isolated_empty_store() -> None:
    llm = FakeChatModel()
    result = get_engine(llm=llm, retriever=None, tools=[]).invoke("annual leave?", [], "s1")
    assert result.sources == [] and "<documents>" not in _system_prompt(llm)


def test_retrieval_errors_surface_from_the_engine() -> None:
    retriever = get_retriever(store=_filled_store(), embedder=FailingEmbedder(), k=2)
    with pytest.raises(RuntimeError):
        get_engine(llm=FakeChatModel(), retriever=retriever, tools=[]).invoke("leave", [], "s1")


def test_graph_runs_carry_the_session_id() -> None:
    engine = get_engine(llm=FakeChatModel(), retriever=StaticRetriever([make_chunk("x")]), tools=[])
    with collect_runs() as collected:
        engine.invoke("q", [], "session-42")
    assert collected.traced_runs, "the graph run should be recorded"
    assert all(run.extra.get("metadata", {}).get("session_id") == "session-42" for run in collected.traced_runs)
