"""Retrieval relevance threshold (RETRIEVAL_MIN_SCORE): engine filtering, settings and wiring.

Fake retrievers with fixed scores for the exact rules; the real local embedder (cached model, no network)
on a temporary store of the sample text documents for the calibrated behaviour. The live index is not used.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import deps
from app.chat.engine import CONTEXT_PROMPT, NO_MATCH_PROMPT, SYSTEM_PROMPT, get_engine
from app.config import CALIBRATED_MIN_SCORES, DEFAULT_EMBEDDING_MODEL, Settings, get_settings, load_settings
from app.deps import Components, get_components
from app.ingest.pipeline import ingest_file
from app.main import create_app
from app.models import ImageResult, Message, RetrievedChunk
from app.tools import get_tools
from app.tools.image_search import TOOL_NAME, make_image_search_tool
from app.vectordb import embeddings
from app.vectordb.chunking import get_chunker
from app.vectordb.retriever import VectorRetriever
from app.vectordb.store import get_vector_store
from tests.conftest import SAMPLE_DOCS
from tests.fakes import TOOL_TRIGGER, FakeChatModel, make_chunk

THRESHOLD = 0.60
DOCUMENT_BLOCK = "<documents>"


class ScoredRetriever:
    """Returns fixed (text, source, score) hits, best first, and records the queries."""

    def __init__(self, hits: list[tuple[str, str, float]]) -> None:
        self.hits = [
            RetrievedChunk(chunk=make_chunk(text, source=source, position=i, doc_id=f"doc-{source}"), score=score)
            for i, (text, source, score) in enumerate(hits)
        ]
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        return list(self.hits)


UNRELATED = ScoredRetriever(
    [("Laptops are issued on day one.", "onboarding_guide.md", 0.51), ("Hotel cap 150 USD.", "expense_policy.md", 0.50)]
    + [
        ("Badge photos on Mondays.", "onboarding_guide.md", 0.50),
        ("Bench periods.", "Work Engagement Policy.pdf", 0.49),
    ]
)
POLICY = ScoredRetriever(
    [
        ("Annual leave is 25 days per year.", "leave_policy.md", 0.79),
        ("Leave requests go through the HR portal.", "leave_policy.md", 0.60),  # exactly at the threshold
        ("Engineers join the on-call rota after 90 days.", "onboarding_guide.md", 0.59),
        ("Meals are capped at 50 USD.", "expense_policy.md", 0.55),
    ]
)


def _run(retriever: Any, *, min_score: float | None = THRESHOLD, tools: list[Any] | None = None, **kw: Any) -> Any:
    llm = kw.pop("llm", None) or FakeChatModel(reply="ANSWER")
    engine = get_engine(llm=llm, retriever=retriever, tools=tools or [], min_score=min_score)
    result = engine.invoke(kw.pop("question", "What does a USB-C connector look like?"), kw.pop("history", []), "s1")
    return result, str(llm.calls[0][0].content)


# ---------------------------------------------------------------- engine filtering
def test_unrelated_question_gets_no_sources_and_no_document_block() -> None:
    result, prompt = _run(UNRELATED)
    assert result.sources == []
    assert DOCUMENT_BLOCK not in prompt and "Laptops are issued" not in prompt
    assert NO_MATCH_PROMPT in prompt and prompt.startswith(SYSTEM_PROMPT)
    assert UNRELATED.queries, "retrieval still runs on every turn"


def test_relevant_policy_question_keeps_relevant_chunks_and_drops_low_ones() -> None:
    result, prompt = _run(POLICY, question="How many days of annual leave do I get?")
    assert [(s.source, s.score) for s in result.sources] == [("leave_policy.md", 0.79), ("leave_policy.md", 0.60)]
    assert "[1] leave_policy.md\nAnnual leave is 25 days" in prompt
    assert "[2] leave_policy.md\nLeave requests go through" in prompt, "a chunk exactly at the threshold is kept"
    assert "[3]" not in prompt and "on-call rota" not in prompt and "Meals are capped" not in prompt
    assert "cite them like [1]" in prompt and NO_MATCH_PROMPT not in prompt


def test_citations_are_renumbered_from_one() -> None:
    retriever = ScoredRetriever([("Too weak.", "a.md", 0.40), ("Kept one.", "b.md", 0.70), ("Kept two.", "c.md", 0.65)])
    result, prompt = _run(retriever)
    assert [s.source for s in result.sources] == ["b.md", "c.md"], "original order kept"
    assert "[1] b.md\nKept one." in prompt and "[2] c.md\nKept two." in prompt and "Too weak." not in prompt


@pytest.mark.parametrize("min_score", [None, 0, 0.0])
def test_without_a_threshold_every_candidate_is_used_as_before(min_score: float | None) -> None:
    result, prompt = _run(UNRELATED, min_score=min_score)
    assert len(result.sources) == 4 and DOCUMENT_BLOCK in prompt and NO_MATCH_PROMPT not in prompt


def test_get_engine_default_is_backward_compatible() -> None:
    llm = FakeChatModel()
    result = get_engine(llm=llm, retriever=UNRELATED, tools=[]).invoke("q", [], "s1")
    assert len(result.sources) == 4
    assert str(llm.calls[0][0].content) == SYSTEM_PROMPT + "\n\n" + CONTEXT_PROMPT.format(
        context="\n\n".join(f"[{i}] {h.chunk.source}\n{h.chunk.text}" for i, h in enumerate(UNRELATED.hits, 1))
    )


@pytest.mark.parametrize("tools", [[], get_tools()])
def test_empty_store_keeps_existing_behaviour(tools: list[Any]) -> None:
    result, prompt = _run(ScoredRetriever([]), tools=tools)
    assert result.sources == [] and NO_MATCH_PROMPT not in prompt and DOCUMENT_BLOCK not in prompt
    if not tools:
        assert prompt == SYSTEM_PROMPT


def test_no_retriever_keeps_existing_behaviour() -> None:
    llm = FakeChatModel()
    engine = get_engine(llm=llm, retriever=ScoredRetriever([]), tools=[], min_score=THRESHOLD)
    engine.retriever = None  # type: ignore[attr-defined]
    engine.invoke("hi", [], "s1")
    assert str(llm.calls[0][0].content) == SYSTEM_PROMPT


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_get_engine_rejects_invalid_thresholds(bad: float) -> None:
    with pytest.raises(ValueError, match="min_score"):
        get_engine(llm=FakeChatModel(), retriever=None, tools=[], min_score=bad)


# ---------------------------------------------------------------- filtered-out questions can still use tools
def test_filtered_out_question_can_use_the_calculator() -> None:
    llm = FakeChatModel(reply="ANSWER", tool_name="calculator", tool_args={"expression": "17 * 23"})
    result, prompt = _run(UNRELATED, tools=get_tools(), llm=llm, question=f"{TOOL_TRIGGER} What is 17 times 23?")
    assert result.answer == "ANSWER | tool said: 391" and result.sources == []
    assert NO_MATCH_PROMPT in prompt and "calculator for arithmetic" in prompt


def test_filtered_out_question_can_use_image_search() -> None:
    class Images:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, limit: int = 3) -> list[ImageResult]:
            self.queries.append(query)
            return [
                ImageResult(
                    title="USB-C plug",
                    thumbnail_url="https://api.openverse.org/v1/images/1/thumb/",
                    source_page_url="https://commons.wikimedia.org/wiki/File:USB-C.jpg",
                    license_name="CC BY 4.0",
                )
            ]

    provider = Images()
    llm = FakeChatModel(reply="ANSWER", tool_name=TOOL_NAME, tool_args={"query": "usb-c connector"})
    history = [Message(role="user", content=TOOL_TRIGGER), Message(role="assistant", content="ok")]
    result, prompt = _run(UNRELATED, tools=[*get_tools(), make_image_search_tool(provider)], llm=llm, history=history)
    assert provider.queries == ["usb-c connector"] and [i.title for i in result.images] == ["USB-C plug"]
    assert result.sources == [] and NO_MATCH_PROMPT in prompt and "image_search when seeing" in prompt


# ---------------------------------------------------------------- settings
def test_settings_default_follows_the_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    assert embeddings.DEFAULT_EMBEDDING_MODEL == DEFAULT_EMBEDDING_MODEL
    assert CALIBRATED_MIN_SCORES == {"BAAI/bge-small-en-v1.5": 0.60}
    for model in ("", "  ", "BAAI/bge-small-en-v1.5"):
        monkeypatch.setenv("EMBEDDING_MODEL", model)
        assert load_settings().retrieval_min_score == 0.60
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    assert load_settings().retrieval_min_score == 0.0, "no assumed threshold for an uncalibrated model"


@pytest.mark.parametrize(("raw", "expected"), [("0.7", 0.7), (" 0.55 ", 0.55), ("1", 1.0), ("0", 0.0), ("0.0", 0.0)])
def test_explicit_threshold_wins_for_any_model(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
    monkeypatch.setenv("RETRIEVAL_MIN_SCORE", raw)
    for model in ("", "BAAI/bge-base-en-v1.5"):
        monkeypatch.setenv("EMBEDDING_MODEL", model)
        assert load_settings().retrieval_min_score == expected


@pytest.mark.parametrize("raw", ["abc", "0,6", "-0.1", "1.01", "nan", "inf", "60%"])
def test_invalid_thresholds_are_rejected(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("RETRIEVAL_MIN_SCORE", raw)
    with pytest.raises(ValueError, match="RETRIEVAL_MIN_SCORE must be a number from 0 to 1"):
        load_settings()


def test_invalid_threshold_is_rejected_when_settings_are_copied(settings: Settings) -> None:
    with pytest.raises(ValueError, match="RETRIEVAL_MIN_SCORE"):
        replace(settings, retrieval_min_score=1.5)


# ---------------------------------------------------------------- wiring
@pytest.mark.parametrize(("raw", "expected"), [(None, 0.60), ("0.7", 0.7), ("0", None)])
def test_deps_passes_the_configured_threshold(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: Any) -> None:
    if raw is not None:
        monkeypatch.setenv("RETRIEVAL_MIN_SCORE", raw)
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    get_settings.cache_clear()
    captured: dict[str, Any] = {}

    def fake_get_engine(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return None

    monkeypatch.setattr(deps, "get_engine", fake_get_engine)
    deps.build_components()
    assert captured["min_score"] == expected


def test_chat_page_shows_no_source_panel_for_filtered_out_questions(settings: Settings) -> None:
    engine = get_engine(llm=FakeChatModel(reply="General answer."), retriever=UNRELATED, tools=[], min_score=THRESHOLD)
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=engine)
    with TestClient(app) as client:
        session_id = client.post("/api/sessions", json={}).json()["id"]
        html = client.post(f"/ui/sessions/{session_id}/messages", data={"content": "Hi"}).text
    assert "General answer." in html and "source(s)" not in html


# ---------------------------------------------------------------- real local embedder (offline, temp store)
@pytest.fixture(scope="module")
def sample_index(tmp_path_factory: pytest.TempPathFactory) -> Iterator[VectorRetriever]:
    store = get_vector_store(tmp_path_factory.mktemp("threshold-index"))
    embedder = embeddings.get_embedder()
    chunker = get_chunker("fixed", chunk_size=800, chunk_overlap=100)  # the default settings
    files = [p for p in sorted(SAMPLE_DOCS.iterdir()) if p.suffix in {".md", ".txt"}]
    for path in files:
        ingest_file(path, chunker=chunker, embedder=embedder, store=store)
    yield VectorRetriever(store, embedder, k=4)


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        ("How many days of annual leave do I get?", "leave_policy.md"),
        ("What is the daily allowance when travelling?", "expense_policy.md"),
        ("How much does Acme Cloud Storage cost?", "product_faq.txt"),
    ],
)
def test_real_embedder_keeps_policy_citations(
    sample_index: VectorRetriever, question: str, expected_source: str
) -> None:
    result, prompt = _run(sample_index, question=question)
    assert result.sources and result.sources[0].source == expected_source
    assert all(s.score is not None and s.score >= THRESHOLD for s in result.sources)
    assert DOCUMENT_BLOCK in prompt and NO_MATCH_PROMPT not in prompt


@pytest.mark.parametrize(
    "question", ["What does a USB-C connector look like?", "What is the capital of France?", "What is 17 times 23?"]
)
def test_real_embedder_drops_unrelated_questions(sample_index: VectorRetriever, question: str) -> None:
    candidates = sample_index.retrieve(question)
    assert len(candidates) == 4 and all(c.score < THRESHOLD for c in candidates), "candidates exist, all below 0.60"
    result, prompt = _run(sample_index, question=question)
    assert result.sources == [] and DOCUMENT_BLOCK not in prompt and NO_MATCH_PROMPT in prompt


def test_real_embedder_temp_index_is_isolated(sample_index: VectorRetriever) -> None:
    store_dir = Path(sample_index.store.persist_dir)  # type: ignore[attr-defined]
    assert "threshold-index" in store_dir.name and SAMPLE_DOCS.parent not in store_dir.parents
