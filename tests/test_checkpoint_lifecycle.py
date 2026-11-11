"""Checkpoint lifecycle: failed-turn cleanup, start-up orphan sweep, SQLite open/close, concurrent first requests.

Temp SQLite files and temp review folders only (the live database is never opened); fake LLMs, no network.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.deps as deps
from app.chat.checkpoints import (
    checkpoint_threads,
    close_checkpointer,
    memory_checkpointer,
    open_checkpointer,
    sweep_orphan_checkpoints,
)
from app.chat.engine import GraphEngine, PlaceholderEngine, get_engine
from app.chat.review import ReviewPolicy
from app.config import Settings
from app.deps import Components
from app.main import create_app
from app.models import ReviewDecision
from app.storage.review_store import ReviewStore, ReviewStoreUnreadableError
from tests.fakes import FakeChatModel, StaticRetriever, make_chunk

SESSION = "a" * 32
OTHER = "b" * 32
LEAVE = make_chunk("Employees get 20 days of annual leave per year.", source="leave_policy.md")


class LLMDownError(RuntimeError):
    pass


class FailingChatModel(FakeChatModel):
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise LLMDownError("LLM down")


def engine(saver: Any, llm: FakeChatModel | None = None) -> GraphEngine:
    built = get_engine(
        llm=llm or FakeChatModel(reply="uncited answer"),
        retriever=StaticRetriever([LEAVE]),
        tools=[],
        review_policy=ReviewPolicy(),
        checkpointer=saver,
    )
    assert isinstance(built, GraphEngine)
    return built


def pause(saver: Any, session_id: str = SESSION) -> Any:
    """A real paused turn (unsupported answer) in `saver`; returns its ReviewRequest."""
    review = engine(saver).invoke("How much leave?", [], session_id).review
    assert review is not None
    return review


_opened: list[Any] = []


def open_saver(path: Path) -> Any:
    saver = open_checkpointer(path)
    _opened.append(saver)
    return saver


@pytest.fixture(autouse=True)
def close_savers() -> Iterator[None]:
    yield
    while _opened:
        close_checkpointer(_opened.pop())


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints" / "graph.sqlite"


@pytest.fixture()
def reviews(tmp_path: Path) -> ReviewStore:
    store = ReviewStore(tmp_path / "reviews")
    store.root.mkdir(parents=True)
    return store


# ---------------------------------------------------------------- Fix A: failed turns clean up their thread
def test_failed_invocation_removes_its_thread_and_keeps_the_original_error() -> None:
    saver = memory_checkpointer()
    with pytest.raises(LLMDownError, match="LLM down"):
        engine(saver, FailingChatModel()).invoke("How much leave?", [], SESSION)
    assert checkpoint_threads(saver) == set()


def test_failed_invocation_does_not_touch_pending_reviews() -> None:
    saver = memory_checkpointer()
    pending = pause(saver)
    with pytest.raises(LLMDownError):
        engine(saver, FailingChatModel()).invoke("Another question", [], SESSION)
    assert checkpoint_threads(saver) == {pending.thread_id}
    resumed = engine(saver).resume(pending.thread_id, SESSION, ReviewDecision(choice="approve"))
    assert resumed.answer == "uncited answer", "the pending turn is still resumable"


def test_cleanup_failure_does_not_hide_the_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    saver = memory_checkpointer()

    def broken_delete(thread_id: str) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(saver, "delete_thread", broken_delete)
    failing = engine(saver, FailingChatModel())
    with pytest.raises(LLMDownError):
        failing.invoke("q", [], SESSION)

    def broken_discard(thread_id: str) -> None:
        raise OSError("discard exploded")

    monkeypatch.setattr(failing, "discard", broken_discard)
    with pytest.raises(LLMDownError):  # not OSError
        failing.invoke("q", [], SESSION)


def test_failed_turn_is_still_reported_to_the_user(settings: Settings) -> None:
    app = create_app(settings)
    app.dependency_overrides[deps.get_components] = lambda: Components(engine=engine(saver, FailingChatModel()))
    with TestClient(app) as client:
        saver = app.state.checkpointer
        sid = client.post("/api/sessions", json={}).json()["id"]
        reply = client.post(f"/api/sessions/{sid}/messages", json={"content": "Leave?"}).json()["assistant_message"]
        assert reply["content"] == "Engine error: LLMDownError: LLM down", "unchanged user-facing behavior"
        assert checkpoint_threads(saver) == set()


# ---------------------------------------------------------------- Fix B: start-up orphan sweep
def test_sweep_removes_orphans_and_keeps_pending_and_processing(db: Path, reviews: ReviewStore) -> None:
    saver = open_saver(db)
    orphan = pause(saver).thread_id  # paused, but no review record (e.g. crash before it was saved)
    pending = reviews.create(SESSION, pause(saver))
    processing = reviews.create(OTHER, pause(saver, OTHER))
    reviews.update(processing, status="processing")
    assert sweep_orphan_checkpoints(saver, reviews) == 1
    assert checkpoint_threads(saver) == {pending.thread_id, processing.thread_id}
    assert orphan not in checkpoint_threads(saver) and db.is_file(), "the database file is never deleted"
    result = engine(saver).resume(pending.thread_id, SESSION, ReviewDecision(choice="approve"))
    assert result.answer == "uncited answer", "a kept checkpoint is still resumable"


def test_sweep_removes_expired_and_decided_review_threads(db: Path, reviews: ReviewStore) -> None:
    saver = open_saver(db)
    expired = reviews.create(SESSION, pause(saver))
    reviews.update(expired, created_at="2000-01-01T00:00:00+00:00")
    decided = reviews.create(SESSION, pause(saver))
    reviews.update(decided, status="approved")
    assert sweep_orphan_checkpoints(saver, reviews) == 2 and checkpoint_threads(saver) == set()
    assert reviews.get(SESSION, expired.id).status == "pending", "review history is not rewritten by the sweep"


def test_sweep_is_idempotent(db: Path, reviews: ReviewStore) -> None:
    saver = open_saver(db)
    pause(saver)
    kept = reviews.create(SESSION, pause(saver))
    assert sweep_orphan_checkpoints(saver, reviews) == 1
    assert sweep_orphan_checkpoints(saver, reviews) == 0
    assert checkpoint_threads(saver) == {kept.thread_id}


@pytest.mark.parametrize("damage", ["garbage", "wrong_session", "missing_folder"])
def test_unreadable_review_records_stop_the_sweep(db: Path, reviews: ReviewStore, damage: str) -> None:
    saver = open_saver(db)
    orphan = pause(saver).thread_id
    kept = reviews.create(SESSION, pause(saver))
    if damage == "garbage":
        (reviews.root / SESSION / ("c" * 32 + ".json")).write_text("{not json", encoding="utf-8")
    elif damage == "wrong_session":  # a record copied into another session's folder
        record = json.loads((reviews.root / SESSION / f"{kept.id}.json").read_text(encoding="utf-8"))
        (reviews.root / OTHER).mkdir()
        (reviews.root / OTHER / f"{kept.id}.json").write_text(json.dumps(record), encoding="utf-8")
    else:
        reviews.root = reviews.root.parent / "not-there"
    with pytest.raises(ReviewStoreUnreadableError):
        reviews.active_thread_ids()
    assert sweep_orphan_checkpoints(saver, reviews) == 0
    assert checkpoint_threads(saver) == {orphan, kept.thread_id}, "nothing is deleted when records are unreadable"


def test_startup_runs_the_sweep_on_the_configured_database(settings: Settings) -> None:
    saver = open_saver(settings.checkpoint_path)
    orphan = pause(saver).thread_id
    pending = ReviewStore(settings.review_root).create(SESSION, pause(saver))
    close_checkpointer(saver)
    app = create_app(settings)
    with TestClient(app):
        live = app.state.checkpointer
        assert checkpoint_threads(live) == {pending.thread_id} and orphan not in checkpoint_threads(live)


def test_failed_sweep_does_not_block_startup(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any) -> int:
        raise RuntimeError("sweep bug")

    monkeypatch.setattr(deps, "sweep_orphan_checkpoints", broken)
    saver = open_saver(settings.checkpoint_path)
    kept = pause(saver).thread_id
    close_checkpointer(saver)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert checkpoint_threads(app.state.checkpointer) == {kept}


# ---------------------------------------------------------------- Fix C: SQLite connection lifecycle
def test_lifespan_opens_the_configured_database_and_closes_it(settings: Settings) -> None:
    app = create_app(settings)
    assert app.state.checkpointer is None
    with TestClient(app):
        saver = app.state.checkpointer
        assert saver is not None and Path(saver.conn.execute("PRAGMA database_list").fetchone()[2]) == (
            settings.checkpoint_path
        )
        assert saver.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert app.state.checkpointer is None and app.state.components is None
    with pytest.raises(sqlite3.ProgrammingError):
        saver.conn.execute("SELECT 1")  # closed
    wal = settings.checkpoint_path.with_name("graph.sqlite-wal")
    assert not wal.exists() or wal.stat().st_size == 0, "closing merged the WAL into the database"


def fake_get_engine(**kwargs: Any) -> GraphEngine:
    return engine(kwargs["checkpointer"])


def test_paused_turn_survives_a_lifespan_restart(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "get_engine", fake_get_engine)
    app = create_app(settings)
    with TestClient(app) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        message = client.post(f"/api/sessions/{sid}/messages", json={"content": "Leave?"}).json()["assistant_message"]
        review_id = message["review"]["id"]
        first_saver = app.state.checkpointer
    with TestClient(create_app(settings)) as client:  # a new process on the same files
        url = f"/api/sessions/{sid}/reviews/{review_id}/decision"
        assert client.post(url, json={"choice": "approve"}).json()["message"]["content"] == "uncited answer"
    with pytest.raises(sqlite3.ProgrammingError):
        first_saver.conn.execute("SELECT 1")


def test_restarted_app_never_uses_the_closed_saver(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "get_engine", fake_get_engine)
    app = create_app(settings)
    with TestClient(app) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        client.get(f"/sessions/{sid}")
        old = app.state.components.engine.checkpointer
    with TestClient(app) as client:  # same app object started again
        reply = client.post(f"/api/sessions/{sid}/messages", json={"content": "Leave?"}).json()["assistant_message"]
        assert not reply["content"].startswith("Engine error")
        assert app.state.components.engine.checkpointer is app.state.checkpointer is not old


def test_startup_failure_is_reported_and_can_be_retried(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(path: Path) -> Any:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(deps, "open_checkpointer", unavailable)
    app = create_app(settings)
    with pytest.raises(sqlite3.OperationalError), TestClient(app):
        pass
    assert app.state.checkpointer is None
    monkeypatch.undo()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200 and app.state.checkpointer is not None


def test_shutdown_failure_is_contained(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(saver: Any) -> None:
        raise sqlite3.OperationalError("close failed")

    monkeypatch.setattr(deps, "close_checkpointer", broken)
    app = create_app(settings)
    with TestClient(app):
        saver = app.state.checkpointer
    assert app.state.checkpointer is None and app.state.components is None
    close_checkpointer(saver)


# ---------------------------------------------------------------- Fix D: concurrent first requests
def counting_patches(monkeypatch: pytest.MonkeyPatch, delay: float = 0.2) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {"open": [], "build": []}
    real_open = deps.open_checkpointer

    def counting_open(path: Path) -> Any:
        saver = real_open(path)
        calls["open"].append(saver)
        _opened.append(saver)
        return saver

    def slow_build(**kwargs: Any) -> Components:
        calls["build"].append(kwargs["checkpointer"])
        time.sleep(delay)  # widen the window in which a second request could start building too
        return Components(engine=PlaceholderEngine())

    monkeypatch.setattr(deps, "open_checkpointer", counting_open)
    monkeypatch.setattr(deps, "build_components", slow_build)
    return calls


def test_simultaneous_first_requests_open_one_saver(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = counting_patches(monkeypatch)
    app = create_app(settings)  # no lifespan start-up: the first request has to open the saver itself
    request = SimpleNamespace(app=app)
    barrier = threading.Barrier(8)

    def first_request(_: int) -> Components:
        barrier.wait()
        return deps.get_components(request)  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(first_request, range(8)))
    assert len(calls["open"]) == 1 and len(calls["build"]) == 1
    assert all(r is results[0] for r in results) and calls["build"][0] is app.state.checkpointer


def test_simultaneous_http_requests_share_one_engine(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = counting_patches(monkeypatch)
    app = create_app(settings)
    with TestClient(app) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        with ThreadPoolExecutor(max_workers=6) as pool:
            statuses = list(pool.map(lambda _: client.get(f"/sessions/{sid}").status_code, range(6)))
        assert statuses == [200] * 6
        assert len(calls["open"]) == 1, "only the start-up opened a connection"
        assert len(calls["build"]) == 1 and calls["build"][0] is app.state.checkpointer


def test_failed_initialization_can_be_retried(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = counting_patches(monkeypatch, delay=0)
    working_build = deps.build_components
    attempts: list[int] = []

    def flaky_build(**kwargs: Any) -> Components:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first build failed")
        return working_build(**kwargs)

    monkeypatch.setattr(deps, "build_components", flaky_build)
    app = create_app(settings)
    request = SimpleNamespace(app=app)
    with pytest.raises(RuntimeError):
        deps.get_components(request)  # type: ignore[arg-type]
    assert app.state.components is None
    components = deps.get_components(request)  # type: ignore[arg-type]
    assert app.state.components is components and len(calls["open"]) == 1, "the opened saver is reused"
