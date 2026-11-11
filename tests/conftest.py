from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chat.engine import PlaceholderEngine
from app.config import PROJECT_ROOT, Settings, get_settings, load_settings
from app.deps import Components, get_components
from app.main import create_app
from app.observability import TRACING_FLAGS, clear_langsmith_env_cache

SAMPLE_DOCS = PROJECT_ROOT / "data" / "documents"


@pytest.fixture(autouse=True)
def no_langsmith_tracing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start every test with LangSmith tracing off, ignoring the flag loaded from .env.

    A test can still switch tracing on itself (e.g. against a local fake LangSmith
    server); monkeypatch restores the environment afterwards, so it cannot leak.
    """
    for name in TRACING_FLAGS:
        monkeypatch.delenv(name, raising=False)
    clear_langsmith_env_cache()
    yield
    clear_langsmith_env_cache()


@pytest.fixture(autouse=True)
def search_features_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with web and image search at their defaults (off), whatever `.env` says.

    `.env` is loaded into the process at import, so without this a local IMAGE_SEARCH_PROVIDER=openverse
    would change test outcomes. Tests that need a feature set the variable (or pass Settings) themselves.
    """
    monkeypatch.setenv("IMAGE_SEARCH_PROVIDER", "none")
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    monkeypatch.delenv("RETRIEVAL_MIN_SCORE", raising=False)  # model-based default unless a test sets it


@pytest.fixture(autouse=True)
def isolated_vector_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Code that opens the default store (e.g. get_engine(retriever=None)) gets a temp folder, never data/vectorstore.

    DATA_DIR is left alone so the cached embedding model in data/models is still used.
    """
    monkeypatch.setenv("VECTOR_DIR", str(tmp_path_factory.mktemp("vectorstore")))
    monkeypatch.setenv("DATASETS_DIR", str(tmp_path_factory.mktemp("datasets")))  # never data/datasets
    monkeypatch.setenv("REVIEWS_DIR", str(tmp_path_factory.mktemp("reviews")))  # never data/reviews
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path_factory.mktemp("checkpoints") / "graph.sqlite"))
    monkeypatch.setenv("EVAL_REPORTS_DIR", str(tmp_path_factory.mktemp("eval_reports")))  # never evaluation/reports
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings pointing every folder at a fresh temp dir."""
    return replace(
        load_settings(),
        data_dir=tmp_path,
        sessions_dir=tmp_path / "sessions",
        documents_dir=tmp_path / "documents",
        vector_dir=tmp_path / "vectorstore",
        datasets_dir=tmp_path / "datasets",
        reviews_dir=tmp_path / "reviews",
        checkpoint_db=tmp_path / "checkpoints" / "graph.sqlite",
        eval_reports_dir=tmp_path / "eval_reports",
    )


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    """App client with placeholder components: no real LLM calls, no writes to the real vector store."""
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=PlaceholderEngine())
    with TestClient(app) as test_client:
        yield test_client
