"""FastAPI entry point.  Run with:  uv run uvicorn app.main:app --reload"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import evals_web, web
from app.api import chat, datasets, documents, evals, exports, reviews, sessions
from app.config import Settings, get_settings
from app.datasets.store import DatasetStore
from app.deps import start_checkpoints, start_mcp, stop_checkpoints, stop_mcp
from app.evaluation.runner import EvalRunManager
from app.observability import configure_tracing
from app.storage.json_store import SessionStore
from app.storage.review_store import ReviewStore

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Owns the checkpoint database: opened (and swept) at start-up, closed at shutdown.
    Nothing is opened at import or by create_app, so importing the app creates no files."""
    await run_in_threadpool(start_checkpoints, application.state)
    await run_in_threadpool(start_mcp, application.state)  # no-op unless MCP_ENABLED; never fails the start-up
    try:
        yield
    finally:
        await run_in_threadpool(stop_checkpoints, application.state)
        await run_in_threadpool(stop_mcp, application.state)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass their own Settings that point at temp folders."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    application = FastAPI(title="RAG Chatbot (training scaffold)", lifespan=lifespan)
    application.state.settings = settings
    application.state.session_store = SessionStore(settings.sessions_dir)
    application.state.dataset_store = DatasetStore(settings.dataset_root)
    application.state.review_store = ReviewStore(settings.review_root)  # folders are created on first write
    application.state.components = None  # built lazily by app.deps.get_components
    application.state.checkpointer = None  # the SQLite saver, owned by `lifespan`
    application.state.mcp_toolset = None  # the MCP session (app/mcp/client.py), owned by `lifespan`
    application.state.mcp_error = None  # why MCP is unavailable, when MCP_ENABLED but the server failed
    application.state.init_lock = threading.Lock()  # guards the saver and the first components build
    application.state.tracing_enabled = configure_tracing()
    # Evals dashboard: idle until the user confirms a run (nothing is evaluated at start-up or on page load).
    application.state.eval_manager = EvalRunManager(settings.eval_reports_dir)

    application.include_router(sessions.router)
    application.include_router(chat.router)
    application.include_router(documents.router)
    application.include_router(datasets.router)
    application.include_router(datasets.workbooks_router)
    application.include_router(exports.router)
    application.include_router(reviews.router)
    application.include_router(evals.router)
    application.include_router(web.router)
    application.include_router(evals_web.router)
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        return {"status": "ok", "tracing": application.state.tracing_enabled, "mcp": _mcp_status(application)}

    return application


def _mcp_status(application: FastAPI) -> str:
    state = application.state
    if not state.settings.mcp_enabled:
        return "disabled"
    if state.mcp_error:
        return "unavailable"
    return "connected" if state.mcp_toolset is not None and state.mcp_toolset.running else "not started"


logging.basicConfig(level=logging.INFO)
app = create_app()
