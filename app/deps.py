"""Wires your modules into the web app. You should not need to change this file.

`build_components` calls each `get_*` factory. A factory that still raises
NotImplementedError is skipped, so the app keeps running while you work
through the milestones. Tests (and the trainer's evaluator) replace the whole
bundle with `app.dependency_overrides[get_components] = ...`.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from fastapi import Request
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.chat.checkpoints import close_checkpointer, open_checkpointer, sweep_orphan_checkpoints
from app.chat.engine import ChatEngine, PlaceholderEngine, get_engine
from app.chat.review import ReviewPolicy
from app.config import Settings, get_settings
from app.datasets.store import DatasetStore
from app.mcp.client import MCPToolset, MCPUnavailableError
from app.storage.json_store import SessionStore
from app.storage.review_store import ReviewStore
from app.tools import get_tools
from app.vectordb.chunking import Chunker, get_chunker
from app.vectordb.embeddings import Embedder, get_embedder
from app.vectordb.retriever import Retriever, get_retriever
from app.vectordb.store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class Components:
    engine: ChatEngine
    chunker: Chunker | None = None
    embedder: Embedder | None = None
    store: VectorStore | None = None
    retriever: Retriever | None = None
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def can_index(self) -> bool:
        return None not in (self.chunker, self.embedder, self.store)


def _try(name: str, factory: Callable[..., T], missing: dict[str, str], **kwargs: Any) -> T | None:
    try:
        return factory(**kwargs)
    except NotImplementedError as exc:
        missing[name] = str(exc)
    except Exception as exc:  # a broken module must not take the whole app down
        logger.exception("Could not build %s", name)
        missing[name] = f"{type(exc).__name__}: {exc}"
    return None


def build_components(
    dataset_store: DatasetStore | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    mcp_tools: list[BaseTool] | None = None,
) -> Components:
    """`dataset_store` (the app's, see get_components) switches on the CSV analysis tools for chat sessions;
    `checkpointer` (the app's SQLite one) keeps turns paused for human review across restarts;
    `mcp_tools` (from the app's MCPToolset) replace the local tools of the same name, so none is registered twice."""
    missing: dict[str, str] = {}
    chunker = _try("chunker", get_chunker, missing)
    embedder = _try("embedder", get_embedder, missing)
    store = _try("vector_store", get_vector_store, missing)
    retriever = None
    if store is not None and embedder is not None:
        retriever = _try("retriever", get_retriever, missing, store=store, embedder=embedder)
    settings = get_settings()
    extra_tools = _optional_tools(settings)
    engine_options: dict[str, Any] = {}
    if dataset_store is not None:
        from app.tools.dataset_tools import dataset_context, make_dataset_tools

        extra_tools += make_dataset_tools(dataset_store)
        engine_options["session_context"] = functools.partial(dataset_context, dataset_store)
    remote = {t.name: t for t in mcp_tools or []}
    base_tools = [remote.get(t.name, t) for t in get_tools()]  # same order and names; MCP-served when available
    tools = [*base_tools, *extra_tools] if extra_tools or remote else None  # None: get_engine's defaults, as before
    min_score = settings.retrieval_min_score or None  # 0 disables relevance filtering
    if settings.human_review_enabled:
        engine_options["review_policy"] = ReviewPolicy()
    if checkpointer is not None:
        engine_options["checkpointer"] = checkpointer
    engine = _try(
        "engine", get_engine, missing, retriever=retriever, tools=tools, min_score=min_score, **engine_options
    )
    return Components(
        engine=engine or PlaceholderEngine(),
        chunker=chunker,
        embedder=embedder,
        store=store,
        retriever=retriever,
        missing=missing,
    )


def _optional_tools(settings: Settings) -> list[BaseTool]:
    """Tools switched on in settings (not part of get_tools(), so tests with injected tools never see them)."""
    if settings.image_search_provider == "openverse":
        from app.search.images import OpenverseImageSearch
        from app.tools.image_search import make_image_search_tool

        logger.info("Image search enabled (Openverse)")
        return [make_image_search_tool(OpenverseImageSearch())]
    return []


def get_store(request: Request) -> SessionStore:
    store: SessionStore = request.app.state.session_store
    return store


def get_dataset_store(request: Request) -> DatasetStore:
    datasets: DatasetStore = request.app.state.dataset_store
    return datasets


def get_review_store(request: Request) -> ReviewStore:
    reviews: ReviewStore = request.app.state.review_store
    return reviews


_FALLBACK_INIT_LOCK = threading.Lock()  # only for app states not made by create_app (which sets its own lock)


def _init_lock(state: Any) -> threading.Lock:
    lock: threading.Lock = getattr(state, "init_lock", None) or _FALLBACK_INIT_LOCK
    return lock


def start_checkpoints(state: Any) -> None:
    """App start-up (lifespan): open the SQLite checkpointer at CHECKPOINT_DB and sweep orphaned threads once.

    Failing to open the database stops the start-up (the error propagates); a failed sweep is logged and
    the app starts anyway, with every checkpoint kept.
    """
    with _init_lock(state):
        if getattr(state, "checkpointer", None) is not None:
            return
        saver = open_checkpointer(state.settings.checkpoint_path)
        try:
            sweep_orphan_checkpoints(saver, state.review_store)
        except Exception:
            logger.exception("Checkpoint cleanup failed at start-up; all checkpoints were kept")
        state.checkpointer = saver


def stop_checkpoints(state: Any) -> None:
    """App shutdown (lifespan): drop the components that use the saver, then close its connection."""
    with _init_lock(state):
        saver = getattr(state, "checkpointer", None)
        state.checkpointer = None
        state.components = None  # rebuilt with a fresh saver if the app is started again
    if saver is not None:
        try:
            close_checkpointer(saver)
        except Exception:
            logger.exception("Closing the checkpoint database failed")


def _open_mcp_toolset() -> MCPToolset:
    """Start the local MCP server and discover its tools (blocking; raises MCPUnavailableError)."""
    toolset = MCPToolset()
    toolset.start()
    return toolset


def start_mcp(state: Any) -> None:
    """App start-up (lifespan): with MCP_ENABLED, start the MCP server once for the app's lifetime.

    A failure is logged and recorded (`state.mcp_error`), and the app starts with the local tools instead.
    """
    with _init_lock(state):
        _ensure_mcp(state)


def _ensure_mcp(state: Any) -> None:
    """Called with the init lock held. Tries once per app start; never raises."""
    settings: Settings | None = getattr(state, "settings", None)
    if settings is None or not settings.mcp_enabled or getattr(state, "mcp_toolset", None) is not None:
        return
    if getattr(state, "mcp_error", None) is not None:
        return
    try:
        state.mcp_toolset = _open_mcp_toolset()
    except Exception as exc:
        state.mcp_error = str(exc) if isinstance(exc, MCPUnavailableError) else f"{type(exc).__name__}: {exc}"
        logger.warning("MCP is enabled but unavailable; using the local tools instead (%s)", state.mcp_error)


def stop_mcp(state: Any) -> None:
    """App shutdown (lifespan): close the MCP session, which also ends the server process."""
    with _init_lock(state):
        toolset = getattr(state, "mcp_toolset", None)
        state.mcp_toolset = None
        state.mcp_error = None
        state.components = None  # their MCP tools would point at the closed session
    if toolset is not None:
        try:
            toolset.close()
        except Exception:
            logger.exception("Closing the MCP session failed")


def get_components(request: Request) -> Components:
    state = request.app.state
    components: Components | None = getattr(state, "components", None)
    if components is None:
        with _init_lock(state):  # concurrent first requests: only one builds (and opens the saver)
            components = getattr(state, "components", None)
            if components is None:
                components = _build_app_components(state)
                state.components = components  # set only after a successful build, so failures are retried
                for name, reason in components.missing.items():
                    logger.info("Not available yet: %s (%s)", name, reason)
    return components


def _build_app_components(state: Any) -> Components:
    """Called with the init lock held. Uses the saver opened at start-up; without a lifespan start-up (e.g. a
    test client used without `with`) it opens one here and registers it, so shutdown closes it."""
    checkpointer = getattr(state, "checkpointer", None)
    settings: Settings | None = getattr(state, "settings", None)
    if checkpointer is None and settings is not None:
        checkpointer = state.checkpointer = open_checkpointer(settings.checkpoint_path)
    _ensure_mcp(state)  # normally already started by the lifespan; without one it starts here, once
    toolset: MCPToolset | None = getattr(state, "mcp_toolset", None)
    components = build_components(
        dataset_store=getattr(state, "dataset_store", None),
        checkpointer=checkpointer,
        mcp_tools=toolset.tools if toolset is not None else None,
    )
    if getattr(state, "mcp_error", None):
        components.missing["mcp"] = str(state.mcp_error)
    return components
