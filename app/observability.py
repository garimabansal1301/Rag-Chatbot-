"""MILESTONE 2 — Observability with LangSmith.

LangChain/LangGraph send traces to LangSmith automatically once tracing is on.
Your job is to switch it on safely from `.env` and make traces easy to find.

  LANGSMITH_TRACING=true          (the older name LANGCHAIN_TRACING_V2=true also counts)
  LANGSMITH_API_KEY=lsv2_...
  LANGSMITH_PROJECT=rag-chatbot-<your-name>
  LANGSMITH_ENDPOINT=...          (optional; the evaluator points this at a local fake server)

Rules the evaluator checks (against a local fake LangSmith server, no account needed):
  - `configure_tracing()` returns True only when tracing is switched on AND an API key is set,
    and False otherwise (it must never crash the app when keys are missing).
  - A chat turn produces a trace in the project named by LANGSMITH_PROJECT.
  - The trace carries `session_id` in its metadata (see app/chat/engine.py).
  - The trace contains an LLM run, and your retriever appears as a "retriever" run.
"""

from __future__ import annotations

import logging
import os

from langsmith import utils as ls_utils

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "rag-chatbot"
# Every name LangSmith checks for the tracing switch.
TRACING_FLAGS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING")
_API_KEYS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
_TRUTHY = {"1", "true", "yes", "on"}


def _is_set(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def clear_langsmith_env_cache() -> None:
    """LangSmith caches env values (incl. the project name); clear them so it sees the current environment."""
    ls_utils.get_env_var.cache_clear()  # type: ignore[attr-defined]  # lru_cache hidden by @overload
    ls_utils.get_tracer_project.cache_clear()


def _write_tracing_flags(value: str) -> None:
    """Write `value` ("true"/"false") to LANGSMITH_TRACING and every other tracing flag that is set.

    LangSmith only accepts the exact string "true", so the flags are normalised.
    """
    for name in TRACING_FLAGS:
        if name == "LANGSMITH_TRACING" or _is_set(name):
            os.environ[name] = value
    clear_langsmith_env_cache()


def configure_tracing() -> bool:
    """Called once at app start-up (see app/main.py).

    OUT: True if LangSmith tracing is active, False if it is off or misconfigured.

    The decision is written back to the environment (not `langsmith.configure`),
    so nothing process-global outlives the call and tests can restore it.
    """
    if not any(os.getenv(name, "").strip().lower() in _TRUTHY for name in TRACING_FLAGS):
        clear_langsmith_env_cache()
        return False
    if not any(_is_set(name) for name in _API_KEYS):
        logger.warning("LangSmith tracing is switched on but LANGSMITH_API_KEY is not set; tracing is disabled.")
        _write_tracing_flags("false")  # otherwise LangSmith would still try to send traces without a key
        return False

    project = os.getenv("LANGSMITH_PROJECT", "").strip() or DEFAULT_PROJECT
    os.environ["LANGSMITH_PROJECT"] = project
    _write_tracing_flags("true")
    logger.info("LangSmith tracing enabled for project %r", project)
    return True
