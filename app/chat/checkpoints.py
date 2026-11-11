"""LangGraph checkpointers for paused (human-review) chat turns.

The app uses SQLite (one local file, survives restarts); tests and engines built without one get an
in-memory saver. Both restrict deserialization to LangChain's safe types plus this app's own state types,
so a tampered checkpoint cannot make the loader import arbitrary classes.

Lifecycle (see app/main.py and app/deps.py): the app opens the SQLite saver at start-up, removes orphaned
threads once (`sweep_orphan_checkpoints`), and closes the connection at shutdown.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from app.models import AnalysisTable, ChartSpec, ImageResult, Source
from app.storage.review_store import ReviewStore, ReviewStoreUnreadableError

logger = logging.getLogger(__name__)

STATE_TYPES = (Source, ImageResult, AnalysisTable, ChartSpec)  # pydantic models kept in EngineState


def checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


def memory_checkpointer() -> BaseCheckpointSaver[str]:
    return InMemorySaver(serde=checkpoint_serde())


def open_checkpointer(path: Path) -> SqliteSaver:
    """A SQLite checkpointer at `path` (created if missing). The connection is shared by the server's threads;
    SqliteSaver serializes access with its own lock. setup() keeps the database in WAL mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn, serde=checkpoint_serde())
        saver.setup()
    except BaseException:
        conn.close()
        raise
    return saver


def close_checkpointer(saver: SqliteSaver) -> None:
    """Close the saver's connection, waiting for any write in progress (closing also merges the WAL file)."""
    with saver.lock:
        saver.conn.close()


def checkpoint_threads(saver: BaseCheckpointSaver[str]) -> set[str]:
    """Every thread id that has checkpoints or pending writes."""
    if isinstance(saver, SqliteSaver):
        with saver.cursor(transaction=False) as cur:
            cur.execute("SELECT thread_id FROM checkpoints UNION SELECT thread_id FROM writes")
            return {str(row[0]) for row in cur.fetchall()}
    return {str(c.config["configurable"]["thread_id"]) for c in saver.list(None)}


def sweep_orphan_checkpoints(saver: BaseCheckpointSaver[str], reviews: ReviewStore) -> int:
    """Delete checkpoint threads that no active review refers to; returns how many were deleted.

    Run once at start-up, before requests are served. Threads of pending or processing (not expired)
    reviews are kept; everything else (crash leftovers, expired or decided reviews) cannot be resumed and
    is removed. If the review records cannot all be read, nothing is deleted. Idempotent; the database
    file itself is never deleted.
    """
    threads = checkpoint_threads(saver)
    if not threads:
        return 0
    try:
        active = reviews.active_thread_ids()
    except ReviewStoreUnreadableError as exc:
        logger.warning("Checkpoint cleanup skipped: %s. No checkpoint was deleted.", exc)
        return 0
    orphans = sorted(threads - active)
    for thread_id in orphans:
        saver.delete_thread(thread_id)
    logger.info("Checkpoint cleanup: removed %d orphaned thread(s), kept %d", len(orphans), len(threads & active))
    return len(orphans)
