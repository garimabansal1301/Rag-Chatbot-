"""Human-review records, one JSON file per review: <root>/<session_id>/<review_id>.json.

A review binds a paused graph thread to the chat session it was started in. Ids are random hex strings
chosen here, and every lookup needs both the session id and the review id, so a review (and the thread it
resumes) is not reachable through another session. The thread id stays on the server.

Decisions are claimed atomically (pending -> processing under a lock), so a second or replayed submission
for the same review is refused instead of resuming the graph twice.
"""

from __future__ import annotations

import builtins  # `list` below is also a method name of ReviewStore
import os
import re
import shutil
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.models import Review, ReviewRequest, utc_now

_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
REVIEW_TTL = timedelta(days=3)  # a pending review older than this can no longer be decided


class ReviewNotFoundError(KeyError):
    pass


class ReviewStoreUnreadableError(RuntimeError):
    """The review records could not all be read, so nobody can say which checkpoints are still needed."""


class ReviewConflictError(RuntimeError):
    """The review is no longer pending (already decided, being decided, expired or failed)."""

    def __init__(self, review: Review) -> None:
        super().__init__(review.status)
        self.review = review


class ReviewExpiredError(RuntimeError):
    def __init__(self, review: Review) -> None:
        super().__init__(review.id)
        self.review = review


def is_expired(review: Review) -> bool:
    try:
        return datetime.now(UTC) - datetime.fromisoformat(review.created_at) > REVIEW_TTL
    except ValueError:
        return True  # unreadable timestamp: treat as expired


class ReviewStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def _session_dir(self, session_id: str) -> Path:
        if not _ID_PATTERN.match(session_id):
            raise ReviewNotFoundError(session_id)
        return self.root / session_id

    def _path(self, session_id: str, review_id: str) -> Path:
        if not _ID_PATTERN.match(review_id):
            raise ReviewNotFoundError(review_id)
        return self._session_dir(session_id) / f"{review_id}.json"

    def _write(self, review: Review) -> None:
        path = self._path(review.session_id, review.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(review.model_dump_json(indent=1), encoding="utf-8")
        os.replace(tmp, path)  # readers never see a half-written review

    # ------------------------------------------------------------------ API
    def create(self, session_id: str, request: ReviewRequest) -> Review:
        review = Review(
            id=uuid.uuid4().hex,
            session_id=session_id,
            thread_id=request.thread_id,
            kind=request.kind,
            reasons=request.reasons,
            question=request.question,
            proposed_answer=request.proposed_answer,
            sources=request.sources,
            actions=request.actions,
        )
        with self._lock:
            self._write(review)
        return review

    def get(self, session_id: str, review_id: str) -> Review:
        path = self._path(session_id, review_id)
        try:
            review = Review.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ReviewNotFoundError(review_id) from None
        if review.session_id != session_id or review.id != review_id:
            raise ReviewNotFoundError(review_id)
        return review

    def list(self, session_id: str) -> builtins.list[Review]:
        """The session's reviews, newest first. Unreadable files are skipped."""
        folder = self._session_dir(session_id)
        found = []
        for path in folder.glob("*.json") if folder.is_dir() else []:
            try:
                found.append(self.get(session_id, path.stem))
            except ReviewNotFoundError:
                continue
        return sorted(found, key=lambda r: (r.created_at, r.id), reverse=True)

    def claim(self, session_id: str, review_id: str) -> Review:
        """Move a pending review to `processing` so only one decision can resume its thread.

        Raises ReviewNotFoundError, ReviewConflictError (not pending) or ReviewExpiredError (marked expired).
        """
        with self._lock:
            review = self.get(session_id, review_id)
            if review.status != "pending":
                raise ReviewConflictError(review)
            if is_expired(review):
                review.status, review.decided_at = "expired", utc_now()
                self._write(review)
                raise ReviewExpiredError(review)
            review.status = "processing"
            self._write(review)
        return review

    def update(self, review: Review, **changes: Any) -> Review:
        updated = review.model_copy(update=changes)
        Review.model_validate(updated.model_dump())  # reject invalid statuses etc.
        with self._lock:
            self._write(updated)
        return updated

    def active_thread_ids(self) -> set[str]:
        """Thread ids of every review that may still be resumed: pending or processing, and not expired.

        Strict on purpose (used to decide which checkpoints may be deleted): raises ReviewStoreUnreadableError
        if the folder is missing or unreadable, or if any review file cannot be read and validated, instead of
        skipping it, because a skipped record could be the one that still needs its checkpoint.
        """
        if not self.root.is_dir():
            raise ReviewStoreUnreadableError("the review folder does not exist")
        active: set[str] = set()
        try:
            for session_dir in self.root.iterdir():
                if not session_dir.is_dir() or not _ID_PATTERN.match(session_dir.name):
                    continue  # not a session folder of this store
                for path in session_dir.glob("*.json"):
                    try:
                        review = self.get(session_dir.name, path.stem)
                    except ReviewNotFoundError:
                        raise ReviewStoreUnreadableError(f"unreadable review record {path.name}") from None
                    if review.status in ("pending", "processing") and not is_expired(review):
                        active.add(review.thread_id)
        except OSError as exc:
            raise ReviewStoreUnreadableError(f"cannot list review records ({type(exc).__name__})") from None
        return active

    def delete_session(self, session_id: str) -> builtins.list[str]:
        """Remove a deleted chat's reviews; returns their thread ids so the caller can drop the checkpoints."""
        reviews = self.list(session_id)
        with self._lock:
            shutil.rmtree(self._session_dir(session_id), ignore_errors=True)
        return [r.thread_id for r in reviews]
