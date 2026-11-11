"""One JSON file per chat session. This module is complete — you should not need to change it."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from pathlib import Path

from app.models import Message, Session, SessionSummary, utc_now

_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
DEFAULT_TITLE = "New chat"


class SessionNotFoundError(KeyError):
    pass


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _path(self, session_id: str) -> Path:
        # Only accept our own id format, so a crafted id can never escape the folder.
        if not _ID_PATTERN.match(session_id):
            raise SessionNotFoundError(session_id)
        return self.directory / f"{session_id}.json"

    def _write(self, session: Session) -> None:
        path = self._path(session.id)
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(session.model_dump(), fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: readers never see a half-written file

    # ------------------------------------------------------------------ API
    def list(self) -> list[SessionSummary]:
        summaries = []
        for path in self.directory.glob("*.json"):
            try:
                session = Session.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            summaries.append(
                SessionSummary(
                    id=session.id,
                    title=session.title,
                    updated_at=session.updated_at,
                    message_count=len(session.messages),
                )
            )
        return sorted(summaries, key=lambda s: s.updated_at, reverse=True)

    def create(self, title: str | None = None) -> Session:
        session = Session(id=uuid.uuid4().hex, title=(title or "").strip() or DEFAULT_TITLE)
        with self._lock:
            self._write(session)
        return session

    def get(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        return Session.model_validate_json(path.read_text(encoding="utf-8"))

    def rename(self, session_id: str, title: str) -> Session:
        with self._lock:
            session = self.get(session_id)
            session.title = title.strip() or DEFAULT_TITLE
            session.updated_at = utc_now()
            self._write(session)
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            path = self._path(session_id)
            if not path.exists():
                raise SessionNotFoundError(session_id)
            path.unlink()

    def replace_review_message(self, session_id: str, review_id: str, message: Message) -> int:
        """Replace the assistant message that waits for review `review_id` (kept in its place); returns its index.

        Raises SessionNotFoundError, or KeyError if no message of the session refers to that review.
        """
        with self._lock:
            session = self.get(session_id)
            index = next(
                (i for i, m in enumerate(session.messages) if m.review is not None and m.review.id == review_id), None
            )
            if index is None:
                raise KeyError(review_id)
            session.messages[index] = message
            session.updated_at = utc_now()
            self._write(session)
        return index

    def append_messages(self, session_id: str, *messages: Message) -> Session:
        with self._lock:
            session = self.get(session_id)
            session.messages.extend(messages)
            if session.title == DEFAULT_TITLE:
                first_user = next((m for m in session.messages if m.role == "user"), None)
                if first_user:
                    session.title = first_user.content.strip()[:60] or DEFAULT_TITLE
            session.updated_at = utc_now()
            self._write(session)
        return session
