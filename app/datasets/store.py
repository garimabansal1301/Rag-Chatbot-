"""Datasets on disk, one folder per chat session: <root>/<session_id>/<dataset_id>/{meta.json, data.csv}.

Ids are random hex strings chosen here, never taken from the file name, and every lookup needs both the
session id and the dataset id: a dataset is not reachable through another session's id.
"""

from __future__ import annotations

import builtins  # `list` below is also a method name of DatasetStore
import json
import logging
import os
import re
import shutil
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.datasets.parsing import DatasetError, ParsedCsv
from app.models import DatasetInfo, DatasetSource, PendingWorkbook, WorkbookSheet

logger = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
META_FILE = "meta.json"
DATA_FILE = "data.csv"
SELECTION_FILE = "selection.json"  # per session: {"dataset_id": ...}; not a dataset folder, so list() skips it
MAX_DATASETS_PER_SESSION = 20
PENDING_DIR = ".pending"  # uploaded workbooks awaiting a sheet choice; dot folder, so list() skips it
MAX_PENDING_PER_SESSION = 3
PENDING_TTL = timedelta(hours=1)


class DatasetNotFoundError(KeyError):
    pass


def _age(created_at: str) -> timedelta:
    try:
        return datetime.now(UTC) - datetime.fromisoformat(created_at)
    except ValueError:
        return PENDING_TTL * 2  # unreadable timestamp: treat as expired


class DatasetStore:
    def __init__(self, root: Path, max_per_session: int = MAX_DATASETS_PER_SESSION) -> None:
        self.root = root
        self.max_per_session = max_per_session
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _session_dir(self, session_id: str) -> Path:
        # Only our own id format, so a crafted id can never point outside the store.
        if not _ID_PATTERN.match(session_id):
            raise DatasetNotFoundError(session_id)
        return self.root / session_id

    def _dataset_dir(self, session_id: str, dataset_id: str) -> Path:
        if not _ID_PATTERN.match(dataset_id):
            raise DatasetNotFoundError(dataset_id)
        return self._session_dir(session_id) / dataset_id

    # ------------------------------------------------------------------ API
    def save(
        self, session_id: str, filename: str, parsed: ParsedCsv, size_bytes: int, source: DatasetSource | None = None
    ) -> DatasetInfo:
        """Register a validated CSV (or a converted worksheet, with its `source`) for the session.

        Raises DatasetError when the session is full."""
        session_dir = self._session_dir(session_id)
        with self._lock:
            if len(self.list(session_id)) >= self.max_per_session:
                raise DatasetError(f"This chat already has {self.max_per_session} datasets; start a new chat for more.")
            info = DatasetInfo(
                id=uuid.uuid4().hex,
                session_id=session_id,
                filename=filename,
                size_bytes=size_bytes,
                encoding=parsed.encoding,
                delimiter=parsed.delimiter,
                rows=parsed.rows,
                columns=parsed.columns,
                preview=parsed.preview,
                warnings=parsed.warnings,
                formula_like_cells=parsed.formula_like_cells,
                source=source,
            )
            staging = session_dir / f".{info.id}.tmp"
            staging.mkdir(parents=True)
            try:
                (staging / DATA_FILE).write_text(parsed.text, encoding="utf-8", newline="")
                (staging / META_FILE).write_text(info.model_dump_json(indent=1), encoding="utf-8")
                os.replace(staging, session_dir / info.id)  # readers never see a half-written dataset
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return info

    def list(self, session_id: str) -> list[DatasetInfo]:
        """The session's datasets, oldest first. Unreadable entries are skipped."""
        session_dir = self._session_dir(session_id)
        if not session_dir.is_dir():
            return []
        found = []
        for folder in session_dir.iterdir():
            if folder.name.startswith(".") or not _ID_PATTERN.match(folder.name):
                continue
            try:
                found.append(self.get(session_id, folder.name))
            except DatasetNotFoundError:
                logger.warning("Skipping unreadable dataset folder in session %s", session_id)
        return sorted(found, key=lambda d: (d.created_at, d.filename.casefold(), d.id))  # deterministic order

    def get(self, session_id: str, dataset_id: str) -> DatasetInfo:
        meta = self._dataset_dir(session_id, dataset_id) / META_FILE
        try:
            info = DatasetInfo.model_validate_json(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise DatasetNotFoundError(dataset_id) from None
        if info.session_id != session_id or info.id != dataset_id:
            raise DatasetNotFoundError(dataset_id)
        return info

    def data_path(self, session_id: str, dataset_id: str) -> Path:
        """The stored UTF-8 CSV (for later analysis phases); checks ownership first."""
        self.get(session_id, dataset_id)
        return self._dataset_dir(session_id, dataset_id) / DATA_FILE

    def select(self, session_id: str, dataset_id: str) -> DatasetInfo:
        """Mark one of the session's datasets as the one to analyse (stored per session, not globally)."""
        info = self.get(session_id, dataset_id)  # ownership check
        path = self._session_dir(session_id) / SELECTION_FILE
        with self._lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"dataset_id": info.id}), encoding="utf-8")
            os.replace(tmp, path)
        return info

    def selected(self, session_id: str) -> DatasetInfo | None:
        """The explicitly selected dataset, or None (no selection, or it is no longer readable)."""
        path = self._session_dir(session_id) / SELECTION_FILE
        try:
            dataset_id = json.loads(path.read_text(encoding="utf-8"))["dataset_id"]
            return self.get(session_id, str(dataset_id))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------ uploaded workbooks awaiting a sheet choice
    def save_pending(
        self, session_id: str, filename: str, data: bytes, sheets: builtins.list[WorkbookSheet]
    ) -> PendingWorkbook:
        """Keep an uploaded .xlsx (already checked) until a worksheet is chosen: per session, briefly, bounded."""
        folder = self._session_dir(session_id) / PENDING_DIR
        pending = PendingWorkbook(
            id=uuid.uuid4().hex, session_id=session_id, filename=filename, size_bytes=len(data), sheets=sheets
        )
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            self._prune_pending(folder, keep=MAX_PENDING_PER_SESSION - 1)
            (folder / f"{pending.id}.xlsx").write_bytes(data)
            (folder / f"{pending.id}.json").write_text(pending.model_dump_json(), encoding="utf-8")
        return pending

    def get_pending(self, session_id: str, upload_id: str) -> tuple[PendingWorkbook, bytes]:
        if not _ID_PATTERN.match(upload_id):
            raise DatasetNotFoundError(upload_id)
        folder = self._session_dir(session_id) / PENDING_DIR
        try:
            pending = PendingWorkbook.model_validate_json((folder / f"{upload_id}.json").read_text(encoding="utf-8"))
            data = (folder / f"{upload_id}.xlsx").read_bytes()
        except (OSError, ValueError):
            raise DatasetNotFoundError(upload_id) from None
        if pending.session_id != session_id or pending.id != upload_id or _age(pending.created_at) > PENDING_TTL:
            raise DatasetNotFoundError(upload_id)
        return pending, data

    def latest_pending(self, session_id: str) -> PendingWorkbook | None:
        """The most recent workbook still waiting for a sheet choice (shown in the Datasets panel)."""
        folder = self._session_dir(session_id) / PENDING_DIR
        newest: tuple[str, int] | None = None
        best: PendingWorkbook | None = None
        for meta in folder.glob("*.json") if folder.is_dir() else []:
            try:
                pending, _ = self.get_pending(session_id, meta.stem)
                key = (pending.created_at, meta.stat().st_mtime_ns)  # timestamps have 1 s resolution
            except (DatasetNotFoundError, OSError):
                continue
            if newest is None or key > newest:
                newest, best = key, pending
        return best

    def delete_pending(self, session_id: str, upload_id: str) -> None:
        """Remove one pending upload of this session. Raises DatasetNotFoundError if there is none."""
        if not _ID_PATTERN.match(upload_id):
            raise DatasetNotFoundError(upload_id)
        folder = self._session_dir(session_id) / PENDING_DIR
        with self._lock:
            if not (folder / f"{upload_id}.json").exists():
                raise DatasetNotFoundError(upload_id)
            for suffix in (".xlsx", ".json"):
                (folder / f"{upload_id}{suffix}").unlink(missing_ok=True)

    @staticmethod
    def _prune_pending(folder: Path, keep: int) -> None:
        """Remove expired pending workbooks, then the oldest ones beyond `keep`."""
        entries = []
        for meta in folder.glob("*.json"):
            try:
                created = PendingWorkbook.model_validate_json(meta.read_text(encoding="utf-8")).created_at
                order = meta.stat().st_mtime_ns  # tie-break: timestamps have 1 s resolution
            except (OSError, ValueError):
                created, order = "", 0
            entries.append((created, order, meta.stem))
        entries.sort(reverse=True)
        for index, (created, _, stem) in enumerate(entries):
            if index >= keep or not created or _age(created) > PENDING_TTL:
                for suffix in (".xlsx", ".json"):
                    (folder / f"{stem}{suffix}").unlink(missing_ok=True)

    def delete_session(self, session_id: str) -> None:
        """Remove every dataset of a deleted chat session (no error if it had none)."""
        session_dir = self._session_dir(session_id)
        with self._lock:
            shutil.rmtree(session_dir, ignore_errors=True)
