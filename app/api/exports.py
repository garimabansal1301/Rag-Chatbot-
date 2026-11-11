"""Excel downloads for one chat session: its datasets and the analysis results saved with its messages.

    GET /api/sessions/{session_id}/datasets/{dataset_id}/data.xlsx      the stored data (Report + Source Data)
    GET /api/sessions/{session_id}/datasets/{dataset_id}/report.xlsx    summary, data quality, optional saved
        ?source=true                                                     result (?message=&table=) and source
    GET /api/sessions/{session_id}/messages/{message}/analysis/{table}/export.xlsx   one saved result table

Authorisation follows the app's existing model (there are no user accounts): everything is scoped to the
session id in the URL. The session must exist, the dataset must belong to it (DatasetStore checks this), and
a result must be saved on one of its assistant messages. Unknown, invalid and other sessions' ids all give
the same 404. No path is ever taken from the request; workbooks are built in memory.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response

from app.datasets.analysis import AnalysisError, load_frame
from app.datasets.excel import (
    XLSX_MEDIA_TYPE,
    ExportError,
    analysis_workbook,
    content_disposition,
    dataset_workbook,
    download_filename,
)
from app.datasets.store import DatasetNotFoundError, DatasetStore
from app.deps import get_dataset_store, get_store
from app.models import AnalysisTable, DatasetInfo, Session
from app.storage.json_store import SessionNotFoundError, SessionStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions/{session_id}", tags=["exports"])

MAX_MESSAGE_INDEX = 100_000
MAX_TABLE_INDEX = 10


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "Not found")


def _session(store: SessionStore, session_id: str) -> Session:
    try:
        return store.get(session_id)  # also refuses ids that are not our 32-hex format
    except SessionNotFoundError:
        raise _not_found() from None


def _dataset(datasets: DatasetStore, session_id: str, dataset_id: str) -> DatasetInfo:
    try:
        return datasets.get(session_id, dataset_id)  # ownership: the dataset must belong to this session
    except DatasetNotFoundError:
        raise _not_found() from None


def _saved_table(session: Session, message: int, table: int) -> AnalysisTable:
    if not 0 <= message < len(session.messages):
        raise _not_found()
    saved = session.messages[message]
    if saved.role != "assistant" or not 0 <= table < len(saved.analysis) or not saved.analysis[table].columns:
        raise _not_found()
    return saved.analysis[table]


def _xlsx(build: Callable[[], bytes], filename: str) -> Response:
    try:
        content = build()
    except ExportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    except AnalysisError:
        raise HTTPException(status.HTTP_409_CONFLICT, "The dataset could not be read. Upload it again.") from None
    except Exception:  # details go to the server log only
        logger.exception("Excel export failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "The workbook could not be created.") from None
    return Response(
        content,
        media_type=XLSX_MEDIA_TYPE,
        headers={
            "Content-Disposition": content_disposition(filename),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/datasets/{dataset_id}/data.xlsx")
def export_dataset_data(
    session_id: str,
    dataset_id: str,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> Response:
    """The dataset's stored rows as typed Excel cells (bounded), with a short Report sheet."""
    _session(store, session_id)
    info = _dataset(datasets, session_id, dataset_id)
    return _xlsx(
        lambda: dataset_workbook(info, load_frame(datasets, session_id, info), summary=False, source=True),
        download_filename(_name_stem(info.filename), "data"),
    )


@router.get("/datasets/{dataset_id}/report.xlsx")
def export_dataset_report(
    session_id: str,
    dataset_id: str,
    source: bool = False,
    message: int | None = Query(default=None, ge=0, le=MAX_MESSAGE_INDEX),
    table: int = Query(default=0, ge=0, le=MAX_TABLE_INDEX),
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> Response:
    """Report, Dataset Summary and Data Quality sheets; a saved result with ?message=&table=; Source Data with
    ?source=true."""
    session = _session(store, session_id)
    info = _dataset(datasets, session_id, dataset_id)
    saved = None
    if message is not None:
        saved = _saved_table(session, message, table)
        if saved.dataset != info.filename:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "That result was calculated from a different dataset.")
    return _xlsx(
        lambda: dataset_workbook(
            info, load_frame(datasets, session_id, info), summary=True, source=source, table=saved
        ),
        download_filename(_name_stem(info.filename), "report"),
    )


@router.get("/messages/{message_index}/analysis/{table_index}/export.xlsx")
def export_analysis_result(
    session_id: str,
    message_index: int,
    table_index: int,
    store: SessionStore = Depends(get_store),
) -> Response:
    """One analysis result exactly as saved with the message (no recalculation, no LLM)."""
    if not (0 <= message_index <= MAX_MESSAGE_INDEX and 0 <= table_index <= MAX_TABLE_INDEX):
        raise _not_found()
    saved = _saved_table(_session(store, session_id), message_index, table_index)
    return _xlsx(lambda: analysis_workbook(saved), download_filename(_name_stem(saved.dataset), saved.title))


_SHEET_DATASET = re.compile(r"^(?P<workbook>.+)\.xlsx \[(?P<sheet>.+)\]$", re.IGNORECASE)


def _name_stem(dataset_name: str) -> str:
    """'sales.csv' -> 'sales'; an imported sheet 'Q1.xlsx [Orders]' -> 'Q1-Orders' (Path.stem would drop the sheet)."""
    match = _SHEET_DATASET.match(dataset_name)
    if match:
        return f"{match['workbook']}-{match['sheet']}"
    return Path(dataset_name).stem
