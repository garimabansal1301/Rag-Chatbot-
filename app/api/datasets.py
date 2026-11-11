"""JSON API for datasets of one chat session: /api/sessions/{session_id}/datasets ...

Excel workbooks are imported in two steps under /api/sessions/{session_id}/workbooks: upload the .xlsx
(its worksheets are listed), then import one worksheet as a dataset.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.datasets.store import DatasetNotFoundError, DatasetStore
from app.deps import get_dataset_store, get_store
from app.models import DatasetInfo, PendingWorkbook
from app.services import UploadError, import_workbook_sheet, stage_workbook, upload_dataset
from app.storage.json_store import SessionNotFoundError, SessionStore

router = APIRouter(prefix="/api/sessions/{session_id}/datasets", tags=["datasets"])
workbooks_router = APIRouter(prefix="/api/sessions/{session_id}/workbooks", tags=["datasets"])


class ImportSheet(BaseModel):
    sheet: int = Field(ge=0, le=99, description="Worksheet index from the upload's sheet list")
    include_hidden: bool = Field(default=False, description="Must be true to import a hidden worksheet")


def _require_session(store: SessionStore, session_id: str) -> None:
    try:
        store.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None


@router.get("", response_model=list[DatasetInfo])
def list_datasets(
    session_id: str,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> list[DatasetInfo]:
    _require_session(store, session_id)
    return datasets.list(session_id)


@router.post("", response_model=DatasetInfo, status_code=status.HTTP_201_CREATED)
async def upload(
    session_id: str,
    file: UploadFile,
    request: Request,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> DatasetInfo:
    settings: Settings = request.app.state.settings
    data = await file.read(settings.max_dataset_bytes + 1)
    try:  # parsing can take a moment for large files: keep the event loop free
        return await run_in_threadpool(upload_dataset, settings, store, datasets, session_id, file.filename or "", data)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    except UploadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None


@router.post("/{dataset_id}/select", response_model=DatasetInfo)
def select_dataset(
    session_id: str,
    dataset_id: str,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> DatasetInfo:
    """Use this dataset for analysis questions in this session."""
    _require_session(store, session_id)
    try:
        return datasets.select(session_id, dataset_id)
    except DatasetNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found") from None


@router.get("/{dataset_id}", response_model=DatasetInfo)
def get_dataset(
    session_id: str,
    dataset_id: str,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> DatasetInfo:
    _require_session(store, session_id)
    try:
        return datasets.get(session_id, dataset_id)
    except DatasetNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found") from None


# ---------------------------------------------------------------- Excel workbooks (two-step import)
@workbooks_router.post("", response_model=PendingWorkbook, status_code=status.HTTP_201_CREATED)
async def upload_workbook(
    session_id: str,
    file: UploadFile,
    request: Request,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> PendingWorkbook:
    """Step 1: upload an .xlsx; returns its worksheets. The file waits in this session for up to an hour."""
    settings: Settings = request.app.state.settings
    data = await file.read(settings.max_dataset_bytes + 1)
    try:
        return await run_in_threadpool(stage_workbook, settings, store, datasets, session_id, file.filename or "", data)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    except UploadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None


@workbooks_router.get("/{upload_id}", response_model=PendingWorkbook)
def get_workbook(
    session_id: str,
    upload_id: str,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> PendingWorkbook:
    _require_session(store, session_id)
    try:
        pending, _ = datasets.get_pending(session_id, upload_id)
    except DatasetNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found") from None
    return pending


@workbooks_router.post("/{upload_id}/import", response_model=DatasetInfo, status_code=status.HTTP_201_CREATED)
async def import_workbook(
    session_id: str,
    upload_id: str,
    body: ImportSheet,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> DatasetInfo:
    """Step 2: import one worksheet as a dataset of this session (hidden sheets need include_hidden=true)."""
    try:
        return await run_in_threadpool(
            import_workbook_sheet, store, datasets, session_id, upload_id, body.sheet, body.include_hidden
        )
    except (SessionNotFoundError, DatasetNotFoundError):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found") from None
    except UploadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None


@workbooks_router.delete("/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
def discard_workbook(
    session_id: str,
    upload_id: str,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> None:
    _require_session(store, session_id)
    try:
        datasets.delete_pending(session_id, upload_id)
    except DatasetNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found") from None
