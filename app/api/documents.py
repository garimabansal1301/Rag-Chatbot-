"""JSON API for documents: list and upload (upload also indexes once ingestion works)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.deps import Components, get_components
from app.services import UploadError, save_and_index_upload

router = APIRouter(prefix="/api/documents", tags=["documents"])


class UploadResponse(BaseModel):
    filename: str
    indexed: bool
    chunks_indexed: int
    detail: str


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


@router.get("", response_model=list[str])
def list_documents(settings: Settings = Depends(_settings)) -> list[str]:
    suffixes = settings.allowed_upload_suffixes
    return sorted(p.name for p in settings.documents_dir.glob("*") if p.suffix.lower() in suffixes)


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(_settings),
    components: Components = Depends(get_components),
) -> UploadResponse:
    data = await file.read(settings.max_upload_bytes + 1)
    try:
        result = await run_in_threadpool(save_and_index_upload, settings, components, file.filename or "", data)
    except UploadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return UploadResponse(**result.__dict__)
