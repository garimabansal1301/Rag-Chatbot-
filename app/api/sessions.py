"""JSON API for chat sessions: /api/sessions ..."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.datasets.store import DatasetStore
from app.deps import Components, get_components, get_dataset_store, get_review_store, get_store
from app.models import Session, SessionSummary
from app.services import delete_session_reviews
from app.storage.json_store import SessionNotFoundError, SessionStore
from app.storage.review_store import ReviewStore

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class CreateSession(BaseModel):
    title: str | None = None


class RenameSession(BaseModel):
    title: str


def _get_or_404(store: SessionStore, session_id: str) -> Session:
    try:
        return store.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None


@router.get("", response_model=list[SessionSummary])
def list_sessions(store: SessionStore = Depends(get_store)) -> list[SessionSummary]:
    return store.list()


@router.post("", response_model=Session, status_code=status.HTTP_201_CREATED)
def create_session(body: CreateSession | None = None, store: SessionStore = Depends(get_store)) -> Session:
    return store.create(body.title if body else None)


@router.get("/{session_id}", response_model=Session)
def get_session(session_id: str, store: SessionStore = Depends(get_store)) -> Session:
    return _get_or_404(store, session_id)


@router.patch("/{session_id}", response_model=Session)
def rename_session(session_id: str, body: RenameSession, store: SessionStore = Depends(get_store)) -> Session:
    _get_or_404(store, session_id)
    return store.rename(session_id, body.title)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
    reviews: ReviewStore = Depends(get_review_store),
    components: Components = Depends(get_components),
) -> None:
    _get_or_404(store, session_id)
    store.delete(session_id)
    datasets.delete_session(session_id)  # a chat's uploaded CSVs go with it
    delete_session_reviews(reviews, components.engine, session_id)  # and its reviews and paused turns
