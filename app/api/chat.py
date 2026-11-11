"""JSON API for chatting: POST /api/sessions/{id}/messages."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.deps import Components, get_components, get_review_store, get_store
from app.models import Message
from app.services import run_chat_turn
from app.storage.json_store import SessionNotFoundError, SessionStore
from app.storage.review_store import ReviewStore

router = APIRouter(prefix="/api/sessions", tags=["chat"])


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class ChatTurn(BaseModel):
    user_message: Message
    assistant_message: Message


@router.post("/{session_id}/messages", response_model=ChatTurn)
def send_message(
    session_id: str,
    body: SendMessage,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
    reviews: ReviewStore = Depends(get_review_store),
) -> ChatTurn:
    """If the turn needs human review, `assistant_message.review` links it to a pending review."""
    try:
        user_message, assistant_message = run_chat_turn(store, components.engine, session_id, body.content, reviews)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return ChatTurn(user_message=user_message, assistant_message=assistant_message)
