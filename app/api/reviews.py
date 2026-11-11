"""JSON API for human reviews of paused chat turns: /api/sessions/{id}/reviews ...

Responses never include the graph thread id or any checkpoint detail (ReviewSummary has neither).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.deps import Components, get_components, get_review_store, get_store
from app.models import Message, ReviewDecision, ReviewSummary
from app.services import ReviewDecisionError, decide_review
from app.storage.json_store import SessionNotFoundError, SessionStore
from app.storage.review_store import ReviewConflictError, ReviewExpiredError, ReviewNotFoundError, ReviewStore

router = APIRouter(prefix="/api/sessions/{session_id}/reviews", tags=["reviews"])


class DecisionResult(BaseModel):
    review: ReviewSummary
    message: Message
    message_index: int


def _check_session(store: SessionStore, session_id: str) -> None:
    try:
        store.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None


@router.get("", response_model=list[ReviewSummary])
def list_reviews(
    session_id: str, store: SessionStore = Depends(get_store), reviews: ReviewStore = Depends(get_review_store)
) -> list[ReviewSummary]:
    _check_session(store, session_id)
    return list(reviews.list(session_id))


@router.get("/{review_id}", response_model=ReviewSummary)
def get_review(
    session_id: str,
    review_id: str,
    store: SessionStore = Depends(get_store),
    reviews: ReviewStore = Depends(get_review_store),
) -> ReviewSummary:
    _check_session(store, session_id)
    try:
        return reviews.get(session_id, review_id)
    except ReviewNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found") from None


@router.post("/{review_id}/decision", response_model=DecisionResult)
async def submit_decision(
    session_id: str,
    review_id: str,
    body: ReviewDecision,
    store: SessionStore = Depends(get_store),
    reviews: ReviewStore = Depends(get_review_store),
    components: Components = Depends(get_components),
) -> DecisionResult:
    """Approve, edit or reject a pending review; resumes the paused turn. 409 if it was already decided."""
    try:
        review, message, index = await run_in_threadpool(
            decide_review, store, reviews, components.engine, session_id, review_id, body
        )
    except (SessionNotFoundError, ReviewNotFoundError):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found") from None
    except ReviewDecisionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None
    except ReviewConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, f"This review is already {exc.review.status}.") from None
    except ReviewExpiredError:
        raise HTTPException(status.HTTP_410_GONE, "This review has expired and can no longer be decided.") from None
    return DecisionResult(review=review, message=message, message_index=index)
