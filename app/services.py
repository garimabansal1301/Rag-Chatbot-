"""Use-cases shared by the JSON API and the HTML pages. Complete — no TODOs here."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.chat.engine import ChatEngine, ReviewableEngine, ReviewUnavailableError
from app.config import Settings
from app.datasets.parsing import DatasetError, parse_csv
from app.datasets.store import DatasetNotFoundError, DatasetStore
from app.datasets.xlsx import import_sheet, is_xlsx_name, list_sheets
from app.deps import Components
from app.ingest.pipeline import document_from_bytes, ingest_document
from app.models import (
    DatasetInfo,
    Document,
    EngineResult,
    Message,
    MessageReview,
    PendingWorkbook,
    Review,
    ReviewDecision,
    ReviewStatus,
    utc_now,
)
from app.storage.json_store import SessionNotFoundError, SessionStore
from app.storage.review_store import (
    ReviewConflictError,
    ReviewExpiredError,
    ReviewNotFoundError,
    ReviewStore,
    is_expired,
)

logger = logging.getLogger(__name__)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9 ._()-]+")
_WINDOWS_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10))}
MAX_NAME_LENGTH = 120
# One upload at a time in this process: the duplicate check, indexing and the final rename happen together.
_UPLOAD_LOCK = threading.Lock()


WAITING_FOR_REVIEW = {
    "unsupported_answer": "This answer is waiting for a support agent's review, because the documents may not "
    "support it. It appears here once it is approved.",
    "sensitive_request": "This request needs a support agent. The suggested reply appears here once it is reviewed.",
    "consequential_action": "This request needs an action that must be approved first. Nothing has been done yet.",
}
REVIEW_EXPIRED = "This review expired before a decision was made, so no answer was given. Please ask again."
REVIEW_FAILED = "The review decision was recorded, but continuing the answer failed. Please ask again."
_DECISION_STATUS: dict[str, ReviewStatus] = {"approve": "approved", "edit": "edited", "reject": "rejected"}


def _waiting(message: Message) -> bool:
    return message.review is not None and message.review.status in ("pending", "processing")


def run_chat_turn(
    store: SessionStore,
    engine: ChatEngine,
    session_id: str,
    content: str,
    reviews: ReviewStore | None = None,
) -> tuple[Message, Message]:
    """Send one user message through the engine and save both messages.

    If the turn pauses for human review, the assistant message is a placeholder linked to a new review
    (saved in `reviews`); deciding the review replaces it (see decide_review).
    """
    session = store.get(session_id)  # raises SessionNotFoundError before we call the engine
    user_message = Message(role="user", content=content)
    # Placeholders of undecided reviews are not answers, so the model does not see them as such.
    history = [m for m in session.messages if not _waiting(m)]
    try:
        result = engine.invoke(message=content, history=history, session_id=session_id)
        if result.review is not None:
            assistant_message = _review_placeholder(reviews, engine, session_id, result)
        else:
            assistant_message = Message(
                role="assistant",
                content=result.answer,
                sources=result.sources,
                web=result.web,
                images=result.images,
                analysis=result.analysis,
            )
    except Exception as exc:
        logger.exception("Engine failed for session %s", session_id)
        assistant_message = Message(role="assistant", content=f"Engine error: {type(exc).__name__}: {exc}")
    store.append_messages(session_id, user_message, assistant_message)
    return user_message, assistant_message


def _review_placeholder(
    reviews: ReviewStore | None, engine: ChatEngine, session_id: str, result: EngineResult
) -> Message:
    request = result.review
    assert request is not None
    if reviews is None or not isinstance(engine, ReviewableEngine):  # nothing could ever resume it
        if isinstance(engine, ReviewableEngine):
            engine.discard(request.thread_id)
        return Message(role="assistant", content="This answer needs human review, but reviews are not available here.")
    review = reviews.create(session_id, request)
    logger.info("Review %s (%s) created for session %s", review.id, review.kind, session_id)
    return Message(
        role="assistant",
        content=WAITING_FOR_REVIEW[review.kind],
        sources=result.sources,
        review=MessageReview(id=review.id, kind=review.kind),
    )


class ReviewDecisionError(ValueError):
    """The submitted decision does not fit the review (e.g. editing an action, or an empty edited answer)."""


def _check_decision(review: Review, decision: ReviewDecision) -> None:
    if decision.choice == "edit":
        if review.kind == "consequential_action":
            raise ReviewDecisionError("An action can only be approved or rejected, not edited.")
        if not (decision.response or "").strip():
            raise ReviewDecisionError("Write the edited response before submitting it.")


def decide_review(
    store: SessionStore,
    reviews: ReviewStore,
    engine: ChatEngine,
    session_id: str,
    review_id: str,
    decision: ReviewDecision,
) -> tuple[Review, Message, int]:
    """Apply a reviewer's decision: resume the paused graph and replace the waiting message with the result.

    Returns the updated review, the new assistant message and its index in the session.
    Raises SessionNotFoundError / ReviewNotFoundError (unknown, or another session's review),
    ReviewDecisionError (invalid decision), ReviewConflictError (already decided: duplicate or replay)
    or ReviewExpiredError (too old, or its checkpoint is gone).
    """
    store.get(session_id)
    review = reviews.get(session_id, review_id)  # ownership: only this session's reviews
    _check_decision(review, decision)
    if not isinstance(engine, ReviewableEngine):
        raise ReviewExpiredError(review)
    try:
        review = reviews.claim(session_id, review_id)  # pending -> processing, once
    except ReviewExpiredError as exc:
        _close_expired(store, engine, exc.review)
        raise
    metadata = {"review_id": review.id, "review_kind": review.kind}
    try:
        result = engine.resume(review.thread_id, session_id, decision, metadata)
    except ReviewUnavailableError:
        expired = reviews.update(review, status="expired", decided_at=utc_now())
        _close_expired(store, engine, expired)
        raise ReviewExpiredError(expired) from None
    except Exception:
        logger.exception("Resuming review %s failed", review.id)
        engine.discard(review.thread_id)
        review = reviews.update(review, status="failed", decided_at=utc_now(), note=decision.note)
        message = Message(role="assistant", content=REVIEW_FAILED, review=_message_review(review))
        return review, message, _replace(store, session_id, review.id, message)
    status = _DECISION_STATUS[decision.choice]
    logger.info("Review %s %s (%s)", review.id, status, review.kind)
    if result.review is not None:  # the resumed turn paused again, e.g. for a further action
        follow_up = reviews.create(session_id, result.review)
        message = Message(
            role="assistant",
            content=WAITING_FOR_REVIEW[follow_up.kind],
            sources=result.sources,
            review=MessageReview(id=follow_up.id, kind=follow_up.kind),
        )
        final_answer = None
    else:
        message = Message(
            role="assistant",
            content=result.answer,
            sources=result.sources,
            images=result.images,
            analysis=result.analysis,
            review=MessageReview(id=review.id, kind=review.kind, status=status),
        )
        final_answer = result.answer
    review = reviews.update(review, status=status, decided_at=utc_now(), final_answer=final_answer, note=decision.note)
    return review, message, _replace(store, session_id, review_id, message)


def _message_review(review: Review) -> MessageReview:
    return MessageReview(id=review.id, kind=review.kind, status=review.status)


def _replace(store: SessionStore, session_id: str, review_id: str, message: Message) -> int:
    try:
        return store.replace_review_message(session_id, review_id, message)
    except KeyError:  # the waiting message is gone: add the result at the end instead
        return len(store.append_messages(session_id, message).messages) - 1


def _close_expired(store: SessionStore, engine: ChatEngine, review: Review) -> None:
    if isinstance(engine, ReviewableEngine):
        engine.discard(review.thread_id)
    message = Message(role="assistant", content=REVIEW_EXPIRED, review=_message_review(review))
    with contextlib.suppress(KeyError, SessionNotFoundError):
        store.replace_review_message(review.session_id, review.id, message)


def expire_stale_reviews(store: SessionStore, reviews: ReviewStore, engine: ChatEngine, session_id: str) -> None:
    """Mark pending reviews past REVIEW_TTL as expired and drop their checkpoints (run when a chat is shown)."""
    for review in reviews.list(session_id):
        if review.status == "pending" and is_expired(review):
            try:
                reviews.claim(session_id, review.id)  # an expired review is marked expired and raises
            except ReviewExpiredError as exc:
                _close_expired(store, engine, exc.review)
            except (ReviewConflictError, ReviewNotFoundError):
                continue  # decided or removed meanwhile


def delete_session_reviews(reviews: ReviewStore, engine: ChatEngine, session_id: str) -> None:
    """A deleted chat's reviews and their paused graph threads go with it."""
    for thread_id in reviews.delete_session(session_id):
        if isinstance(engine, ReviewableEngine):
            engine.discard(thread_id)


class UploadError(ValueError):
    pass


@dataclass
class UploadResult:
    filename: str
    indexed: bool
    chunks_indexed: int
    detail: str


def safe_filename(filename: str) -> str:
    """The last path component, keeping letters, digits, spaces and . _ ( ) -; '' if nothing usable is left."""
    name = re.split(r"[\\/]", filename or "")[-1]
    name = " ".join(_SAFE_NAME.sub("_", name).split()).strip(" ._")
    stem, suffix = os.path.splitext(name)
    if stem.split(".")[0].upper() in _WINDOWS_DEVICE_NAMES:  # e.g. "con.md" is a device on Windows
        stem = f"_{stem}"
    return stem[: MAX_NAME_LENGTH - len(suffix)] + suffix if stem else ""


def save_and_index_upload(settings: Settings, components: Components, filename: str, data: bytes) -> UploadResult:
    """Check, index and save one uploaded file into DOCUMENTS_DIR.

    Raises UploadError if the file is rejected or indexing fails; the folder and the index are then
    unchanged. The file is moved into place only after indexing succeeded, so a failed upload never
    replaces an earlier copy. Uploading the same name again replaces that document and its chunks.
    """
    safe_name = safe_filename(filename)
    if not safe_name or Path(safe_name).suffix.lower() not in settings.allowed_upload_suffixes:
        allowed = ", ".join(settings.allowed_upload_suffixes)
        raise UploadError(f"Unsupported file type. Allowed: {allowed}")
    if len(data) > settings.max_upload_bytes:
        raise UploadError(f"File too large (limit {settings.max_upload_bytes / 1024 / 1024:g} MB)")
    if not data.strip():
        raise UploadError("File is empty")
    try:
        document = document_from_bytes(safe_name, data)
    except ValueError as exc:  # unreadable text, damaged or scanned PDF, ...
        raise UploadError(str(exc)) from None
    if not document.text.strip():
        raise UploadError(f"{safe_name}: no text to index")

    folder = settings.documents_dir
    folder.mkdir(parents=True, exist_ok=True)
    with _UPLOAD_LOCK:
        _check_no_duplicate(folder, safe_name, data)
        partial = folder / f".{uuid4().hex}.part"  # ignored by listing and ingestion (wrong suffix)
        partial.write_bytes(data)
        try:
            count = _index(components, document)
            os.replace(partial, folder / safe_name)
        finally:
            partial.unlink(missing_ok=True)
    if count is None:
        missing = ", ".join(sorted(k for k in ("chunker", "embedder", "vector_store") if k in components.missing))
        return UploadResult(safe_name, False, 0, f"Saved. Indexing not available yet ({missing or 'not implemented'}).")
    return UploadResult(safe_name, True, count, f"Saved and indexed {count} chunk{'' if count == 1 else 's'}.")


def upload_dataset(
    settings: Settings, sessions: SessionStore, datasets: DatasetStore, session_id: str, filename: str, data: bytes
) -> DatasetInfo:
    """Validate a CSV and register it for one chat session (preview and metadata only; nothing goes to the LLM).

    Raises SessionNotFoundError for an unknown session, UploadError for anything wrong with the file.
    """
    sessions.get(session_id)
    safe_name = safe_filename(filename)
    if safe_name and is_xlsx_name(safe_name):
        raise UploadError(
            "Excel workbooks are imported in two steps: upload them to /workbooks, then pick a worksheet."
        )
    if not safe_name or Path(safe_name).suffix.lower() != ".csv":
        raise UploadError("Unsupported file type. Upload a .csv file or an Excel .xlsx workbook.")
    if len(data) > settings.max_dataset_bytes:
        raise UploadError(f"File too large (limit {settings.max_dataset_bytes / 1024 / 1024:g} MB)")
    try:  # the content decides, not the file name or the browser's MIME type
        return datasets.save(session_id, safe_name, parse_csv(data), len(data))
    except DatasetError as exc:
        raise UploadError(str(exc)) from None


def stage_workbook(
    settings: Settings, sessions: SessionStore, datasets: DatasetStore, session_id: str, filename: str, data: bytes
) -> PendingWorkbook:
    """Step 1 of an Excel import: check the .xlsx and list its worksheets; the file waits in this session.

    Raises SessionNotFoundError for an unknown session, UploadError for anything wrong with the file.
    """
    sessions.get(session_id)
    safe_name = safe_filename(filename)
    if not safe_name or not is_xlsx_name(safe_name):
        raise UploadError("Unsupported file type. Excel workbooks must be .xlsx (not .xls or .xlsm).")
    if len(data) > settings.max_dataset_bytes:
        raise UploadError(f"File too large (limit {settings.max_dataset_bytes / 1024 / 1024:g} MB)")
    try:
        sheets = list_sheets(data)
    except DatasetError as exc:
        raise UploadError(str(exc)) from None
    return datasets.save_pending(session_id, safe_name, data, sheets)


def import_workbook_sheet(
    sessions: SessionStore,
    datasets: DatasetStore,
    session_id: str,
    upload_id: str,
    sheet_index: int,
    include_hidden: bool = False,
) -> DatasetInfo:
    """Step 2: import one worksheet of a pending workbook as a dataset of this session.

    Raises SessionNotFoundError, DatasetNotFoundError (unknown or expired upload) or UploadError.
    """
    sessions.get(session_id)
    pending, data = datasets.get_pending(session_id, upload_id)  # ownership: only this session's uploads
    sheet = next((s for s in pending.sheets if s.index == sheet_index), None)
    if sheet is None:
        raise UploadError("That worksheet does not exist in this workbook.")
    if sheet.state != "visible" and not include_hidden:
        raise UploadError(f"Worksheet {sheet.name!r} is hidden in the workbook. Confirm to import a hidden sheet.")
    try:
        parsed, source = import_sheet(data, sheet_index)
        info = datasets.save(
            session_id,
            f"{pending.filename} [{sheet.name}]",
            parsed,
            pending.size_bytes,
            source=source.model_copy(update={"workbook": pending.filename}),
        )
    except DatasetError as exc:
        raise UploadError(str(exc)) from None
    with contextlib.suppress(DatasetNotFoundError):  # already removed (e.g. a second click): the import succeeded
        datasets.delete_pending(session_id, upload_id)
    return info


def _check_no_duplicate(folder: Path, safe_name: str, data: bytes) -> None:
    """Reject the same bytes under another name, and names differing only in case (one file on Windows)."""
    for existing in folder.iterdir():
        if not existing.is_file() or existing.name == safe_name or existing.name.startswith("."):
            continue
        if existing.name.lower() == safe_name.lower():
            raise UploadError(f"{existing.name} already exists; upload it with exactly that name to replace it")
        if existing.stat().st_size == len(data) and existing.read_bytes() == data:
            raise UploadError(f"{safe_name}: the same file is already uploaded as {existing.name}")


def _index(components: Components, document: Document) -> int | None:
    """Chunks stored, or None while indexing is not implemented. Failures become UploadError."""
    if not components.can_index:
        return None
    assert components.chunker and components.embedder and components.store
    try:
        return ingest_document(
            document, chunker=components.chunker, embedder=components.embedder, store=components.store
        )
    except NotImplementedError:
        return None
    except Exception as exc:  # embedding or store failure: report it instead of claiming success
        logger.exception("Indexing %s failed", document.source)
        raise UploadError(f"{document.source}: indexing failed, file not saved ({type(exc).__name__}: {exc})") from None
