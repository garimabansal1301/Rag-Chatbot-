"""Server-rendered pages (Jinja2 + HTMX). Complete — no TODOs here."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.datasets.parsing import MAX_COLUMNS, MAX_ROWS
from app.datasets.store import DatasetNotFoundError, DatasetStore
from app.datasets.xlsx import is_xlsx_name
from app.deps import Components, get_components, get_dataset_store, get_review_store, get_store
from app.models import ReviewDecision
from app.services import (
    ReviewDecisionError,
    UploadError,
    decide_review,
    delete_session_reviews,
    expire_stale_reviews,
    import_workbook_sheet,
    run_chat_turn,
    save_and_index_upload,
    stage_workbook,
    upload_dataset,
)
from app.storage.json_store import SessionNotFoundError, SessionStore
from app.storage.review_store import ReviewConflictError, ReviewExpiredError, ReviewNotFoundError, ReviewStore

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter(include_in_schema=False)


def _redirect(request: Request, url: str) -> Response:
    if request.headers.get("HX-Request"):
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


def _datasets_context(request: Request, session_id: str, status_: dict[str, str] | None = None) -> dict[str, object]:
    """What templates/partials/datasets.html needs for one session."""
    settings: Settings = request.app.state.settings
    datasets: DatasetStore = request.app.state.dataset_store
    items, selected = datasets.list(session_id), datasets.selected(session_id)
    active = selected or (items[0] if len(items) == 1 else None)  # same rule as the analysis tools
    return {
        "session_id": session_id,
        "datasets": items,
        "active_dataset": active,
        "dataset_selected": selected is not None,
        "dataset_limit_mb": f"{settings.max_dataset_bytes / 1024 / 1024:g}",
        "dataset_max_rows": f"{MAX_ROWS:,}",
        "dataset_max_columns": MAX_COLUMNS,
        "dataset_status": status_,
        "pending_workbook": datasets.latest_pending(session_id),  # an .xlsx waiting for a worksheet choice
    }


MAX_DECIDED_REVIEWS_SHOWN = 5


def _reviews_context(request: Request, session_id: str, status_: dict[str, str] | None = None) -> dict[str, object]:
    """What templates/partials/reviews.html needs: undecided reviews first, then the latest decided ones."""
    reviews: ReviewStore = request.app.state.review_store
    items = reviews.list(session_id)
    waiting = [r for r in items if r.status in ("pending", "processing")]
    decided = [r for r in items if r.status not in ("pending", "processing")][:MAX_DECIDED_REVIEWS_SHOWN]
    return {"session_id": session_id, "reviews": waiting + decided, "review_status": status_}


def _render_page(request: Request, store: SessionStore, components: Components, session_id: str | None) -> HTMLResponse:
    current = None
    if session_id:
        try:
            current = store.get(session_id)
        except SessionNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
        expire_stale_reviews(store, request.app.state.review_store, components.engine, session_id)
        current = store.get(session_id)  # expired reviews may have updated their messages
    settings: Settings = request.app.state.settings
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "sessions": store.list(),
            "current": current,
            "missing": components.missing,
            "upload_suffixes": settings.allowed_upload_suffixes,
            "upload_limit_mb": f"{settings.max_upload_bytes / 1024 / 1024:g}",
            **(_datasets_context(request, current.id) if current else {}),
            **(_reviews_context(request, current.id) if current else {}),
        },
    )


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> Response:
    sessions = store.list()
    if sessions:
        return RedirectResponse(f"/sessions/{sessions[0].id}", status_code=status.HTTP_303_SEE_OTHER)
    return _render_page(request, store, components, None)


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
def session_page(
    session_id: str,
    request: Request,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    return _render_page(request, store, components, session_id)


@router.post("/ui/sessions")
def ui_create_session(request: Request, store: SessionStore = Depends(get_store)) -> Response:
    session = store.create()
    return _redirect(request, f"/sessions/{session.id}")


@router.post("/ui/sessions/{session_id}/rename")
def ui_rename_session(
    session_id: str,
    request: Request,
    title: str = Form(...),
    store: SessionStore = Depends(get_store),
) -> Response:
    try:
        store.rename(session_id, title)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return _redirect(request, f"/sessions/{session_id}")


@router.post("/ui/sessions/{session_id}/delete")
def ui_delete_session(
    session_id: str,
    request: Request,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
    reviews: ReviewStore = Depends(get_review_store),
    components: Components = Depends(get_components),
) -> Response:
    try:
        store.delete(session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    datasets.delete_session(session_id)  # a chat's uploaded CSVs go with it
    delete_session_reviews(reviews, components.engine, session_id)  # and its reviews and paused turns
    return _redirect(request, "/")


@router.post("/ui/sessions/{session_id}/datasets/{dataset_id}/select", response_class=HTMLResponse)
def ui_select_dataset(
    session_id: str,
    dataset_id: str,
    request: Request,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> HTMLResponse:
    """Choose which of this chat's datasets the analysis tools use; returns the refreshed datasets panel."""
    try:
        store.get(session_id)
        info = datasets.select(session_id, dataset_id)
    except (SessionNotFoundError, DatasetNotFoundError):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found") from None
    status_ = {"level": "ok", "message": f"{info.filename} will be used for analysis questions in this chat."}
    return templates.TemplateResponse(
        request, "partials/datasets.html", _datasets_context(request, session_id, status_)
    )


@router.post("/ui/sessions/{session_id}/datasets", response_class=HTMLResponse)
async def ui_upload_dataset(
    session_id: str,
    request: Request,
    file: UploadFile,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> HTMLResponse:
    """Upload a CSV, or an .xlsx workbook (then a worksheet is chosen), and return the refreshed datasets panel."""
    settings: Settings = request.app.state.settings
    data = await file.read(settings.max_dataset_bytes + 1)
    filename = file.filename or ""
    try:
        if is_xlsx_name(filename):
            pending = await run_in_threadpool(stage_workbook, settings, store, datasets, session_id, filename, data)
            status_ = {"level": "ok", "message": f"{pending.filename}: choose the worksheet to import below."}
        else:
            info = await run_in_threadpool(upload_dataset, settings, store, datasets, session_id, filename, data)
            status_ = {
                "level": "ok",
                "message": f"{info.filename}: {info.rows:,} rows × {len(info.columns)} columns uploaded and checked.",
                "dataset_id": info.id,
            }
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    except UploadError as exc:
        status_ = {"level": "error", "message": str(exc)}
    return templates.TemplateResponse(
        request, "partials/datasets.html", _datasets_context(request, session_id, status_)
    )


@router.post("/ui/sessions/{session_id}/workbooks/{upload_id}/import", response_class=HTMLResponse)
async def ui_import_worksheet(
    session_id: str,
    upload_id: str,
    request: Request,
    sheet: int = Form(..., ge=0, le=99),
    include_hidden: bool = Form(False),
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> HTMLResponse:
    """Import the chosen worksheet of a pending workbook as a dataset; returns the refreshed datasets panel."""
    try:
        info = await run_in_threadpool(
            import_workbook_sheet, store, datasets, session_id, upload_id, sheet, include_hidden
        )
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    except DatasetNotFoundError:
        status_ = {"level": "error", "message": "That workbook upload has expired. Upload the workbook again."}
    except UploadError as exc:
        status_ = {"level": "error", "message": str(exc)}
    else:
        status_ = {
            "level": "ok",
            "message": f"{info.filename}: {info.rows:,} rows × {len(info.columns)} columns imported and checked.",
            "dataset_id": info.id,
        }
    return templates.TemplateResponse(
        request, "partials/datasets.html", _datasets_context(request, session_id, status_)
    )


@router.post("/ui/sessions/{session_id}/workbooks/{upload_id}/discard", response_class=HTMLResponse)
def ui_discard_workbook(
    session_id: str,
    upload_id: str,
    request: Request,
    store: SessionStore = Depends(get_store),
    datasets: DatasetStore = Depends(get_dataset_store),
) -> HTMLResponse:
    try:
        store.get(session_id)
        datasets.delete_pending(session_id, upload_id)
    except (SessionNotFoundError, DatasetNotFoundError):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found") from None
    return templates.TemplateResponse(request, "partials/datasets.html", _datasets_context(request, session_id))


@router.post("/ui/sessions/{session_id}/messages", response_class=HTMLResponse)
def ui_send_message(
    session_id: str,
    request: Request,
    content: str = Form(..., min_length=1, max_length=8000),
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
    reviews: ReviewStore = Depends(get_review_store),
) -> HTMLResponse:
    try:
        user_message, assistant_message = run_chat_turn(store, components.engine, session_id, content, reviews)
        saved = len(store.get(session_id).messages)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return templates.TemplateResponse(
        request,
        "partials/chat_turn.html",  # the new messages, plus the refreshed Pending reviews panel (out of band)
        {
            # Positions of the two new messages in the saved chat, for the "Export to Excel" links.
            "messages": [user_message, assistant_message],
            "message_offset": saved - 2,
            **_reviews_context(request, session_id),
        },
    )


@router.post("/ui/sessions/{session_id}/reviews/{review_id}/decide", response_class=HTMLResponse)
async def ui_decide_review(
    session_id: str,
    review_id: str,
    request: Request,
    choice: str = Form(...),
    response: str | None = Form(None, max_length=8000),
    note: str | None = Form(None, max_length=1000),
    store: SessionStore = Depends(get_store),
    reviews: ReviewStore = Depends(get_review_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    """Approve, edit or reject a pending review. On success the waiting message is replaced in place
    (HX-Retarget to that message) and the reviews panel is refreshed out of band; on failure only the
    panel is returned, with the reason."""
    try:
        decision = ReviewDecision(choice=choice, response=response, note=(note or "").strip() or None)  # type: ignore[arg-type]
    except ValidationError:
        return _reviews_panel(request, session_id, "That decision is not valid.")
    try:
        review, message, index = await run_in_threadpool(
            decide_review, store, reviews, components.engine, session_id, review_id, decision
        )
    except (SessionNotFoundError, ReviewNotFoundError):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found") from None
    except ReviewDecisionError as exc:
        return _reviews_panel(request, session_id, str(exc))
    except ReviewConflictError as exc:
        return _reviews_panel(request, session_id, f"This review is already {exc.review.status}; nothing was changed.")
    except ReviewExpiredError:
        return _reviews_panel(request, session_id, "This review has expired, so the answer was not sent. Ask again.")
    status_ = {"level": "ok", "message": f"Decision saved: {review.status}."}
    return templates.TemplateResponse(
        request,
        "partials/review_decided.html",
        {"message": message, "message_index": index, **_reviews_context(request, session_id, status_)},
        headers={"HX-Retarget": f"#message-{index}", "HX-Reswap": "innerHTML"},
    )


def _reviews_panel(request: Request, session_id: str, error: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/reviews.html", _reviews_context(request, session_id, {"level": "error", "message": error})
    )


@router.post("/ui/documents", response_class=HTMLResponse)
async def ui_upload_document(
    request: Request,
    file: UploadFile,
    components: Components = Depends(get_components),
) -> HTMLResponse:
    settings: Settings = request.app.state.settings
    data = await file.read(settings.max_upload_bytes + 1)
    try:
        # Embedding can take seconds; a worker thread keeps the server responsive meanwhile.
        result = await run_in_threadpool(save_and_index_upload, settings, components, file.filename or "", data)
    except UploadError as exc:
        message, level = str(exc), "error"
    else:
        level = "ok" if result.indexed else "warn"
        message = f"{result.filename}: {result.detail}" + (" You can ask about it now." if result.indexed else "")
    return templates.TemplateResponse(request, "partials/upload_status.html", {"message": message, "level": level})
