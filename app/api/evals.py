"""JSON API for the Evals dashboard: /api/evals/...

Every GET and the preview are read-only and local: they read the dataset file and the report folders, and
never call Gemini or LangSmith. Only POST /api/evals/run starts an evaluation, and only with the explicit
confirmations below. The app has no user accounts, so like the rest of the app these endpoints are open to
whoever can reach the server.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.deps import get_components
from app.evaluation import dashboard
from app.evaluation.cli import estimate_calls, langsmith_key_present
from app.evaluation.dataset import DATASET_NAME, DatasetError, EvalExample, load_dataset, select
from app.evaluation.runner import EvalRunManager, RunInProgressError

router = APIRouter(prefix="/api/evals", tags=["evals"])

ExampleId = Annotated[str, StringConstraints(pattern=r"^q\d{2,3}$")]
RequestId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9-]{8,64}$")]
Mode = Literal["local", "langsmith"]
Scope = Literal["all", "selected"]

DATA_SENT_TO_GEMINI = (
    "Gemini answer-generation calls may incur usage: each question is sent with up to the configured number of "
    "retrieved excerpts."
)
DATA_SENT_TO_JUDGE = (
    "LLM-as-a-judge calls may incur additional usage: the question, the generated answer and the excerpts or the "
    "reference answer are sent to the judge model."
)
LANGSMITH_NOTICE = (
    "LangSmith mode sends the evaluation questions, generated answers, retrieved excerpts, scores and "
    "explanations to LangSmith as an experiment."
)
LOCAL_NOTICE = "Local mode does not upload evaluation results to LangSmith; the report stays on this machine."


class PreviewRequest(BaseModel):
    scope: Scope = "all"
    example_ids: list[ExampleId] = Field(default_factory=list, max_length=200)
    mode: Mode = "local"


class RunRequest(PreviewRequest):
    request_id: RequestId
    confirm_gemini_usage: bool = False
    confirm_langsmith_upload: bool = False
    acknowledged_max_calls: int = Field(ge=0)


def get_eval_manager(request: Request) -> EvalRunManager:
    manager: EvalRunManager = request.app.state.eval_manager
    return manager


def reports_dir(request: Request) -> Any:
    settings: Settings = request.app.state.settings
    return settings.eval_reports_dir


def dataset_examples() -> list[EvalExample]:
    """The local dataset file, validated (no document check, no network). 503 if it cannot be read."""
    try:
        return load_dataset()
    except DatasetError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The evaluation dataset could not be read.") from None


def resolve_selection(examples: list[EvalExample], scope: Scope, ids: list[str]) -> list[EvalExample]:
    """The examples a request names. 422 for an empty selection, duplicates or unknown ids."""
    if scope == "all":
        return examples
    if not ids:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Select at least one example.")
    if len(set(ids)) != len(ids):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Each example can be selected only once.")
    try:
        return select(examples, ids)
    except DatasetError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None


def preview_payload(chosen: list[EvalExample], mode: Mode) -> dict[str, Any]:
    calls = estimate_calls(len(chosen))
    return {
        "dataset": DATASET_NAME,
        "mode": mode,
        "example_count": len(chosen),
        "example_ids": [e.id for e in chosen],
        "answer_calls": calls["answer_calls"],
        "judge_calls": calls["judge_calls"],
        "max_gemini_calls": calls["max_total"],
        "model": get_settings().llm_model or None,
        "sends_to_langsmith": mode == "langsmith",
        "langsmith_configured": langsmith_key_present(),
        "warnings": [
            DATA_SENT_TO_GEMINI,
            DATA_SENT_TO_JUDGE,
            LANGSMITH_NOTICE if mode == "langsmith" else LOCAL_NOTICE,
        ],
    }


def _not_found(what: str = "Evaluation run not found") -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, what)


# --------------------------------------------------------------------------- read-only
@router.get("/summary")
def summary(request: Request, manager: EvalRunManager = Depends(get_eval_manager)) -> dict[str, Any]:
    data = dashboard.dashboard_summary(reports_dir(request), len(dataset_examples()))
    active = manager.active()
    return {**data, "active_job": active.public() if active else None}


@router.get("/dataset")
def dataset(
    q: str = Query("", max_length=200),
    category: str = Query("", max_length=40),
) -> dict[str, Any]:
    examples = dataset_examples()
    rows = dashboard.dataset_rows(examples, q, category)
    return {
        "dataset": DATASET_NAME,
        "total": len(examples),
        "count": len(rows),
        "categories": dashboard.categories(examples),
        "examples": rows,
    }


@router.get("/dataset/{example_id}")
def dataset_example(example_id: str) -> dict[str, Any]:
    if not dashboard.EXAMPLE_ID.match(example_id):
        raise _not_found("Example not found")
    example = next((e for e in dataset_examples() if e.id == example_id), None)
    if example is None:
        raise _not_found("Example not found")
    return {
        **dashboard.dataset_row(example),
        "expected_facts": example.metadata.expected_facts,
        "evidence": [e.model_dump() for e in example.metadata.evidence],
    }


@router.get("/history")
def history(request: Request) -> dict[str, Any]:
    runs = dashboard.list_runs(reports_dir(request))
    return {"runs": runs, "count": len(runs)}


@router.get("/runs/{run_id}")
def run_detail(run_id: str, request: Request) -> dict[str, Any]:
    try:
        return dashboard.run_detail(reports_dir(request), run_id, dataset_examples())
    except dashboard.ReportNotFoundError:
        raise _not_found() from None


@router.get("/runs/{run_id}/examples/{example_id}")
def run_example(run_id: str, example_id: str, request: Request) -> dict[str, Any]:
    try:
        return dashboard.example_detail(reports_dir(request), run_id, example_id, dataset_examples())
    except dashboard.ReportNotFoundError:
        raise _not_found("Result not found") from None


@router.get("/compare")
def compare(
    request: Request, base: str = Query(..., max_length=200), target: str = Query(..., max_length=200)
) -> dict[str, Any]:
    if base == target:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Choose two different runs to compare.")
    try:
        return dashboard.compare_runs(reports_dir(request), base, target)
    except dashboard.ReportNotFoundError:
        raise _not_found() from None


@router.get("/jobs/{job_id}")
def job_status(job_id: str, manager: EvalRunManager = Depends(get_eval_manager)) -> dict[str, Any]:
    job = manager.get(job_id) if len(job_id) == 32 and job_id.isalnum() else None
    if job is None:
        raise _not_found("Evaluation job not found")
    return job.public()


@router.post("/preview")
def preview(body: PreviewRequest) -> dict[str, Any]:
    """What a run would do and cost. Safe: reads the dataset file only, makes no external call."""
    chosen = resolve_selection(dataset_examples(), body.scope, body.example_ids)
    return preview_payload(chosen, body.mode)


# --------------------------------------------------------------------------- the only action with external calls
@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run(body: RunRequest, request: Request, manager: EvalRunManager = Depends(get_eval_manager)) -> JSONResponse:
    """Start an evaluation in the background. Requires confirm_gemini_usage, and for LangSmith mode also
    confirm_langsmith_upload; `acknowledged_max_calls` must equal the current estimate shown to the user.
    202 with the new job; 200 with the existing job for a repeated request_id; 409 while another run is active."""
    if not body.confirm_gemini_usage:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Confirm that this run may incur Gemini usage.")
    if body.mode == "langsmith":
        if not body.confirm_langsmith_upload:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Confirm that evaluation data will be sent to LangSmith.")
        if not langsmith_key_present():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "LangSmith is not configured on the server. Use local mode instead."
            )
    chosen = resolve_selection(dataset_examples(), body.scope, body.example_ids)
    expected = estimate_calls(len(chosen))["max_total"]
    if body.acknowledged_max_calls != expected:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"The call estimate is {expected}, not {body.acknowledged_max_calls}. "
            "Review the estimate and confirm again.",
        )
    retriever = None
    if manager.target_factory is None:  # reuse the app's retriever (no second embedding model in memory)
        components = await run_in_threadpool(get_components, request)
        retriever = components.retriever
    try:
        job, created = manager.submit([e.id for e in chosen], body.mode, body.request_id, retriever)
    except RunInProgressError as exc:
        return JSONResponse(
            {"detail": "An evaluation is already running. Wait for it to finish.", "job": exc.job.public()},
            status_code=status.HTTP_409_CONFLICT,
        )
    payload = {**job.public(), "duplicate": not created, "max_gemini_calls": expected}
    return JSONResponse(payload, status_code=status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK)
