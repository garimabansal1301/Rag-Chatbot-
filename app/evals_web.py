"""The Evals dashboard page and its HTMX partials (Jinja2). Read-only: rendering never calls Gemini or LangSmith.

Runs are started from the page through POST /api/evals/run (app/api/evals.py), after the user's confirmations.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse

from app.api.evals import dataset_examples, get_eval_manager, reports_dir
from app.deps import get_store
from app.evaluation import dashboard
from app.evaluation.cli import JUDGE_CALLS_PER_EXAMPLE, TARGET_CALLS_PER_EXAMPLE, langsmith_key_present
from app.evaluation.dataset import DATASET_NAME
from app.evaluation.evaluators import PASS_THRESHOLD
from app.evaluation.runner import EvalJob, EvalRunManager
from app.storage.json_store import SessionStore
from app.web import templates

router = APIRouter(include_in_schema=False)
RUN_FINISHED_EVENT = "evals-run-finished"


def _results_context(request: Request, run_id: str | None) -> dict[str, Any]:
    """The results table of `run_id`, or of the newest readable run when None."""
    folder = reports_dir(request)
    if run_id is None:
        newest = dashboard.list_runs(folder)
        if not newest:
            return {"results": None}
        run_id = newest[0]["id"]
    try:
        return {"results": dashboard.run_detail(folder, run_id, dataset_examples())}
    except dashboard.ReportNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Evaluation run not found") from None


def _summary_context(request: Request) -> dict[str, Any]:
    return {"summary": dashboard.dashboard_summary(reports_dir(request), len(dataset_examples()))}


def _history_context(request: Request) -> dict[str, Any]:
    return {"history": dashboard.list_runs(reports_dir(request))}


@router.get("/evals", response_class=HTMLResponse)
def evals_page(
    request: Request,
    store: SessionStore = Depends(get_store),
    manager: EvalRunManager = Depends(get_eval_manager),
) -> HTMLResponse:
    examples = dataset_examples()
    return templates.TemplateResponse(
        request,
        "evals.html",
        {
            "sessions": store.list(),
            "current": None,
            "active_page": "evals",
            "dataset_name": DATASET_NAME,
            "dataset_rows": dashboard.dataset_rows(examples),
            "categories": dashboard.categories(examples),
            "langsmith_configured": langsmith_key_present(),
            "calls_per_example": TARGET_CALLS_PER_EXAMPLE + JUDGE_CALLS_PER_EXAMPLE,
            "pass_threshold": round(PASS_THRESHOLD * 100),
            "job": manager.latest(),
            **_summary_context(request),
            **_results_context(request, None),
            **_history_context(request),
        },
    )


@router.get("/ui/evals/summary", response_class=HTMLResponse)
def ui_summary(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/evals/summary.html", _summary_context(request))


@router.get("/ui/evals/history", response_class=HTMLResponse)
def ui_history(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/evals/history.html", _history_context(request))


@router.get("/ui/evals/results", response_class=HTMLResponse)
def ui_results(request: Request, run: str | None = Query(None, max_length=200)) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/evals/results.html", _results_context(request, run))


@router.get("/ui/evals/runs/{run_id}/examples/{example_id}", response_class=HTMLResponse)
def ui_example(run_id: str, example_id: str, request: Request) -> HTMLResponse:
    try:
        detail = dashboard.example_detail(reports_dir(request), run_id, example_id, dataset_examples())
    except dashboard.ReportNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Result not found") from None
    return templates.TemplateResponse(request, "partials/evals/detail.html", {"d": detail})


@router.get("/ui/evals/compare", response_class=HTMLResponse)
def ui_compare(
    request: Request, base: str = Query("", max_length=200), target: str = Query("", max_length=200)
) -> HTMLResponse:
    context: dict[str, Any] = {"comparison": None, "compare_error": None}
    if not base or not target or base == target:
        context["compare_error"] = "Choose two different runs to compare."
    else:
        try:
            context["comparison"] = dashboard.compare_runs(reports_dir(request), base, target)
        except dashboard.ReportNotFoundError:
            context["compare_error"] = "One of these runs could not be found."
    return templates.TemplateResponse(request, "partials/evals/compare.html", context)


@router.get("/ui/evals/status", response_class=HTMLResponse)
def ui_status(
    request: Request,
    job: str | None = Query(None, max_length=64),
    manager: EvalRunManager = Depends(get_eval_manager),
) -> HTMLResponse:
    """The run status banner. Polled while a run is active; when the polled job ends, the response fires
    RUN_FINISHED_EVENT so the summary, results and history reload."""
    current: EvalJob | None = manager.get(job) if job else manager.latest()
    headers = {"HX-Trigger": RUN_FINISHED_EVENT} if job and current and current.status != "running" else None
    return templates.TemplateResponse(request, "partials/evals/status.html", {"job": current}, headers=headers)
