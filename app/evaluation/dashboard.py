"""Read-only views for the Evals dashboard: the dataset, local reports, run summaries and run comparisons.

Nothing here makes a network call or runs an evaluation; it only reads the version-controlled dataset and the
report folders written by `app.evaluation.cli.execute_run` (evaluation/reports/<stamp>-<mode>-<dataset>/).

Report ids are the folder names. They are accepted only if they match REPORT_ID, and the resolved folder must
be a direct child of the reports folder, so a browser can never name another path.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from app.evaluation.dataset import DATASET_NAME, EvalExample
from app.evaluation.evaluators import METRICS, PASS_THRESHOLD
from app.evaluation.report import ExampleRun, build_report

logger = logging.getLogger(__name__)

REPORT_ID = re.compile(r"^\d{8}T\d{6}Z-(local|langsmith)-[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
EXAMPLE_ID = re.compile(r"^q\d{2,3}$")
MAX_REPORT_BYTES = 20 * 1024 * 1024
METRIC_INFO = {
    "faithfulness": ("Faithfulness", "Claims in the answer are supported by the retrieved excerpts."),
    "answer_relevance": ("Answer relevance", "The answer responds to the question that was asked."),
    "retrieval_relevance": ("Retrieval relevance", "Share of retrieved excerpts that help answer the question."),
    "correctness": ("Correctness", "Agreement with the verified reference answer (or a correct abstention)."),
}
CATEGORY_LABELS = {
    "direct_factual": "Direct factual",
    "multi_chunk": "Multi-chunk",
    "multi_document": "Multi-document",
    "scenario": "Scenario",
    "careful_grounding": "Careful grounding",
    "unanswerable": "Unanswerable",
}


class ReportNotFoundError(LookupError):
    pass


# --------------------------------------------------------------------------- helpers
def pct(score: float | None) -> float | None:
    """A 0-1 score as a percentage rounded to one decimal, or None."""
    return None if score is None else round(score * 100, 1)


def safe_url(url: Any) -> str | None:
    """`url` if it is an https URL with a host, else None (only such links are shown)."""
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    return url if parsed.scheme == "https" and parsed.netloc else None


def is_valid_report_id(run_id: str) -> bool:
    return bool(REPORT_ID.match(run_id))


def example_status(run: ExampleRun) -> str:
    """ "error" if the answer could not be generated or any metric could not be scored (an evaluation failure),
    "failed" if any metric scored below the pass threshold (a low-quality answer), else "passed"."""
    if run.target_error or any(m.error for m in run.metrics.values()):
        return "error"
    if any(m.passed is False for m in run.metrics.values()):
        return "failed"
    return "passed"


# --------------------------------------------------------------------------- dataset
def dataset_row(example: EvalExample) -> dict[str, Any]:
    meta = example.metadata
    return {
        "example_id": example.id,
        "question": example.inputs.question,
        "category": meta.category,
        "category_label": CATEGORY_LABELS.get(meta.category, meta.category),
        "expected_sources": list(meta.expected_sources),
        "reference_answer": example.outputs.reference_answer,
        "answerable": not meta.is_unanswerable,
    }


def dataset_rows(examples: list[EvalExample], query: str = "", category: str = "") -> list[dict[str, Any]]:
    """Dataset rows, optionally filtered by a case-insensitive search (id, question, reference) and category."""
    needle = query.strip().casefold()
    rows = []
    for example in examples:
        row = dataset_row(example)
        if category and row["category"] != category:
            continue
        haystack = f"{row['example_id']} {row['question']} {row['reference_answer']}".casefold()
        if needle and needle not in haystack:
            continue
        rows.append(row)
    return rows


def categories(examples: list[EvalExample]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    for example in examples:
        seen[example.metadata.category] = seen.get(example.metadata.category, 0) + 1
    return [{"key": k, "label": CATEGORY_LABELS.get(k, k), "count": n} for k, n in seen.items()]


# --------------------------------------------------------------------------- reports
def report_folder(reports_dir: Path, run_id: str) -> Path:
    """The folder of a valid report id, inside `reports_dir`; ReportNotFoundError otherwise."""
    if not is_valid_report_id(run_id):
        raise ReportNotFoundError(run_id)
    root = reports_dir.resolve()
    folder = (root / run_id).resolve()
    if folder.parent != root or not (folder / "report.json").is_file():
        raise ReportNotFoundError(run_id)
    return folder


def load_runs(reports_dir: Path, run_id: str) -> tuple[dict[str, Any], list[ExampleRun]]:
    """The report's info and its validated example runs. Unreadable or malformed reports are 'not found'."""
    path = report_folder(reports_dir, run_id) / "report.json"
    try:
        if path.stat().st_size > MAX_REPORT_BYTES:
            raise ValueError("report too large")
        data = json.loads(path.read_text(encoding="utf-8"))
        info = data.get("info") if isinstance(data, dict) else None
        examples = data.get("examples") if isinstance(data, dict) else None
        if not isinstance(info, dict) or not isinstance(examples, list):
            raise ValueError("unexpected report layout")
        runs = [ExampleRun.model_validate(e) for e in examples]
    except (OSError, ValueError, ValidationError) as exc:  # json.JSONDecodeError is a ValueError
        logger.warning("Evaluation report %s could not be read (%s)", run_id, type(exc).__name__)
        raise ReportNotFoundError(run_id) from None
    return info, runs


def _created_label(created_at: Any) -> str:
    try:
        moment = datetime.fromisoformat(str(created_at))
    except ValueError:
        return str(created_at or "")
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def run_summary(run_id: str, info: dict[str, Any], runs: list[ExampleRun]) -> dict[str, Any]:
    """Scores (in %) and pass / fail / error counts for one report, recomputed from its example runs."""
    summary = build_report(runs, info)["summary"]
    statuses = [example_status(r) for r in runs]
    metrics = {}
    for key in METRICS:
        s = summary[key]
        label, description = METRIC_INFO[key]
        metrics[key] = {
            "key": key,
            "label": label,
            "description": description,
            "score": pct(s["mean"]),
            "scored": s["scored"],
            "passed": s["passed"],
            "failed": s["failed"],
            "errors": s["errors"],
        }
    means = [m["score"] for m in metrics.values() if m["score"] is not None]
    return {
        "id": run_id,
        "created_at": str(info.get("created_at") or ""),
        "created_label": _created_label(info.get("created_at")),
        "mode": "langsmith" if info.get("mode") == "langsmith" else "local",
        "dataset": str(info.get("dataset") or DATASET_NAME),
        "judge_model": str(info.get("judge_model") or ""),
        "prompt_version": str(info.get("prompt_version") or ""),
        "pass_threshold": pct(float(info.get("pass_threshold") or PASS_THRESHOLD)),
        "experiment_url": safe_url(info.get("experiment_url")),
        "examples_total": len(runs),
        "examples_passed": statuses.count("passed"),
        "examples_failed": statuses.count("failed"),
        "examples_errored": statuses.count("error"),
        "target_errors": sum(r.target_error is not None for r in runs),
        "metric_errors": sum(m.error is not None for r in runs for m in r.metrics.values()),
        "overall": round(sum(means) / len(means), 1) if means else None,
        "metrics": metrics,
        "example_ids": [r.example_id for r in runs],
    }


def list_runs(reports_dir: Path) -> list[dict[str, Any]]:
    """Summaries of every readable report, newest first. Folders with other names are ignored."""
    if not reports_dir.is_dir():
        return []
    summaries = []
    for folder in sorted(reports_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not folder.is_dir() or not is_valid_report_id(folder.name):
            continue
        try:
            info, runs = load_runs(reports_dir, folder.name)
        except ReportNotFoundError:
            continue
        summaries.append(run_summary(folder.name, info, runs))
    return summaries


def dashboard_summary(reports_dir: Path, dataset_size: int) -> dict[str, Any]:
    """The latest run with each metric's change against the run before it (None when there are no runs)."""
    history = list_runs(reports_dir)
    latest = history[0] if history else None
    previous = history[1] if len(history) > 1 else None
    cards = []
    for key in METRICS:
        label, description = METRIC_INFO[key]
        current = latest["metrics"][key] if latest else None
        before = previous["metrics"][key]["score"] if previous else None
        score = current["score"] if current else None
        cards.append(
            {
                "key": key,
                "label": label,
                "description": description,
                "score": score,
                "scored": current["scored"] if current else 0,
                "errors": current["errors"] if current else 0,
                "previous": before,
                "delta": round(score - before, 1) if score is not None and before is not None else None,
            }
        )
    return {
        "dataset": {"name": DATASET_NAME, "size": dataset_size},
        "latest": latest,
        "previous": previous,
        "metrics": cards,
        "run_count": len(history),
    }


def _reference_answers(examples: list[EvalExample]) -> dict[str, str]:
    return {e.id: e.outputs.reference_answer for e in examples}


def _score_cells(run: ExampleRun) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for key in METRICS:
        m = run.metrics.get(key)
        cells[key] = {
            "score": pct(m.score) if m else None,
            "passed": m.passed if m else None,
            "error": (m.error if m else "no result recorded"),
            "explanation": m.explanation if m else "",
            "label": METRIC_INFO[key][0],
        }
    return cells


def run_rows(runs: list[ExampleRun], examples: list[EvalExample]) -> list[dict[str, Any]]:
    """One table row per evaluated example. Reference answers come from the current dataset file."""
    references = _reference_answers(examples)
    return [
        {
            "example_id": r.example_id,
            "category": r.category,
            "category_label": CATEGORY_LABELS.get(r.category, r.category),
            "question": r.question,
            "answer": str((r.outputs or {}).get("answer") or ""),
            "reference_answer": references.get(r.example_id, ""),
            "target_error": r.target_error,
            "scores": _score_cells(r),
            "status": example_status(r),
        }
        for r in runs
    ]


def run_detail(reports_dir: Path, run_id: str, examples: list[EvalExample]) -> dict[str, Any]:
    info, runs = load_runs(reports_dir, run_id)
    return {"run": run_summary(run_id, info, runs), "examples": run_rows(runs, examples)}


def example_detail(reports_dir: Path, run_id: str, example_id: str, examples: list[EvalExample]) -> dict[str, Any]:
    """Everything recorded for one example of one run: answer, contexts, citation check and each metric."""
    if not EXAMPLE_ID.match(example_id):
        raise ReportNotFoundError(example_id)
    info, runs = load_runs(reports_dir, run_id)
    run = next((r for r in runs if r.example_id == example_id), None)
    if run is None:
        raise ReportNotFoundError(example_id)
    row = run_rows([run], examples)[0]
    outputs = run.outputs or {}
    contexts = [c for c in outputs.get("contexts") or [] if isinstance(c, dict)]
    raw_cites = outputs.get("citations")
    cites: dict[str, Any] = raw_cites if isinstance(raw_cites, dict) else {}
    return {
        **row,
        "run_id": run_id,
        "run_label": _created_label(info.get("created_at")),
        "expected_sources": run.expected_sources,
        "answerable": not run.is_unanswerable,
        "sources": sorted({str(c.get("source", "")) for c in contexts if c.get("source")}),
        "contexts": [
            {
                "number": c.get("number", i),
                "source": str(c.get("source", "")),
                "score": c.get("score") if isinstance(c.get("score"), int | float) else None,
                "text": str(c.get("text", "")),
            }
            for i, c in enumerate(contexts, start=1)
        ],
        "citations": {
            "cited": [n for n in cites.get("cited") or [] if isinstance(n, int)],
            "invalid": [n for n in cites.get("invalid") or [] if isinstance(n, int)],
            "has_contexts": bool(contexts),
        },
    }


def compare_runs(reports_dir: Path, base_id: str, target_id: str) -> dict[str, Any]:
    """Metric changes from `base_id` to `target_id`, and how examples evaluated in both runs moved."""
    base_info, base_runs = load_runs(reports_dir, base_id)
    target_info, target_runs = load_runs(reports_dir, target_id)
    base, target = run_summary(base_id, base_info, base_runs), run_summary(target_id, target_info, target_runs)
    metrics = []
    for key in (*METRICS, "overall"):
        before = base["overall"] if key == "overall" else base["metrics"][key]["score"]
        after = target["overall"] if key == "overall" else target["metrics"][key]["score"]
        delta = round(after - before, 1) if before is not None and after is not None else None
        direction = "unknown" if delta is None else "improved" if delta > 0 else "declined" if delta < 0 else "same"
        label = "Overall" if key == "overall" else METRIC_INFO[key][0]
        metrics.append(
            {"key": key, "label": label, "base": before, "target": after, "delta": delta, "direction": direction}
        )
    before_status = {r.example_id: (example_status(r), r.question) for r in base_runs}
    after_status = {r.example_id: (example_status(r), r.question) for r in target_runs}
    shared = sorted(set(before_status) & set(after_status))

    def pick(rule: Any) -> list[dict[str, str]]:
        return [
            {
                "example_id": i,
                "question": after_status[i][1],
                "before": before_status[i][0],
                "after": after_status[i][0],
            }
            for i in shared
            if rule(before_status[i][0], after_status[i][0])
        ]

    bad = {"failed", "error"}
    return {
        "base": base,
        "target": target,
        "metrics": metrics,
        "same_examples": set(before_status) == set(after_status),
        "shared_count": len(shared),
        "failed_in_both": pick(lambda a, b: a in bad and b in bad),
        "fixed": pick(lambda a, b: a in bad and b == "passed"),
        "regressed": pick(lambda a, b: a == "passed" and b in bad),
        "only_in_base": sorted(set(before_status) - set(after_status)),
        "only_in_target": sorted(set(after_status) - set(before_status)),
    }
