"""Running examples locally, collecting LangSmith experiment rows, and writing local reports.

Reports go to evaluation/reports/<timestamp>-<name>/ (report.json and report.md), separate from chat
sessions, reviews and checkpoints. They contain questions, answers, scores and explanations, never keys.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT
from app.evaluation.dataset import EvalExample
from app.evaluation.evaluators import METRICS, PASS_THRESHOLD, MetricResult, abstains, evaluate_example
from app.evaluation.judge import Judge
from app.evaluation.prompts import PROMPT_VERSION

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
MAX_ERROR_CHARS = 300


class ExampleRun(BaseModel):
    example_id: str
    category: str
    question: str
    is_unanswerable: bool
    expected_sources: list[str] = Field(default_factory=list)
    outputs: dict[str, Any] | None = None
    target_error: str | None = None
    metrics: dict[str, MetricResult] = Field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.target_error is not None or any(m.passed is not True for m in self.metrics.values())


def _base(example: EvalExample) -> dict[str, Any]:
    return {
        "example_id": example.id,
        "category": example.metadata.category,
        "question": example.inputs.question,
        "is_unanswerable": example.metadata.is_unanswerable,
        "expected_sources": example.metadata.expected_sources,
    }


def run_local(
    examples: list[EvalExample], target: Callable[[dict[str, Any]], dict[str, Any]], judge: Judge
) -> list[ExampleRun]:
    """Run the target and the four evaluators on each example, sequentially. A failing target is recorded as
    a target error (every metric then is an error), never as a score."""
    runs = []
    for example in examples:
        try:
            outputs = target(example.inputs.model_dump())
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:MAX_ERROR_CHARS]}"
            failed: dict[str, MetricResult] = {
                k: MetricResult(key=k, error=f"target failed ({type(exc).__name__})") for k in METRICS
            }
            runs.append(ExampleRun(**_base(example), target_error=error, metrics=failed))
            continue
        metrics: dict[str, MetricResult] = dict(evaluate_example(example, outputs, judge).items())
        runs.append(ExampleRun(**_base(example), outputs=outputs, metrics=metrics))
    return runs


def runs_from_experiment(rows: Iterable[Any], examples: dict[str, EvalExample]) -> list[ExampleRun]:
    """ExampleRuns from LangSmith ExperimentResults rows ({"run", "example", "evaluation_results"})."""
    runs = []
    for row in rows:
        run, example = row["run"], row["example"]
        example_id = str((example.metadata or {}).get("example_id", ""))
        if example_id not in examples:
            continue
        metrics: dict[str, MetricResult] = {}
        for result in row["evaluation_results"]["results"]:
            if result.key not in METRICS:
                continue
            if result.value == "error" or result.score is None:
                metrics[result.key] = MetricResult(key=result.key, error=str(result.comment or "evaluator error"))
            else:
                metrics[result.key] = MetricResult(
                    key=result.key, score=float(result.score), explanation=str(result.comment or "")
                )
        for key in METRICS:
            metrics.setdefault(key, MetricResult(key=key, error="no result recorded by LangSmith"))
        error = str(run.error)[:MAX_ERROR_CHARS] if getattr(run, "error", None) else None
        runs.append(
            ExampleRun(**_base(examples[example_id]), outputs=run.outputs or None, target_error=error, metrics=metrics)
        )
    return sorted(runs, key=lambda r: r.example_id)


# --------------------------------------------------------------------------- report
def _run_patterns(r: ExampleRun) -> list[str]:
    """The failure patterns one example shows (deterministic rules on its outputs, metadata and scores)."""
    if r.target_error:
        return ["target run failed"]
    out, m = r.outputs or {}, r.metrics
    answer, contexts = str(out.get("answer") or ""), out.get("contexts") or []
    cites = out.get("citations") or {}
    retrieved = {c.get("source") for c in contexts}
    checks = {
        "answered an unanswerable question": r.is_unanswerable
        and "correctness" in m
        and m["correctness"].passed is False,
        "said the documents do not cover an answerable question": not r.is_unanswerable and abstains(answer),
        "nothing retrieved for an answerable question": not r.is_unanswerable and not contexts,
        "an expected source document was not retrieved": bool(contexts)
        and any(s not in retrieved for s in r.expected_sources),
        "cites excerpt numbers that do not exist": bool(cites.get("invalid")),
        "no citation although excerpts were retrieved": bool(contexts)
        and not cites.get("cited")
        and not abstains(answer),
        "claims not supported by the retrieved context": "faithfulness" in m and m["faithfulness"].passed is False,
    }
    found = [name for name, hit in checks.items() if hit]
    found += [f"evaluation error: {x.error.split(':')[0][:60]}" for x in m.values() if x.error]
    return found


def _patterns(runs: list[ExampleRun]) -> list[dict[str, Any]]:
    """Common failure patterns across examples, most frequent first."""
    found: dict[str, list[str]] = defaultdict(list)
    for r in runs:
        for name in _run_patterns(r):
            found[name].append(r.example_id)
    return [
        {"pattern": name, "count": len(set(ids)), "examples": sorted(set(ids))}
        for name, ids in sorted(found.items(), key=lambda item: (-len(set(item[1])), item[0]))
    ]


def build_report(runs: list[ExampleRun], info: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key in METRICS:
        results = [r.metrics[key] for r in runs if key in r.metrics]
        scores = [m.score for m in results if m.score is not None]
        summary[key] = {
            "mean": round(sum(scores) / len(scores), 4) if scores else None,
            "scored": len(scores),
            "passed": sum(m.passed is True for m in results),
            "failed": sum(m.passed is False for m in results),
            "errors": sum(m.error is not None for m in results),
        }
    failed = [r for r in runs if r.failed]
    by_category = Counter(r.category for r in failed)
    return {
        "info": {**info, "prompt_version": PROMPT_VERSION, "pass_threshold": PASS_THRESHOLD},
        "summary": summary,
        "examples_total": len(runs),
        "examples_passed": len(runs) - len(failed),
        "examples_failed": len(failed),
        "target_errors": sum(r.target_error is not None for r in runs),
        "evaluation_errors": sum(m.error is not None for r in runs for m in r.metrics.values()),
        "failed_by_category": dict(sorted(by_category.items())),
        "failed_examples": [r.example_id for r in failed],
        "failure_patterns": _patterns(runs),
        "examples": [r.model_dump(mode="json") for r in runs],
    }


def _markdown(report: dict[str, Any]) -> str:
    info = report["info"]
    lines = [
        f"# Evaluation report: {info.get('dataset', '')}",
        "",
        f"- Created: {info.get('created_at', '')} · mode: {info.get('mode', '')} · prompts: {info['prompt_version']}",
        f"- Examples: {report['examples_total']} · passed: {report['examples_passed']} · failed: "
        f"{report['examples_failed']} · target errors: {report['target_errors']} · evaluation errors: "
        f"{report['evaluation_errors']} (errors are not counted as scores)",
    ]
    if info.get("experiment_url"):
        lines.append(f"- LangSmith experiment: {info['experiment_url']}")
    lines += [
        "",
        "## Metrics",
        "",
        f"| Metric | Mean | Passed (≥ {info['pass_threshold']}) | Failed | Errors |",
        "|---|---|---|---|---|",
    ]
    for key, s in report["summary"].items():
        mean = "—" if s["mean"] is None else f"{s['mean']:.2f}"
        lines.append(f"| {key} | {mean} | {s['passed']} | {s['failed']} | {s['errors']} |")
    lines += ["", "## Failure patterns", ""]
    lines += [f"- {p['pattern']}: {p['count']} ({', '.join(p['examples'])})" for p in report["failure_patterns"]] or [
        "- none"
    ]
    lines += ["", "## Examples", ""]
    for example in report["examples"]:
        status = "FAILED" if example["example_id"] in report["failed_examples"] else "passed"
        lines += [
            f"### {example['example_id']} ({example['category']}) — {status}",
            "",
            f"**Question:** {example['question']}",
            "",
        ]
        if example["target_error"]:
            lines += [f"**Target error:** {example['target_error']}", ""]
        elif example["outputs"]:
            lines += [
                f"**Answer:** {example['outputs'].get('answer', '')}",
                "",
                f"**Retrieved sources:** {', '.join(example['outputs'].get('sources') or []) or 'none'}",
                "",
            ]
        for key in METRICS:
            m = example["metrics"].get(key)
            if m is None:
                continue
            value = f"ERROR — {m['error']}" if m["error"] else f"{m['score']:.2f} — {m['explanation']}"
            lines.append(f"- **{key}:** {value}")
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Write report.json and report.md into a new folder; returns the folder."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = reports_dir / f"{stamp}-{name}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (folder / "report.md").write_text(_markdown(report), encoding="utf-8")
    return folder
