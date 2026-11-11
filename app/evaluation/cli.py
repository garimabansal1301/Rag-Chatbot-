"""Command-line entry points (scripts/create_evaluation_dataset.py, scripts/run_evaluation.py).

Both commands are DRY RUNS unless --confirm is given: they validate the dataset and print what a live run
would send and roughly how many Gemini calls it would make, without contacting LangSmith or Gemini.
API keys are only checked for presence and never printed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, get_settings
from app.evaluation.dataset import (
    DATASET_DESCRIPTION,
    DATASET_NAME,
    DATASET_PATH,
    DatasetError,
    EvalExample,
    check_evidence,
    load_dataset,
    select,
)
from app.evaluation.evaluators import METRICS, langsmith_evaluators
from app.evaluation.judge import JUDGE_TIMEOUT_SECONDS, Judge, build_judge
from app.evaluation.prompts import PROMPT_VERSION
from app.evaluation.report import REPORTS_DIR, build_report, run_local, runs_from_experiment, write_report
from app.evaluation.target import build_target

DEFAULT_PROJECT = "RAG-Chatbot"
EXPERIMENT_PREFIX = "rag-chatbot-eval"
TARGET_CALLS_PER_EXAMPLE = 1  # one generate call: the evaluation engine has no tools, so no tool rounds
JUDGE_CALLS_PER_EXAMPLE = len(METRICS)  # at most; skipped for empty answers, empty retrieval and abstentions
WARNING = "This command makes EXTERNAL calls{what}. It may incur Gemini usage. Nothing is sent without --confirm."

ClientFactory = Callable[[], Any]


def _default_client() -> Any:
    from langsmith import Client

    return Client()


def _api_key_present() -> bool:
    return any(os.getenv(name, "").strip() for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"))


def langsmith_key_present() -> bool:
    """Whether a LangSmith API key is set (checked for presence only; the value is never read out)."""
    return _api_key_present()


def _project() -> str:
    return os.getenv("LANGSMITH_PROJECT", "").strip() or DEFAULT_PROJECT


def _load(args: argparse.Namespace) -> list[EvalExample]:
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
    return load_examples(Path(args.dataset_file), ids, args.limit)


def load_examples(
    path: Path = DATASET_PATH, ids: list[str] | None = None, limit: int | None = None
) -> list[EvalExample]:
    """The validated dataset (checked against the local documents when present), narrowed to `ids` / `limit`."""
    examples = load_dataset(path)
    documents = _documents()
    if documents:
        problems = check_evidence(examples, documents)
        if problems:
            raise DatasetError("Reference answers do not match the documents:\n  " + "\n  ".join(problems))
    return select(examples, ids, limit)


def estimate_calls(n: int) -> dict[str, int]:
    """Upper bound on Gemini calls for `n` examples: one answer call plus at most one judge call per metric."""
    target, judge = n * TARGET_CALLS_PER_EXAMPLE, n * JUDGE_CALLS_PER_EXAMPLE
    return {"answer_calls": target, "judge_calls": judge, "max_total": target + judge}


def _documents() -> dict[str, str]:
    """The knowledge-base documents' text (local only, used to verify the evidence quotes)."""
    from app.ingest.pipeline import document_from_bytes

    folder = get_settings().documents_dir
    if not folder.is_dir():
        return {}
    return {
        p.name: document_from_bytes(p.name, p.read_bytes()).text
        for p in sorted(folder.iterdir())
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in get_settings().allowed_upload_suffixes
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-file", default=str(DATASET_PATH), help="local JSONL dataset")
    parser.add_argument("--dataset-name", default=DATASET_NAME, help="LangSmith dataset name")
    parser.add_argument("--ids", help="comma-separated example ids, e.g. q01,q05")
    parser.add_argument("--limit", type=int, help="use at most this many examples")
    parser.add_argument("--confirm", action="store_true", help="really make the external calls")


# --------------------------------------------------------------------------- dataset creation
def dataset_plan(examples: list[EvalExample], name: str) -> str:
    payload = [e.to_langsmith() for e in examples]
    size = len(json.dumps(payload).encode("utf-8"))
    return (
        f"Dataset {name!r}: {len(examples)} example(s), about {size / 1024:.1f} KB.\n"
        "Sent to LangSmith: each question, verified reference answer, expected source file names, expected facts,\n"
        "category and short evidence quotes. No document files, no chat sessions, no keys. No Gemini calls."
    )


def create_dataset_main(argv: list[str] | None = None, client_factory: ClientFactory = _default_client) -> int:
    parser = argparse.ArgumentParser(
        prog="create_evaluation_dataset", description=WARNING.format(what=" to LangSmith (dataset upload)")
    )
    _common(parser)
    args = parser.parse_args(argv)
    try:
        examples = _load(args)
    except DatasetError as exc:
        print(exc)
        return 1
    print(WARNING.format(what=" to LangSmith (dataset upload)"))
    print(dataset_plan(examples, args.dataset_name))
    if not args.confirm:
        print("Dry run: nothing was sent. Re-run with --confirm to upload.")
        return 0
    if not _api_key_present():
        print("LANGSMITH_API_KEY is not set; nothing was sent.")
        return 2
    client = client_factory()
    created, skipped, conflicts = upload_examples(client, args.dataset_name, examples)
    print(f"Uploaded {created} new example(s); {skipped} already present; {len(conflicts)} conflict(s).")
    for conflict in conflicts:
        print(f"  {conflict}: differs from the copy in LangSmith; not overwritten (use a new dataset version).")
    return 3 if conflicts else 0


def upload_examples(client: Any, name: str, examples: list[EvalExample]) -> tuple[int, int, list[str]]:
    """Create the dataset if needed and add only the examples it does not have yet (matched by example_id).
    An existing example with different content is reported as a conflict and never overwritten."""
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        existing = {}
        for remote in client.list_examples(dataset_id=dataset.id):
            example_id = (remote.metadata or {}).get("example_id")
            if example_id:
                existing[example_id] = remote
    else:
        dataset = client.create_dataset(
            name, description=DATASET_DESCRIPTION, metadata={"source": "evaluation/datasets (local JSONL)"}
        )
        existing = {}
    new, skipped, conflicts = [], 0, []
    for example in examples:
        remote = existing.get(example.id)
        if remote is None:
            new.append(example.to_langsmith())
        elif (
            dict(remote.inputs) == example.inputs.model_dump()
            and dict(remote.outputs or {}) == example.outputs.model_dump()
        ):
            skipped += 1
        else:
            conflicts.append(example.id)
    if new:
        client.create_examples(dataset_id=dataset.id, examples=new)
    return len(new), skipped, conflicts


# --------------------------------------------------------------------------- experiments
def run_plan(examples: list[EvalExample], local: bool) -> str:
    calls = estimate_calls(len(examples))
    n, target, judge = len(examples), calls["answer_calls"], calls["judge_calls"]
    settings = get_settings()
    lines = [
        f"Examples: {n}. Expected Gemini calls: {target} answer call(s) + at most {judge} judge call(s) "
        f"= at most {target + judge} (model {settings.llm_model or '(LLM_MODEL not set)'}). Embeddings are local.",
        f"Each answer call sends the question and up to {settings.top_k} retrieved excerpts (about "
        f"{settings.chunk_size} characters each) to Gemini; judge calls send the question, the answer and the "
        "excerpts or the reference answer.",
    ]
    if local:
        lines.append("Mode --local: results stay on this machine; LangSmith tracing is switched off for this run.")
    else:
        lines.append(
            f"Sent to LangSmith (experiment under dataset, project {_project()!r}): per example the question, the "
            "answer, the retrieved excerpts (not whole documents), the traced prompts, and the four scores with "
            "explanations. No chat sessions, uploads or keys."
        )
    return "\n".join(lines)


def run_evaluation_main(
    argv: list[str] | None = None,
    client_factory: ClientFactory = _default_client,
    target_factory: Callable[[], Callable[[dict[str, Any]], dict[str, Any]]] = build_target,
    judge_factory: Callable[[str | None, float], Judge] = build_judge,
    reports_dir: Path = REPORTS_DIR,
) -> int:
    parser = argparse.ArgumentParser(
        prog="run_evaluation", description=WARNING.format(what=" to Gemini and (unless --local) LangSmith")
    )
    _common(parser)
    parser.add_argument("--local", action="store_true", help="evaluate locally; send nothing to LangSmith")
    parser.add_argument("--judge-model", help="Gemini model for the judge (default: the app's LLM_MODEL)")
    parser.add_argument("--judge-timeout", type=float, default=JUDGE_TIMEOUT_SECONDS)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--experiment-prefix", default=EXPERIMENT_PREFIX)
    args = parser.parse_args(argv)
    try:
        examples = _load(args)
    except DatasetError as exc:
        print(exc)
        return 1
    print(WARNING.format(what=" to Gemini" + ("" if args.local else " and LangSmith")))
    print(run_plan(examples, args.local))
    if not args.confirm:
        print("Dry run: no Gemini or LangSmith call was made. Re-run with --confirm to evaluate.")
        return 0
    if not args.local and not _api_key_present():
        print("LANGSMITH_API_KEY is not set; nothing was run. Use --local to evaluate without LangSmith.")
        return 2
    report, folder = execute_run(
        examples,
        local=args.local,
        dataset_name=args.dataset_name,
        judge_model=args.judge_model,
        judge_timeout=args.judge_timeout,
        max_concurrency=args.max_concurrency,
        experiment_prefix=args.experiment_prefix,
        client_factory=client_factory,
        target_factory=target_factory,
        judge_factory=judge_factory,
        reports_dir=reports_dir,
    )
    _print_summary(report, folder)
    return 0


def execute_run(
    examples: list[EvalExample],
    *,
    local: bool,
    dataset_name: str = DATASET_NAME,
    judge_model: str | None = None,
    judge_timeout: float = JUDGE_TIMEOUT_SECONDS,
    max_concurrency: int = 1,
    experiment_prefix: str = EXPERIMENT_PREFIX,
    client_factory: ClientFactory = _default_client,
    target_factory: Callable[[], Callable[[dict[str, Any]], dict[str, Any]]] = build_target,
    judge_factory: Callable[[str | None, float], Judge] = build_judge,
    reports_dir: Path = REPORTS_DIR,
    process_wide_tracing_off: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Run a confirmed evaluation (EXTERNAL CALLS) and write its report; returns the report and its folder.

    Shared by the CLI and the web dashboard. The CLI switches tracing off for the whole process in local mode;
    the web app passes `process_wide_tracing_off=False` so chat tracing is untouched (the run is still
    untraced through LangSmith's tracing context).
    """
    info: dict[str, Any] = {
        "dataset": dataset_name,
        "mode": "local" if local else "langsmith",
        "examples": [e.id for e in examples],
        "judge_model": judge_model or get_settings().llm_model,
        "created_at": _now(),
    }
    target, judge = target_factory(), judge_factory(judge_model, judge_timeout)
    if local:
        runs = _run_untraced(examples, target, judge, process_wide_tracing_off)
    else:
        runs, url = _run_langsmith(
            client_factory(), dataset_name, experiment_prefix, max_concurrency, examples, target, judge, info
        )
        info["experiment_url"] = url
    report = build_report(runs, info)
    folder = write_report(report, f"{info['mode']}-{dataset_name}", reports_dir)
    return report, folder


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _run_untraced(examples: list[EvalExample], target: Any, judge: Judge, process_wide: bool = True) -> Any:
    import langsmith

    from app.observability import TRACING_FLAGS, clear_langsmith_env_cache

    if process_wide:
        for name in TRACING_FLAGS:  # also covers LangChain's env-based tracer
            os.environ[name] = "false"
        clear_langsmith_env_cache()
    with langsmith.tracing_context(enabled=False):
        return run_local(examples, target, judge)


def _run_langsmith(
    client: Any,
    dataset_name: str,
    experiment_prefix: str,
    max_concurrency: int,
    examples: list[EvalExample],
    target: Any,
    judge: Judge,
    info: dict[str, Any],
) -> tuple[Any, str | None]:
    if not client.has_dataset(dataset_name=dataset_name):
        raise SystemExit(
            f"LangSmith dataset {dataset_name!r} does not exist. Create it first with "
            "scripts/create_evaluation_dataset.py --confirm."
        )
    wanted = {e.id: e for e in examples}
    remote = [
        r for r in client.list_examples(dataset_name=dataset_name) if (r.metadata or {}).get("example_id") in wanted
    ]
    missing = sorted(set(wanted) - {r.metadata["example_id"] for r in remote})
    if missing:
        print(f"Not in the LangSmith dataset (skipped): {', '.join(missing)}")
    results = client.evaluate(
        target,
        data=remote,
        evaluators=langsmith_evaluators(judge),
        experiment_prefix=experiment_prefix,
        description="Offline RAG evaluation: faithfulness, answer relevance, retrieval relevance, correctness.",
        metadata={
            "dataset_version": dataset_name,
            "prompt_version": PROMPT_VERSION,
            "judge_model": info["judge_model"],
            "app_project": _project(),
            "example_count": len(remote),
        },
        max_concurrency=max_concurrency,
    )
    try:
        url = results.url
    except Exception:
        url = None
    return runs_from_experiment(list(results), wanted), url


def _print_summary(report: dict[str, Any], folder: Path) -> None:
    print(
        f"Examples: {report['examples_total']}, passed {report['examples_passed']}, failed "
        f"{report['examples_failed']}, evaluation errors {report['evaluation_errors']}"
    )
    for key, s in report["summary"].items():
        mean = "n/a" if s["mean"] is None else f"{s['mean']:.2f}"
        print(f"  {key:<20} mean {mean}  passed {s['passed']}  failed {s['failed']}  errors {s['errors']}")
    if report["info"].get("experiment_url"):
        print(f"Experiment: {report['info']['experiment_url']}")
    print(f"Report: {folder.relative_to(PROJECT_ROOT) if folder.is_relative_to(PROJECT_ROOT) else folder}")
