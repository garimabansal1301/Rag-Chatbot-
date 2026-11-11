"""Runs a confirmed evaluation from the Evals dashboard in a background thread (EXTERNAL CALLS).

One run at a time per app process: a second submission while a run is active is refused, and a repeated
submission with the same `request_id` (a double click, a retried request) returns the run already started.
The work itself is `app.evaluation.cli.execute_run`, the same code the CLI uses, so local runs are untraced
(without touching the chat app's tracing settings) and LangSmith runs create a normal experiment.

Nothing here starts on its own: a run begins only from `EvalRunManager.submit`, which the API calls after
checking the user's explicit confirmations. Jobs are kept in memory; a restart forgets them (reports stay).
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.evaluation.cli import ClientFactory, _default_client, execute_run, load_examples
from app.evaluation.dataset import DATASET_NAME, DATASET_PATH, DatasetError
from app.evaluation.judge import Judge, build_judge
from app.evaluation.target import build_target
from app.vectordb.retriever import Retriever

logger = logging.getLogger(__name__)

Mode = Literal["local", "langsmith"]
JobStatus = Literal["running", "completed", "failed"]
Target = Callable[[dict[str, Any]], dict[str, Any]]
MAX_JOBS_KEPT = 20


class RunInProgressError(RuntimeError):
    def __init__(self, job: EvalJob) -> None:
        super().__init__("An evaluation is already running.")
        self.job = job


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class EvalJob:
    id: str
    request_id: str
    mode: Mode
    example_ids: list[str]
    status: JobStatus = "running"
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    answered: int = 0
    report_id: str | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        """What the browser may see: no paths, no stack traces, no configuration."""
        return {
            "job_id": self.id,
            "status": self.status,
            "mode": self.mode,
            "example_count": len(self.example_ids),
            "answered": min(self.answered, len(self.example_ids)),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "report_id": self.report_id,
            "error": self.error,
        }


class EvalRunManager:
    """Owns the single evaluation slot of an app. The factories are injectable so tests never call Gemini."""

    def __init__(
        self,
        reports_dir: Path,
        dataset_path: Path = DATASET_PATH,
        *,
        target_factory: Callable[[], Target] | None = None,
        judge_factory: Callable[[str | None, float], Judge] = build_judge,
        client_factory: ClientFactory = _default_client,
    ) -> None:
        self.reports_dir = reports_dir
        self.dataset_path = dataset_path
        self.target_factory = target_factory
        self.judge_factory = judge_factory
        self.client_factory = client_factory
        self._lock = threading.Lock()
        self._jobs: dict[str, EvalJob] = {}
        self._by_request: dict[str, str] = {}
        self._threads: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------ queries
    def get(self, job_id: str) -> EvalJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> EvalJob | None:
        with self._lock:
            return next((j for j in self._jobs.values() if j.status == "running"), None)

    def latest(self) -> EvalJob | None:
        with self._lock:
            return next(reversed(self._jobs.values()), None)

    def wait(self, job_id: str, timeout: float = 10.0) -> None:
        """Block until a job's thread ends (for tests and shutdown)."""
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)

    # ------------------------------------------------------------------ submission
    def submit(
        self, example_ids: list[str], mode: Mode, request_id: str, retriever: Retriever | None = None
    ) -> tuple[EvalJob, bool]:
        """Start a run of the given (already validated) examples. Returns (job, created); created is False when
        `request_id` was already submitted. Raises RunInProgressError if another run is active."""
        with self._lock:
            known = self._by_request.get(request_id)
            if known is not None and known in self._jobs:
                return self._jobs[known], False
            running = next((j for j in self._jobs.values() if j.status == "running"), None)
            if running is not None:
                raise RunInProgressError(running)
            job = EvalJob(id=uuid.uuid4().hex, request_id=request_id, mode=mode, example_ids=list(example_ids))
            self._jobs[job.id] = job
            self._by_request[request_id] = job.id
            self._forget_old()
            thread = threading.Thread(
                target=self._work, args=(job, retriever), name=f"evaluation-{job.id[:8]}", daemon=True
            )
            self._threads[job.id] = thread
        thread.start()
        return job, True

    def _forget_old(self) -> None:
        finished = [j for j in self._jobs.values() if j.status != "running"]
        for job in finished[: max(0, len(self._jobs) - MAX_JOBS_KEPT)]:
            del self._jobs[job.id]
            self._threads.pop(job.id, None)
            self._by_request = {r: j for r, j in self._by_request.items() if j != job.id}

    # ------------------------------------------------------------------ the work (background thread)
    def _counting_target(self, job: EvalJob, retriever: Retriever | None) -> Callable[[], Target]:
        def factory() -> Target:
            target = self.target_factory() if self.target_factory else build_target(retriever=retriever)

            def run(inputs: dict[str, Any]) -> dict[str, Any]:
                try:
                    return target(inputs)
                finally:
                    job.answered += 1

            return run

        return factory

    def _work(self, job: EvalJob, retriever: Retriever | None) -> None:
        try:
            examples = load_examples(self.dataset_path, job.example_ids)
            _, folder = execute_run(
                examples,
                local=job.mode == "local",
                dataset_name=DATASET_NAME,
                client_factory=self.client_factory,
                target_factory=self._counting_target(job, retriever),
                judge_factory=self.judge_factory,
                reports_dir=self.reports_dir,
                process_wide_tracing_off=False,  # never switch off the chat app's tracing
            )
        except DatasetError as exc:
            self._finish(job, error=f"The dataset could not be used: {str(exc)[:400]}")
        except SystemExit as exc:  # raised by the shared runner, e.g. the LangSmith dataset does not exist
            self._finish(job, error=str(exc)[:400] or "The evaluation stopped before it ran.")
        except Exception as exc:
            logger.exception("Evaluation run %s failed", job.id)
            self._finish(job, error=f"The evaluation could not complete ({type(exc).__name__}). See the server log.")
        else:
            self._finish(job, report_id=folder.name)

    def _finish(self, job: EvalJob, report_id: str | None = None, error: str | None = None) -> None:
        with self._lock:
            job.report_id, job.error = report_id, error
            job.status = "completed" if error is None else "failed"
            job.finished_at = _now()
