"""Evals dashboard: page, navigation, dataset, summary, preview, confirmed runs, history, report access.

Mocks only: scripted targets and judges, MagicMock LangSmith clients. No API keys, no network, no Gemini.
Reports are written to the temp folder from the `settings` fixture, never evaluation/reports.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langsmith import utils as ls_utils
from langsmith.evaluation.evaluator import EvaluationResult

from app.chat.engine import PlaceholderEngine
from app.config import Settings
from app.deps import Components, get_components
from app.evaluation.dataset import DATASET_NAME, load_dataset
from app.evaluation.evaluators import MetricResult
from app.evaluation.judge import ChunkVerdict, JudgeError, LLMJudge, ScoreVerdict
from app.evaluation.report import ExampleRun, build_report
from app.evaluation.runner import EvalRunManager
from app.evaluation.target import citations
from app.main import create_app

DATASET = {e.id: e for e in load_dataset()}
LONG_TEXT = "Full-time employees receive 24 days of paid annual leave per calendar year. " * 3


# ---------------------------------------------------------------- fakes
def fake_outputs(answer: str = "You get 24 days [1].") -> dict[str, Any]:
    contexts = [{"number": 1, "source": "leave_policy.md", "chunk_id": "leave:0", "score": 0.81, "text": LONG_TEXT}]
    return {
        "answer": answer,
        "contexts": contexts,
        "sources": ["leave_policy.md"],
        "citations": citations(answer, len(contexts)),
        "retrieval_empty": False,
    }


class ScriptedJudge:
    def __init__(self, score: float = 0.9) -> None:
        self.score, self.calls = score, 0

    def ask(self, prompt: str, schema: type[Any]) -> Any:
        self.calls += 1
        if schema is ChunkVerdict:
            return ChunkVerdict(relevant_chunks=[1], explanation="chunk 1 is relevant")
        return ScoreVerdict(score=self.score, explanation="scripted verdict")


class Recorder:
    """Factories for the run manager that count their use. `never` ones fail the test if used."""

    def __init__(self) -> None:
        self.targets_built = self.judges_built = self.clients_built = 0
        self.questions: list[str] = []
        self.judge = ScriptedJudge()
        self.release = threading.Event()
        self.release.set()
        self.tracing_seen: list[bool] = []

    def target_factory(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        self.targets_built += 1

        def target(inputs: dict[str, Any]) -> dict[str, Any]:
            self.release.wait(10)
            self.tracing_seen.append(bool(ls_utils.tracing_is_enabled()))
            self.questions.append(inputs["question"])
            return fake_outputs()

        return target

    def judge_factory(self, model: str | None, timeout: float) -> Any:
        self.judges_built += 1
        return self.judge

    def never_client(self) -> Any:
        self.clients_built += 1
        raise AssertionError("LangSmith must not be contacted")


@pytest.fixture()
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture()
def reports(settings: Settings) -> Path:
    return settings.eval_reports_dir


@pytest.fixture()
def app(settings: Settings, recorder: Recorder) -> FastAPI:
    application = create_app(settings)
    application.dependency_overrides[get_components] = lambda: Components(engine=PlaceholderEngine())
    application.state.eval_manager = EvalRunManager(
        settings.eval_reports_dir,
        target_factory=recorder.target_factory,
        judge_factory=recorder.judge_factory,
        client_factory=recorder.never_client,
    )
    return application


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def no_langsmith_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)


def run_of(example_id: str, score: float = 0.9, target_error: str | None = None, error_metric: str = "") -> ExampleRun:
    example = DATASET[example_id]
    metrics = {}
    for key in ("faithfulness", "answer_relevance", "retrieval_relevance", "correctness"):
        if target_error or key == error_metric:
            metrics[key] = MetricResult(key=key, error="judge timeout: no answer within 90 s")  # type: ignore[arg-type]
        else:
            metrics[key] = MetricResult(key=key, score=score, explanation=f"{key} explanation")  # type: ignore[arg-type]
    return ExampleRun(
        example_id=example_id,
        category=example.metadata.category,
        question=example.inputs.question,
        is_unanswerable=example.metadata.is_unanswerable,
        expected_sources=example.metadata.expected_sources,
        outputs=None if target_error else fake_outputs(),
        target_error=target_error,
        metrics=metrics,
    )


def write_report(folder: Path, stamp: str, runs: list[ExampleRun], mode: str = "local", url: str | None = None) -> str:
    info = {
        "dataset": DATASET_NAME,
        "mode": mode,
        "created_at": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T10:00:00+00:00",
    }
    if url:
        info["experiment_url"] = url
    run_id = f"{stamp}-{mode}-{DATASET_NAME}"
    (folder / run_id).mkdir(parents=True)
    (folder / run_id / "report.json").write_text(json.dumps(build_report(runs, info)), encoding="utf-8")
    return run_id


def two_runs(reports: Path) -> tuple[str, str]:
    old = write_report(reports, "20261101T100000Z", [run_of("q01", 0.5), run_of("q02", 0.9), run_of("q03", 0.4)])
    new = write_report(
        reports,
        "20261102T100000Z",
        [run_of("q01", 0.6), run_of("q02", 0.95), run_of("q03", target_error="TimeoutError: retriever timed out")],
        mode="langsmith",
        url="https://smith.langchain.com/o/x/datasets/y/compare?selectedSessions=z",
    )
    return old, new


def run_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "scope": "selected",
        "example_ids": ["q01", "q05", "q17"],
        "mode": "local",
        "request_id": "req-0000-0001",
        "confirm_gemini_usage": True,
        "confirm_langsmith_upload": False,
        "acknowledged_max_calls": 15,
    }
    return {**body, **overrides}


def wait_for(app: FastAPI, job_id: str) -> dict[str, Any]:
    app.state.eval_manager.wait(job_id, timeout=20)
    job = app.state.eval_manager.get(job_id)
    assert job is not None and job.status != "running"
    return dict(job.public())


def assert_nothing_ran(recorder: Recorder, reports: Path) -> None:
    assert (recorder.targets_built, recorder.judges_built, recorder.clients_built) == (0, 0, 0)
    assert not reports.exists() or not any(reports.iterdir())


# ---------------------------------------------------------------- page and navigation
def test_evals_page_renders_empty_state_without_running_anything(
    client: TestClient, app: FastAPI, recorder: Recorder, reports: Path
) -> None:
    page = client.get("/evals")
    assert page.status_code == 200
    html = page.text
    assert "<h1>Evals</h1>" in html and "Evaluate the quality and reliability of your RAG chatbot." in html
    assert DATASET_NAME in html and "19 examples" in html
    for label in ("Faithfulness", "Answer relevance", "Retrieval relevance", "Correctness"):
        assert label in html
    assert "No evaluation results yet" in html and "Not evaluated yet" in html
    assert "%</span>" not in html, "no score is shown before the first run"
    assert 'id="open-run-dialog"' in html and "Run Evaluation" in html
    assert "/static/evals.js" in html
    assert_nothing_ran(recorder, reports)
    assert app.state.eval_manager.active() is None and app.state.eval_manager.latest() is None


def test_sidebar_links_chat_and_evals(client: TestClient) -> None:
    client.post("/ui/sessions")
    chat = client.get("/").text
    assert 'href="/evals"' in chat and "Evals" in chat
    assert "+ New chat" in chat and "Documents" in chat, "existing sidebar items are kept"
    evals = client.get("/evals").text
    assert 'href="/evals" aria-current="page"' in evals
    assert 'class="session-link' in evals, "the chats stay reachable from the Evals page"


def test_run_dialog_shows_warnings_and_both_confirmations(client: TestClient) -> None:
    html = client.get("/evals").text
    assert "Gemini answer-generation calls may incur usage." in html
    assert "LLM-as-a-judge calls may incur additional usage." in html
    assert "LangSmith mode sends evaluation questions, generated answers, retrieved excerpts" in html
    assert "Local mode does not upload evaluation results to LangSmith." in html
    assert 'name="confirm_gemini_usage"' in html and 'name="confirm_langsmith_upload"' in html
    assert 'value="langsmith" disabled' in html, "LangSmith mode is unavailable without a configured key"


def test_langsmith_mode_is_offered_when_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test-secret-value")
    html = client.get("/evals").text
    assert 'value="langsmith" disabled' not in html
    assert "lsv2-test-secret-value" not in html


# ---------------------------------------------------------------- dataset
def test_dataset_listing(client: TestClient) -> None:
    data = client.get("/api/evals/dataset").json()
    assert data["dataset"] == DATASET_NAME and data["total"] == data["count"] == 19
    first = data["examples"][0]
    assert set(first) >= {"example_id", "question", "category", "expected_sources", "reference_answer", "answerable"}
    assert first["reference_answer"] == DATASET["q01"].outputs.reference_answer
    assert sum(not e["answerable"] for e in data["examples"]) == sum(
        e.metadata.is_unanswerable for e in DATASET.values()
    )
    assert sum(c["count"] for c in data["categories"]) == 19


def test_dataset_filtering(client: TestClient) -> None:
    unanswerable = client.get("/api/evals/dataset", params={"category": "unanswerable"}).json()
    assert unanswerable["count"] > 0 and all(e["category"] == "unanswerable" for e in unanswerable["examples"])
    found = client.get("/api/evals/dataset", params={"q": "HOTEL RATE"}).json()
    assert [e["example_id"] for e in found["examples"]] == ["q02"]
    assert client.get("/api/evals/dataset", params={"q": "no such words anywhere"}).json()["count"] == 0


def test_dataset_page_supports_selection(client: TestClient) -> None:
    html = client.get("/evals").text
    assert html.count('class="dataset-check"') == 19
    assert 'id="dataset-select-all"' in html and 'id="dataset-selected"' in html and "0 selected" in html
    assert 'id="dataset-search"' in html and 'id="dataset-category"' in html
    assert 'data-category="unanswerable"' in html


def test_dataset_example_and_invalid_ids(client: TestClient) -> None:
    one = client.get("/api/evals/dataset/q02").json()
    assert one["example_id"] == "q02" and one["evidence"] and one["expected_facts"]
    for bad in ("q99", "x01", "..", "q01.json", "%2e%2e%2fapp"):
        assert client.get(f"/api/evals/dataset/{bad}").status_code == 404


def test_dataset_cannot_be_edited(client: TestClient) -> None:
    for method in ("put", "patch", "delete", "post"):
        assert getattr(client, method)("/api/evals/dataset/q01").status_code == 405


# ---------------------------------------------------------------- summary
def test_summary_without_runs(client: TestClient) -> None:
    data = client.get("/api/evals/summary").json()
    assert data["latest"] is None and data["previous"] is None and data["run_count"] == 0
    assert data["dataset"] == {"name": DATASET_NAME, "size": 19}
    assert all(card["score"] is None and card["delta"] is None for card in data["metrics"])
    assert data["active_job"] is None


def test_summary_with_previous_run_and_errors_kept_apart(client: TestClient, reports: Path) -> None:
    old, new = two_runs(reports)
    data = client.get("/api/evals/summary").json()
    latest = data["latest"]
    assert latest["id"] == new and data["previous"]["id"] == old
    assert latest["examples_total"] == 3
    assert (latest["examples_passed"], latest["examples_failed"], latest["examples_errored"]) == (1, 1, 1)
    assert latest["metric_errors"] == 4 and latest["target_errors"] == 1
    faith = next(c for c in data["metrics"] if c["key"] == "faithfulness")
    assert faith["score"] == 77.5, "mean of 0.6 and 0.95; the errored example is not a zero"
    assert faith["previous"] == 60.0 and faith["delta"] == 17.5
    assert latest["overall"] == 77.5 and latest["experiment_url"].startswith("https://smith.langchain.com/")
    html = client.get("/ui/evals/summary").text
    assert "77.5" in html and "+17.5 pts" in html and "Evaluation errors" in html and "Open in LangSmith" in html


def test_unsafe_experiment_links_are_not_shown(client: TestClient, reports: Path) -> None:
    write_report(reports, "20261103T100000Z", [run_of("q01")], mode="langsmith", url="javascript:alert(1)")
    assert client.get("/api/evals/summary").json()["latest"]["experiment_url"] is None
    assert "javascript:" not in client.get("/evals").text


# ---------------------------------------------------------------- preview
def test_preview_estimates_calls_without_external_calls(client: TestClient, recorder: Recorder, reports: Path) -> None:
    three = client.post("/api/evals/preview", json={"scope": "selected", "example_ids": ["q01", "q05", "q17"]}).json()
    assert three["example_count"] == 3 and three["max_gemini_calls"] == 15
    assert (three["answer_calls"], three["judge_calls"]) == (3, 12)
    assert three["sends_to_langsmith"] is False and "does not upload" in three["warnings"][-1]
    everything = client.post("/api/evals/preview", json={"scope": "all", "mode": "langsmith"}).json()
    assert everything["example_count"] == 19 and everything["max_gemini_calls"] == 95
    assert everything["sends_to_langsmith"] is True and "sends the evaluation questions" in everything["warnings"][-1]
    assert_nothing_ran(recorder, reports)


@pytest.mark.parametrize(
    "body",
    [
        {"scope": "selected", "example_ids": []},
        {"scope": "selected", "example_ids": ["q01", "q01"]},
        {"scope": "selected", "example_ids": ["q99"]},
        {"scope": "selected", "example_ids": ["../../etc/passwd"]},
        {"scope": "all", "mode": "remote"},
    ],
)
def test_preview_rejects_invalid_selections(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/api/evals/preview", json=body).status_code == 422


# ---------------------------------------------------------------- running (confirmations)
def test_local_run_requires_confirmation(client: TestClient, recorder: Recorder, reports: Path) -> None:
    response = client.post("/api/evals/run", json=run_body(confirm_gemini_usage=False))
    assert response.status_code == 400 and "Gemini usage" in response.json()["detail"]
    missing = run_body()
    del missing["confirm_gemini_usage"]
    assert client.post("/api/evals/run", json=missing).status_code == 400
    assert_nothing_ran(recorder, reports)


def test_run_requires_the_acknowledged_estimate(client: TestClient, recorder: Recorder, reports: Path) -> None:
    response = client.post("/api/evals/run", json=run_body(acknowledged_max_calls=5))
    assert response.status_code == 422 and "estimate is 15" in response.json()["detail"]
    assert_nothing_ran(recorder, reports)


def test_run_rejects_invalid_example_ids(client: TestClient, recorder: Recorder, reports: Path) -> None:
    assert client.post("/api/evals/run", json=run_body(example_ids=["q01", "q99"])).status_code == 422
    assert client.post("/api/evals/run", json=run_body(example_ids=["q01/../q02"])).status_code == 422
    assert client.post("/api/evals/run", json=run_body(mode="cloud")).status_code == 422
    assert client.post("/api/evals/run", json=run_body(request_id="x")).status_code == 422
    assert_nothing_ran(recorder, reports)


def test_confirmed_local_run_writes_a_report_and_sends_nothing_to_langsmith(
    client: TestClient, app: FastAPI, recorder: Recorder, reports: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")  # the chat app traces; the evaluation must not
    response = client.post("/api/evals/run", json=run_body())
    assert response.status_code == 202
    started = response.json()
    assert started["status"] == "running" and started["example_count"] == 3 and started["max_gemini_calls"] == 15
    job = wait_for(app, started["job_id"])
    assert job["status"] == "completed" and job["answered"] == 3 and job["error"] is None
    assert recorder.clients_built == 0, "local mode never builds a LangSmith client"
    assert recorder.tracing_seen == [False, False, False], "the run is untraced"
    assert os.environ["LANGSMITH_TRACING"] == "true", "the app's tracing setting is untouched"
    report = json.loads((reports / job["report_id"] / "report.json").read_text(encoding="utf-8"))
    assert report["info"]["mode"] == "local" and [e["example_id"] for e in report["examples"]] == ["q01", "q05", "q17"]
    assert client.get(f"/api/evals/jobs/{started['job_id']}").json()["report_id"] == job["report_id"]
    assert client.get("/api/evals/history").json()["count"] == 1


def test_local_run_judge_calls_are_untraced(
    client: TestClient, app: FastAPI, recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real LLMJudge runs its call in a worker thread; the run's 'tracing off' context must reach it."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test-secret-value")
    seen: list[bool] = []

    def answer(schema: type[Any], messages: list[Any]) -> Any:
        seen.append(bool(ls_utils.tracing_is_enabled()))
        if schema is ChunkVerdict:
            return ChunkVerdict(relevant_chunks=[1], explanation="relevant")
        return ScoreVerdict(score=0.8, explanation="fine")

    fake_llm = SimpleNamespace(
        with_structured_output=lambda schema: SimpleNamespace(invoke=lambda m: answer(schema, m))
    )
    app.state.eval_manager.judge_factory = lambda model, timeout: LLMJudge(fake_llm, timeout)  # type: ignore[arg-type]
    job = wait_for(
        app,
        client.post("/api/evals/run", json=run_body(example_ids=["q01"], acknowledged_max_calls=5)).json()["job_id"],
    )
    assert job["status"] == "completed" and seen and not any(seen)


def test_evaluation_runs_in_isolation_from_chat_data(
    client: TestClient, app: FastAPI, settings: Settings, recorder: Recorder
) -> None:
    sid = client.post("/api/sessions").json()["id"]
    before = client.get(f"/api/sessions/{sid}").json()
    job = wait_for(app, client.post("/api/evals/run", json=run_body()).json()["job_id"])
    assert job["status"] == "completed"
    assert client.get(f"/api/sessions/{sid}").json() == before
    assert [s["id"] for s in client.get("/api/sessions").json()] == [sid], "no chat session was created"
    assert not settings.review_root.exists() or not any(settings.review_root.rglob("*.json"))
    assert job["report_id"] and (settings.eval_reports_dir / job["report_id"]).is_dir()


def test_langsmith_run_needs_a_configured_key(client: TestClient, recorder: Recorder, reports: Path) -> None:
    body = run_body(mode="langsmith", confirm_langsmith_upload=True)
    response = client.post("/api/evals/run", json=body)
    assert response.status_code == 400 and "not configured" in response.json()["detail"]
    assert_nothing_ran(recorder, reports)


def test_langsmith_run_requires_its_own_confirmation(
    client: TestClient, recorder: Recorder, reports: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test-secret-value")
    response = client.post("/api/evals/run", json=run_body(mode="langsmith"))
    assert response.status_code == 400 and "LangSmith" in response.json()["detail"]
    assert "lsv2-test-secret-value" not in response.text
    assert_nothing_ran(recorder, reports)


def test_confirmed_langsmith_run_creates_an_experiment(
    client: TestClient, app: FastAPI, reports: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test-secret-value")
    remote = [
        SimpleNamespace(
            inputs=DATASET[i].inputs.model_dump(),
            outputs=DATASET[i].outputs.model_dump(),
            metadata=DATASET[i].metadata.model_dump(mode="json"),
        )
        for i in ("q01", "q05", "q17")
    ]

    class Experiment:
        """Minimal stand-in for langsmith's ExperimentResults: iterable rows and a url."""

        def __init__(self, url: str, rows: list[dict[str, Any]]) -> None:
            self.url, self.rows = url, rows

        def __iter__(self) -> Any:
            return iter(self.rows)

    def fake_evaluate(target: Any, **kwargs: Any) -> Experiment:
        rows = []
        for ex in kwargs["data"]:
            out = target(ex.inputs)
            run = SimpleNamespace(outputs=out, error=None)
            results = [EvaluationResult(**e(run, ex)) for e in kwargs["evaluators"]]
            rows.append({"run": run, "example": ex, "evaluation_results": {"results": results}})
        return Experiment("https://smith.langchain.com/o/x/experiment", rows)

    client_mock = MagicMock()
    client_mock.has_dataset.return_value = True
    client_mock.list_examples.return_value = remote
    client_mock.evaluate.side_effect = fake_evaluate
    app.state.eval_manager.client_factory = lambda: client_mock
    body = run_body(
        mode="langsmith", confirm_langsmith_upload=True, example_ids=["q01", "q17"], acknowledged_max_calls=10
    )
    job = wait_for(app, client.post("/api/evals/run", json=body).json()["job_id"])
    assert job["status"] == "completed"
    assert [ex.metadata["example_id"] for ex in client_mock.evaluate.call_args.kwargs["data"]] == ["q01", "q17"]
    summary = client.get("/api/evals/summary").json()["latest"]
    assert summary["mode"] == "langsmith" and summary["experiment_url"] == "https://smith.langchain.com/o/x/experiment"


def test_duplicate_submissions_are_prevented(client: TestClient, app: FastAPI, recorder: Recorder) -> None:
    recorder.release.clear()  # hold the run so it is still active
    try:
        first = client.post("/api/evals/run", json=run_body())
        assert first.status_code == 202
        again = client.post("/api/evals/run", json=run_body())
        assert again.status_code == 200 and again.json()["duplicate"] is True
        assert again.json()["job_id"] == first.json()["job_id"]
        other = client.post(
            "/api/evals/run",
            json=run_body(request_id="req-0000-0002", scope="all", example_ids=[], acknowledged_max_calls=95),
        )
        assert other.status_code == 409 and other.json()["job"]["job_id"] == first.json()["job_id"]
        assert client.get("/api/evals/summary").json()["active_job"]["job_id"] == first.json()["job_id"]
        status_html = client.get("/evals").text
        assert "Evaluation running" in status_html and 'hx-trigger="every 2s"' in status_html
    finally:
        recorder.release.set()
    assert wait_for(app, first.json()["job_id"])["status"] == "completed"
    assert recorder.targets_built == 1, "only one run was started"


def test_chat_keeps_working_while_an_evaluation_runs(client: TestClient, app: FastAPI, recorder: Recorder) -> None:
    recorder.release.clear()
    try:
        job_id = client.post("/api/evals/run", json=run_body()).json()["job_id"]
        sid = client.post("/api/sessions").json()["id"]
        reply = client.post(f"/api/sessions/{sid}/messages", json={"content": "hello"})
        assert reply.status_code == 200
    finally:
        recorder.release.set()
    wait_for(app, job_id)


def test_failed_run_is_an_error_not_a_score(client: TestClient, app: FastAPI, reports: Path) -> None:
    def broken() -> Any:
        raise RuntimeError("secret internal detail /home/user/.env")

    app.state.eval_manager.target_factory = broken
    job = wait_for(app, client.post("/api/evals/run", json=run_body()).json()["job_id"])
    assert job["status"] == "failed" and job["report_id"] is None
    assert "RuntimeError" in job["error"] and "secret internal detail" not in job["error"]
    assert "Traceback" not in job["error"]
    html = client.get(f"/ui/evals/status?job={job['job_id']}")
    assert "could not complete" in html.text and "not a quality score" in html.text
    assert html.headers["HX-Trigger"] == "evals-run-finished"
    assert not any(reports.iterdir()) if reports.exists() else True
    assert client.get("/api/evals/summary").json()["latest"] is None


def test_judge_errors_are_reported_as_evaluation_errors(client: TestClient, app: FastAPI) -> None:
    class FailingJudge:
        def ask(self, prompt: str, schema: type[Any]) -> Any:
            raise JudgeError("timeout", "no answer within 1 s")

    app.state.eval_manager.judge_factory = lambda model, timeout: FailingJudge()
    job = wait_for(
        app,
        client.post("/api/evals/run", json=run_body(example_ids=["q01"], acknowledged_max_calls=5)).json()["job_id"],
    )
    assert job["status"] == "completed"
    detail = client.get(f"/api/evals/runs/{job['report_id']}").json()
    assert detail["examples"][0]["status"] == "error"
    assert detail["run"]["examples_errored"] == 1 and detail["run"]["examples_failed"] == 0


def test_status_polling_stops_and_fires_refresh_when_done(client: TestClient, app: FastAPI) -> None:
    job_id = client.post("/api/evals/run", json=run_body()).json()["job_id"]
    wait_for(app, job_id)
    done = client.get(f"/ui/evals/status?job={job_id}")
    assert done.headers.get("HX-Trigger") == "evals-run-finished" and "every 2s" not in done.text
    assert "Evaluation finished" in done.text and "View results" in done.text
    assert "HX-Trigger" not in client.get("/ui/evals/status").headers
    assert client.get("/api/evals/jobs/" + "0" * 32).status_code == 404
    assert client.get("/api/evals/jobs/../../etc").status_code == 404


# ---------------------------------------------------------------- results, history, comparison
def test_run_results_and_detail(client: TestClient, reports: Path) -> None:
    _, new = two_runs(reports)
    data = client.get(f"/api/evals/runs/{new}").json()
    rows = {r["example_id"]: r for r in data["examples"]}
    assert rows["q01"]["status"] == "failed" and rows["q02"]["status"] == "passed" and rows["q03"]["status"] == "error"
    assert rows["q02"]["reference_answer"] == DATASET["q02"].outputs.reference_answer
    assert rows["q02"]["answer"] == "You get 24 days [1]." and rows["q02"]["scores"]["correctness"]["score"] == 95.0
    detail = client.get(f"/api/evals/runs/{new}/examples/q02").json()
    assert detail["sources"] == ["leave_policy.md"] and detail["contexts"][0]["text"] == LONG_TEXT
    assert detail["citations"]["cited"] == [1] and detail["citations"]["invalid"] == []
    assert detail["scores"]["faithfulness"]["explanation"] == "faithfulness explanation"
    errored = client.get(f"/api/evals/runs/{new}/examples/q03").json()
    assert errored["target_error"].startswith("TimeoutError") and errored["scores"]["faithfulness"]["error"]
    assert client.get(f"/api/evals/runs/{new}/examples/q09").status_code == 404

    table = client.get(f"/ui/evals/results?run={new}").text
    assert table.count('class="clickable"') == 3 and f"/ui/evals/runs/{new}/examples/q02" in table
    assert "Evaluation error" in table and "Passed" in table and "Failed" in table
    html = client.get(f"/ui/evals/runs/{new}/examples/q03").text
    assert "the answer could not be generated" in html and "Evaluation error:" in html
    html = client.get(f"/ui/evals/runs/{new}/examples/q02").text
    assert "Valid: cites excerpt" in html and "leave_policy.md" in html and "Reference answer" in html


def test_latest_results_are_shown_on_the_page(client: TestClient, reports: Path) -> None:
    _, new = two_runs(reports)
    html = client.get("/evals").text
    assert f"/ui/evals/runs/{new}/examples/q01" in html and "No results to show" not in html


def test_history_lists_runs_newest_first(client: TestClient, reports: Path) -> None:
    old, new = two_runs(reports)
    (reports / "notes").mkdir()
    (reports / "README.md").write_text("not a report", encoding="utf-8")
    history = client.get("/api/evals/history").json()
    assert [r["id"] for r in history["runs"]] == [new, old]
    html = client.get("/ui/evals/history").text
    assert f"/ui/evals/results?run={old}" in html and "Compare two runs" in html


def test_compare_two_runs(client: TestClient, reports: Path) -> None:
    old, new = two_runs(reports)
    data = client.get("/api/evals/compare", params={"base": old, "target": new}).json()
    metrics = {m["key"]: m for m in data["metrics"]}
    assert metrics["faithfulness"]["direction"] == "improved" and metrics["faithfulness"]["delta"] == 17.5
    assert [i["example_id"] for i in data["failed_in_both"]] == ["q01", "q03"]
    assert data["fixed"] == [] and data["regressed"] == [] and data["same_examples"] is True
    back = client.get("/api/evals/compare", params={"base": new, "target": old}).json()
    assert {m["key"]: m["direction"] for m in back["metrics"]}["faithfulness"] == "declined"
    html = client.get("/ui/evals/compare", params={"base": old, "target": new}).text
    assert "Failed in both runs" in html and "improved" in html
    assert client.get("/api/evals/compare", params={"base": old, "target": old}).status_code == 422
    assert "Choose two different runs" in client.get("/ui/evals/compare", params={"base": old, "target": old}).text


# ---------------------------------------------------------------- report access and path traversal
@pytest.mark.parametrize(
    "run_id",
    [
        "..",
        "..%2F..%2Fapp",
        "%2e%2e",
        "20261101T100000Z-local-..",
        "20261101T100000Z-local-x%2F..%2F..%2Fapp",
        "20261101T100000Z-other-RAG-Chatbot-Evaluation-v1",
        "20991231T000000Z-local-RAG-Chatbot-Evaluation-v1",
        "report.json",
    ],
)
def test_report_access_is_restricted_to_valid_report_ids(client: TestClient, reports: Path, run_id: str) -> None:
    two_runs(reports)
    assert client.get(f"/api/evals/runs/{run_id}").status_code == 404
    assert client.get(f"/api/evals/runs/{run_id}/examples/q01").status_code == 404
    assert client.get(f"/ui/evals/results?run={run_id}").status_code == 404
    assert client.get("/api/evals/compare", params={"base": run_id, "target": "x"}).status_code == 404


def test_report_outside_the_reports_folder_is_not_reachable(client: TestClient, reports: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    run_id = write_report(outside, "20261105T100000Z", [run_of("q01")])
    assert client.get(f"/api/evals/runs/{run_id}").status_code == 404
    assert client.get(f"/api/evals/runs/..%2Felsewhere%2F{run_id}").status_code == 404


def test_corrupt_reports_are_skipped(client: TestClient, reports: Path) -> None:
    old, _ = two_runs(reports)
    broken = reports / "20261104T100000Z-local-RAG-Chatbot-Evaluation-v1"
    broken.mkdir()
    (broken / "report.json").write_text("{not json", encoding="utf-8")
    assert client.get(f"/api/evals/runs/{broken.name}").status_code == 404
    assert broken.name not in [r["id"] for r in client.get("/api/evals/history").json()["runs"]]
    assert client.get("/evals").status_code == 200


def test_responses_never_include_secrets_or_paths(
    client: TestClient, reports: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test-secret-value")
    _, new = two_runs(reports)
    for url in ("/evals", "/api/evals/summary", "/api/evals/history", f"/api/evals/runs/{new}", "/api/evals/dataset"):
        text = client.get(url).text
        assert "lsv2-test-secret-value" not in text and str(settings.eval_reports_dir) not in text
    preview = client.post("/api/evals/preview", json={"scope": "all", "mode": "langsmith"}).text
    assert "lsv2-test-secret-value" not in preview
