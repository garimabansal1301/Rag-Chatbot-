"""Offline evaluation: dataset, evaluators, judge reliability, isolated target, reports, LangSmith scripts.

Mocks only: scripted judges, fake LLMs and retrievers, MagicMock LangSmith clients. No API keys, no network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.tracers.context import collect_runs
from langsmith.evaluation.evaluator import EvaluationResult, run_evaluator
from pydantic import BaseModel, ValidationError

from app.chat.review import REVIEW_TOOL_NAME
from app.config import PROJECT_ROOT, get_settings
from app.evaluation import cli
from app.evaluation.dataset import (
    DATASET_NAME,
    DatasetError,
    EvalExample,
    check_evidence,
    load_dataset,
    parse_lines,
    select,
)
from app.evaluation.evaluators import (
    METRICS,
    MetricResult,
    abstains,
    answer_relevance,
    correctness,
    evaluate_example,
    faithfulness,
    langsmith_evaluators,
    retrieval_relevance,
)
from app.evaluation.judge import ChunkVerdict, JudgeError, LLMJudge, ScoreVerdict
from app.evaluation.report import ExampleRun, build_report, run_local, write_report
from app.evaluation.target import EVAL_SESSION_PREFIX, EvaluationTarget, build_target, citations
from app.ingest.pipeline import document_from_bytes
from tests.fakes import FakeChatModel, StaticRetriever, make_chunk

DOCUMENTS = PROJECT_ROOT / "data" / "documents"
LONG_TEXT = "Full-time employees receive 24 days of paid annual leave per calendar year. " * 6  # > 200 chars
CHUNKS = [
    make_chunk(LONG_TEXT, source="leave_policy.md", position=0, doc_id="leave"),
    make_chunk("Hotel rates are capped at 180 USD per night.", source="expense_policy.md", position=0, doc_id="exp"),
]


class ScriptedJudge:
    """Returns scripted verdicts (or raises) and records every prompt."""

    def __init__(self, score: float = 0.9, chunks: list[int] | None = None, error: JudgeError | None = None) -> None:
        self.score, self.chunks, self.error = score, [1] if chunks is None else chunks, error
        self.prompts: list[str] = []

    def ask(self, prompt: str, schema: type[Any]) -> Any:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        if schema is ChunkVerdict:
            return ChunkVerdict(relevant_chunks=self.chunks, explanation="chunk 1 states the leave days")
        return ScoreVerdict(score=self.score, explanation="scripted verdict")


def example(**overrides: Any) -> EvalExample:
    data: dict[str, Any] = {
        "inputs": {"question": "How many days of annual leave do full-time employees get?"},
        "outputs": {"reference_answer": "Full-time employees receive 24 days of paid annual leave per calendar year."},
        "metadata": {
            "example_id": "q01",
            "category": "direct_factual",
            "expected_sources": ["leave_policy.md"],
            "expected_facts": ["24 days"],
            "is_unanswerable": False,
            "evidence": [
                {"source": "leave_policy.md", "quote": "Full-time employees receive 24 days of paid annual leave"}
            ],
        },
    }
    for key, value in overrides.items():
        section, field = key.split("__")
        data[section][field] = value
    return EvalExample.model_validate(data)


UNANSWERABLE = example(
    metadata__example_id="q17",
    inputs__question="What is the policy on pets at the office?",
    outputs__reference_answer="The documents do not cover pets at the office.",
    metadata__category="unanswerable",
    metadata__is_unanswerable=True,
    metadata__expected_sources=[],
    metadata__expected_facts=[],
    metadata__evidence=[],
)


def outputs(answer: str = "You get 24 days [1].", contexts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    contexts = (
        [{"source": "leave_policy.md", "chunk_id": "leave:0", "text": LONG_TEXT}] if contexts is None else contexts
    )
    return {
        "answer": answer,
        "contexts": contexts,
        "sources": sorted({c["source"] for c in contexts}),
        "citations": citations(answer, len(contexts)),
        "retrieval_empty": not contexts,
    }


def jsonl(*records: dict[str, Any]) -> list[str]:
    return [json.dumps(r) for r in records]


# ---------------------------------------------------------------- dataset
def test_repository_dataset_is_valid_and_representative() -> None:
    examples = load_dataset()
    assert 15 <= len(examples) <= 20
    categories = {e.metadata.category for e in examples}
    assert categories == {
        "direct_factual",
        "multi_chunk",
        "multi_document",
        "scenario",
        "careful_grounding",
        "unanswerable",
    }
    assert any(len(e.metadata.expected_sources) > 1 for e in examples), "multi-document questions"
    sources = {s for e in examples for s in e.metadata.expected_sources}
    assert sources == {p.name for p in DOCUMENTS.iterdir() if p.is_file() and not p.name.startswith(".")}


@pytest.mark.skipif(not DOCUMENTS.is_dir(), reason="knowledge-base documents not present")
def test_reference_answers_are_backed_by_verbatim_document_quotes() -> None:
    docs = {
        p.name: document_from_bytes(p.name, p.read_bytes()).text
        for p in DOCUMENTS.iterdir()
        if p.is_file() and not p.name.startswith(".")
    }
    assert check_evidence(load_dataset(), docs) == []


def test_evidence_check_catches_invented_facts_and_quotes() -> None:
    docs = {"leave_policy.md": "Full-time employees receive 24 days of paid annual leave per calendar year."}
    assert check_evidence([example()], docs) == []
    invented = example(metadata__expected_facts=["30 days"])
    assert "expected fact '30 days' is not in any evidence quote" in check_evidence([invented], docs)[0]
    misquoted = example(
        metadata__evidence=[{"source": "leave_policy.md", "quote": "Employees receive 30 days of leave"}]
    )
    assert "quote not found in leave_policy.md" in check_evidence([misquoted], docs)[0]


def test_schema_errors_are_reported_with_line_numbers() -> None:
    good = example().model_dump()
    no_reference = json.loads(json.dumps(good))
    del no_reference["outputs"]["reference_answer"]
    bad_unanswerable = json.loads(json.dumps(good))
    bad_unanswerable["metadata"].update(example_id="q02", is_unanswerable=True)
    with pytest.raises(DatasetError) as error:
        parse_lines([*jsonl(no_reference, bad_unanswerable), "{not json"])
    message = str(error.value)
    assert "line 1: outputs.reference_answer: Field required" in message
    assert "line 2:" in message and "line 3: not valid JSON" in message


def test_duplicate_ids_and_questions_are_rejected() -> None:
    first = example().model_dump()
    same_id = example(inputs__question="Another question about leave?").model_dump()
    same_question = example(
        inputs__question="  how many DAYS of annual leave do full-time employees get? "
    ).model_dump()
    same_question["metadata"]["example_id"] = "q03"
    with pytest.raises(DatasetError) as error:
        parse_lines(jsonl(first, same_id, same_question))
    assert "duplicate example_id q01" in str(error.value) and "q03 repeats the question of q01" in str(error.value)


def test_subset_selection() -> None:
    examples = load_dataset()
    assert [e.id for e in select(examples, ["q17", "q01"])] == ["q01", "q17"]
    assert len(select(examples, limit=3)) == 3
    with pytest.raises(DatasetError, match="Unknown example id"):
        select(examples, ["q99"])


# ---------------------------------------------------------------- evaluators
def test_faithfulness_judges_the_answer_against_the_retrieved_contexts() -> None:
    judge = ScriptedJudge(score=0.5)
    result = faithfulness("Leave?", outputs("You get 24 days [1] and a car."), judge)
    assert result.score == 0.5 and result.judged and result.passed is False
    assert LONG_TEXT[:150] in judge.prompts[0] and "a car" in judge.prompts[0], "contexts and answer reach the judge"


def test_faithfulness_with_empty_retrieval_needs_no_judge() -> None:
    judge = ScriptedJudge()
    abstained = faithfulness("Pets?", outputs("The documents do not cover pets.", contexts=[]), judge)
    claims = faithfulness("Pets?", outputs("Pets are allowed on Fridays.", contexts=[]), judge)
    assert (abstained.score, claims.score) == (1.0, 0.0) and judge.prompts == []


def test_empty_answer_scores_zero_without_judging() -> None:
    judge = ScriptedJudge()
    results = evaluate_example(example(), outputs(""), judge)
    assert results["faithfulness"].score == results["answer_relevance"].score == results["correctness"].score == 0.0
    assert all("empty answer" in results[k].explanation for k in ("faithfulness", "answer_relevance", "correctness"))
    assert len(judge.prompts) == 1, "only retrieval relevance (which does not depend on the answer) was judged"


def test_answer_relevance_uses_the_judge_score() -> None:
    judge = ScriptedJudge(score=1.0)
    assert answer_relevance("Leave?", outputs(), judge).score == 1.0
    assert "<question>Leave?</question>" in judge.prompts[0]


def test_retrieval_relevance_counts_relevant_chunks_with_sources_as_evidence() -> None:
    two = [{"source": "leave_policy.md", "text": LONG_TEXT}, {"source": "expense_policy.md", "text": "Hotel caps."}]
    judge = ScriptedJudge(chunks=[1])
    result = retrieval_relevance("Leave?", outputs(contexts=two), ["leave_policy.md"], False, judge)
    assert result.score == 0.5 and "1 of 2 retrieved chunks are relevant" in result.explanation
    assert "The verified answer is found in: leave_policy.md" in judge.prompts[0]
    only_other = retrieval_relevance(
        "Leave?", outputs(contexts=two[1:]), ["leave_policy.md"], False, ScriptedJudge(chunks=[])
    )
    assert only_other.score == 0.0 and "Expected sources not retrieved: leave_policy.md" in only_other.explanation


def test_retrieval_relevance_edge_cases() -> None:
    judge = ScriptedJudge(chunks=[1, 5])
    invalid = retrieval_relevance("Leave?", outputs(), ["leave_policy.md"], False, judge)
    assert invalid.score is None and "outside 1-1" in (invalid.error or ""), "invalid chunk numbers are an error"
    empty = outputs(contexts=[])
    assert retrieval_relevance("Leave?", empty, ["leave_policy.md"], False, judge).score == 0.0
    assert retrieval_relevance("Pets?", empty, [], True, judge).score == 1.0


def test_correctness_uses_the_reference_answer_and_facts() -> None:
    judge = ScriptedJudge(score=0.8)
    ex = example()
    result = correctness(ex.inputs.question, outputs(), ex.outputs.reference_answer, ["24 days"], False, judge)
    assert result.score == 0.8
    assert (
        "<reference>Full-time employees receive 24 days" in judge.prompts[0]
        and "<facts>24 days</facts>" in judge.prompts[0]
    )
    missing = correctness("Leave?", outputs(), "  ", ["24 days"], False, judge)
    assert missing.score is None and missing.error == "missing reference answer"


def test_unanswerable_questions_check_abstention() -> None:
    judge = ScriptedJudge(score=0.0)
    ref = UNANSWERABLE.outputs.reference_answer
    abstained = correctness("Pets?", outputs("The documents do not mention pets.", []), ref, [], True, judge)
    assert abstained.score == 1.0 and not abstained.judged
    invented = correctness("Pets?", outputs("Pets are welcome on Fridays.", []), ref, [], True, judge)
    assert invented.score == 0.0 and "do NOT contain the answer" in judge.prompts[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The documents do not cover pets.", True),
        ("The provided documents don't mention an Enterprise price.", True),
        ("I could not find this in the documents.", True),
        ("There is no information about this.", True),
        ("You get 24 days of annual leave [1].", False),
        ("Alcohol is never reimbursed [1].", False),
    ],
)
def test_abstention_detection(text: str, expected: bool) -> None:
    assert abstains(text) is expected


def test_judge_failures_are_errors_not_scores() -> None:
    for kind in ("timeout", "llm_failure", "invalid_output"):
        judge = ScriptedJudge(error=JudgeError(kind, "boom"))  # type: ignore[arg-type]
        results = evaluate_example(example(), outputs(), judge)
        assert all(
            r.score is None and r.passed is None and (r.error or "").startswith(f"judge {kind}")
            for r in results.values()
        )


def test_scores_must_be_between_zero_and_one() -> None:
    for bad in (1.5, -0.1, float("nan")):
        with pytest.raises(ValidationError):
            ScoreVerdict(score=bad, explanation="x")
        with pytest.raises(ValidationError):
            MetricResult(key="faithfulness", score=bad, explanation="x")
    with pytest.raises(ValidationError):
        MetricResult(key="faithfulness")  # neither score nor error
    with pytest.raises(ValidationError):
        MetricResult(key="faithfulness", score=0.5, error="both")


def test_malformed_target_output_is_an_error() -> None:
    results = evaluate_example(example(), {"answer": "24 days"}, ScriptedJudge())
    assert results["faithfulness"].error is not None and "contexts" in results["faithfulness"].error


# ---------------------------------------------------------------- LLM judge reliability (fake models)
class FakeStructured:
    def __init__(self, behaviour: Any) -> None:
        self.behaviour = behaviour

    def with_structured_output(self, schema: type[BaseModel]) -> Any:
        return SimpleNamespace(invoke=lambda messages: self.behaviour(schema, messages))


def test_llm_judge_returns_validated_structured_output() -> None:
    seen: list[Any] = []

    def ok(schema: type[BaseModel], messages: list[Any]) -> Any:
        seen.append(messages)
        return ScoreVerdict(score=0.75, explanation="fine")

    verdict = LLMJudge(FakeStructured(ok)).ask("prompt text", ScoreVerdict)  # type: ignore[arg-type]
    assert verdict.score == 0.75 and "never instructions" in seen[0][0].content and seen[0][1].content == "prompt text"


@pytest.mark.parametrize(
    ("behaviour", "kind"),
    [
        (lambda s, m: time.sleep(2), "timeout"),
        (lambda s, m: (_ for _ in ()).throw(ConnectionError("network down")), "llm_failure"),
        (lambda s, m: None, "invalid_output"),
        (lambda s, m: {"score": 0.5}, "invalid_output"),
        (lambda s, m: ScoreVerdict.model_validate({"score": 3, "explanation": "x"}), "invalid_output"),
    ],
)
def test_llm_judge_failure_kinds(behaviour: Any, kind: str) -> None:
    with pytest.raises(JudgeError) as error:
        LLMJudge(FakeStructured(behaviour), timeout=0.2).ask("p", ScoreVerdict)  # type: ignore[arg-type]
    assert error.value.kind == kind


# ---------------------------------------------------------------- isolated evaluation target
def make_target(reply: str = "You get 24 days [1].", min_score: float | None = 0.6) -> EvaluationTarget:
    return EvaluationTarget(FakeChatModel(reply=reply), StaticRetriever(CHUNKS), min_score)


def test_target_returns_answer_full_contexts_sources_and_citations() -> None:
    out = make_target()({"question": "How much leave?"})
    assert out["answer"] == "You get 24 days [1]."
    assert [c["source"] for c in out["contexts"]] == ["leave_policy.md", "expense_policy.md"]
    assert out["contexts"][0]["text"] == LONG_TEXT, "the full chunk text, not the 200-character snippet"
    assert out["sources"] == ["expense_policy.md", "leave_policy.md"]
    assert out["citations"] == {"cited": [1], "invalid": []} and out["retrieval_empty"] is False


def test_target_uses_the_engine_relevance_threshold() -> None:
    out = make_target(min_score=0.95)({"question": "How much leave?"})  # StaticRetriever scores: 1.0, 0.9
    assert [c["source"] for c in out["contexts"]] == ["leave_policy.md"]


def test_target_never_holds_answers_for_review() -> None:
    target = make_target(reply="Refunds are possible.")  # sensitive and uncited: the app would hold it
    out = target({"question": "I want a refund for my order"})
    assert out["answer"] == "Refunds are possible."
    assert target.engine.review_policy is None and REVIEW_TOOL_NAME not in target.engine.llm.bound_tools  # type: ignore[attr-defined]


def test_target_has_no_tools_even_with_image_search_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_SEARCH_PROVIDER", "openverse")
    get_settings.cache_clear()
    target = build_target(llm=FakeChatModel(reply="ok [1]"), retriever=StaticRetriever(CHUNKS))
    assert target.engine.tools == [] and target.engine.approval_tools == frozenset()
    assert target.engine.llm.bound_tools == []  # type: ignore[attr-defined]


def test_target_writes_no_sessions_reviews_or_checkpoints(settings: Any) -> None:
    sessions_before = (
        sorted(p.name for p in settings.sessions_dir.parent.glob("*")) if settings.sessions_dir.exists() else []
    )
    target = build_target(llm=FakeChatModel(reply="ok [1]"), retriever=StaticRetriever(CHUNKS))
    target({"question": "How much leave?"})
    configured = get_settings()  # what the app would use (temp paths from conftest)
    assert not configured.checkpoint_path.exists(), "no SQLite checkpoint file"
    assert not any(configured.review_root.iterdir()), "no review records"
    assert type(target.engine.checkpointer).__name__ == "InMemorySaver"
    assert (
        not settings.sessions_dir.exists()
        or sorted(p.name for p in settings.sessions_dir.parent.glob("*")) == sessions_before
    )
    assert configured.human_review_enabled is True, "the app's HUMAN_REVIEW setting is untouched"


def test_target_traces_use_an_evaluation_session_id() -> None:
    with collect_runs() as runs:
        make_target()({"question": "How much leave?"})
    session = runs.traced_runs[0].extra["metadata"]["session_id"]
    assert session.startswith(EVAL_SESSION_PREFIX)


# ---------------------------------------------------------------- local runs and reports
def test_local_run_and_report(tmp_path: Path) -> None:
    examples = [example(), UNANSWERABLE, example(metadata__example_id="q03", inputs__question="Broken target?")]

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        if inputs["question"] == "Broken target?":
            raise TimeoutError("retriever timed out")
        if "pets" in inputs["question"]:
            return outputs("Pets are welcome on Fridays.", contexts=[])
        return outputs()

    runs = run_local(examples, target, ScriptedJudge(score=0.2))
    report = build_report(runs, {"dataset": DATASET_NAME, "mode": "local"})
    assert report["examples_total"] == 3 and report["target_errors"] == 1
    assert report["summary"]["faithfulness"]["errors"] == 1, "the failed target is an error, not a zero"
    assert report["summary"]["correctness"]["failed"] == 2 and report["summary"]["correctness"]["mean"] == 0.2
    patterns = {p["pattern"]: p["examples"] for p in report["failure_patterns"]}
    assert patterns["target run failed"] == ["q03"]
    assert patterns["answered an unanswerable question"] == ["q17"]
    assert patterns["claims not supported by the retrieved context"] == ["q01", "q17"]  # q17 invented an answer
    assert report["failed_by_category"] == {"direct_factual": 2, "unanswerable": 1}
    folder = write_report(report, "local-test", tmp_path)
    markdown = (folder / "report.md").read_text(encoding="utf-8")
    assert "| correctness | 0.20 |" in markdown and "ERROR — target failed (TimeoutError)" in markdown
    assert json.loads((folder / "report.json").read_text(encoding="utf-8"))["failed_examples"] == ["q01", "q17", "q03"]


def test_report_keeps_errors_out_of_the_means() -> None:
    good = MetricResult(key="faithfulness", score=1.0, explanation="ok")
    error = MetricResult(key="faithfulness", error="judge timeout: slow")
    runs = [
        ExampleRun(
            example_id="q01",
            category="direct_factual",
            question="a",
            is_unanswerable=False,
            metrics={"faithfulness": good},
        ),
        ExampleRun(
            example_id="q02",
            category="direct_factual",
            question="b",
            is_unanswerable=False,
            metrics={"faithfulness": error},
        ),
    ]
    summary = build_report(runs, {})["summary"]["faithfulness"]
    assert summary == {"mean": 1.0, "scored": 1, "passed": 1, "failed": 0, "errors": 1}


# ---------------------------------------------------------------- LangSmith adapters and scripts (mocked)
def ls_example(ex: EvalExample) -> Any:
    meta = {**ex.metadata.model_dump(mode="json"), "dataset_split": ["base"]}  # LangSmith adds its own keys
    return SimpleNamespace(inputs=ex.inputs.model_dump(), outputs=ex.outputs.model_dump(), metadata=meta)


def test_langsmith_evaluators_have_the_sdk_supported_signature() -> None:
    judge = ScriptedJudge(score=0.9)
    evaluators = langsmith_evaluators(judge)
    assert [e.__name__ for e in evaluators] == list(METRICS)
    for evaluator in evaluators:
        run_evaluator(evaluator)  # the installed SDK accepts the (run, example) form
    run = SimpleNamespace(outputs=outputs(), error=None)
    feedback = [e(run, ls_example(example())) for e in evaluators]
    assert all(f["score"] == 0.9 for f in feedback if f["key"] != "retrieval_relevance")
    assert len(judge.prompts) == 4, "one judge call per metric, not four"


def test_langsmith_evaluators_report_errors_explicitly() -> None:
    evaluator = langsmith_evaluators(ScriptedJudge())[3]
    failed_run = evaluator(SimpleNamespace(outputs=None, error="boom"), ls_example(example()))
    assert (
        failed_run["score"] is None and failed_run["value"] == "error" and "target run failed" in failed_run["comment"]
    )
    broken = ls_example(example())
    broken.outputs = {}
    no_reference = evaluator(SimpleNamespace(outputs=outputs(), error=None), broken)
    assert no_reference["value"] == "error" and "invalid dataset example" in no_reference["comment"]
    timeout = langsmith_evaluators(ScriptedJudge(error=JudgeError("timeout", "slow")))[0]
    assert timeout(SimpleNamespace(outputs=outputs(), error=None), ls_example(example()))["value"] == "error"


def never(*args: Any) -> Any:
    raise AssertionError("must not be called in a dry run")


def test_dataset_script_is_a_dry_run_without_confirm(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.create_dataset_main([], client_factory=never) == 0
    out = capsys.readouterr().out
    assert "EXTERNAL calls" in out and "Dry run: nothing was sent" in out and "19 example(s)" in out


def test_dataset_script_creates_dataset_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-placeholder-value")
    client = MagicMock()
    client.has_dataset.return_value = False
    client.create_dataset.return_value = SimpleNamespace(id="ds-1")
    assert cli.create_dataset_main(["--confirm"], client_factory=lambda: client) == 0
    client.create_dataset.assert_called_once()
    assert client.create_dataset.call_args.args[0] == DATASET_NAME
    sent = client.create_examples.call_args.kwargs
    assert sent["dataset_id"] == "ds-1" and len(sent["examples"]) == 19
    assert set(sent["examples"][0]) == {"inputs", "outputs", "metadata"}
    assert "test-placeholder-value" not in capsys.readouterr().out, "the key is never printed"


def test_dataset_script_skips_existing_and_never_overwrites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-placeholder-value")
    local = {e.id: e for e in load_dataset()}
    same = SimpleNamespace(
        inputs=local["q01"].inputs.model_dump(),
        outputs=local["q01"].outputs.model_dump(),
        metadata={"example_id": "q01"},
    )
    changed = SimpleNamespace(
        inputs=local["q02"].inputs.model_dump(), outputs={"reference_answer": "edited"}, metadata={"example_id": "q02"}
    )
    client = MagicMock()
    client.has_dataset.return_value = True
    client.read_dataset.return_value = SimpleNamespace(id="ds-1")
    client.list_examples.return_value = [same, changed]
    assert cli.create_dataset_main(["--confirm"], client_factory=lambda: client) == 3  # conflict reported
    uploaded = [e["metadata"]["example_id"] for e in client.create_examples.call_args.kwargs["examples"]]
    assert "q01" not in uploaded and "q02" not in uploaded and len(uploaded) == 17
    client.create_dataset.assert_not_called()


def test_dataset_script_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert cli.create_dataset_main(["--confirm"], client_factory=never) == 2


def test_evaluation_script_dry_run_reports_expected_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.run_evaluation_main([], client_factory=never, target_factory=never, judge_factory=never) == 0
    out = capsys.readouterr().out
    assert "may incur Gemini usage" in out and "19 answer call(s) + at most 76 judge call(s) = at most 95" in out
    assert "Dry run: no Gemini or LangSmith call was made" in out


def test_local_evaluation_sends_nothing_to_langsmith(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")  # restored by monkeypatch after the test
    code = cli.run_evaluation_main(
        ["--local", "--confirm", "--ids", "q01,q17"],
        client_factory=never,
        target_factory=lambda: lambda inputs: outputs(),
        judge_factory=lambda model, timeout: ScriptedJudge(),
        reports_dir=tmp_path,
    )
    assert code == 0
    report = json.loads(next(tmp_path.glob("*/report.json")).read_text(encoding="utf-8"))
    assert report["info"]["mode"] == "local" and [e["example_id"] for e in report["examples"]] == ["q01", "q17"]


def test_langsmith_evaluation_uses_supported_client_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-placeholder-value")
    local = {e.id: e for e in load_dataset()}
    remote = [ls_example(local[i]) for i in ("q01", "q05", "q17")]

    def fake_evaluate(target: Any, **kwargs: Any) -> FakeExperiment:
        """Mimics Client.evaluate: runs the target and evaluators, returns iterable rows with a url."""
        rows = []
        for ex in kwargs["data"]:
            out = target(ex.inputs)
            results = [
                EvaluationResult(**e(SimpleNamespace(outputs=out, error=None), ex)) for e in kwargs["evaluators"]
            ]
            rows.append(
                {
                    "run": SimpleNamespace(outputs=out, error=None),
                    "example": ex,
                    "evaluation_results": {"results": results},
                }
            )
        return FakeExperiment("https://smith.example/experiment", rows)

    client = MagicMock()
    client.has_dataset.return_value = True
    client.list_examples.return_value = remote
    client.evaluate.side_effect = fake_evaluate
    code = cli.run_evaluation_main(
        ["--confirm", "--ids", "q01,q17"],
        client_factory=lambda: client,
        target_factory=lambda: lambda inputs: outputs(),
        judge_factory=lambda model, timeout: ScriptedJudge(),
        reports_dir=tmp_path,
    )
    assert code == 0
    kwargs = client.evaluate.call_args.kwargs
    assert [ex.metadata["example_id"] for ex in kwargs["data"]] == ["q01", "q17"]
    assert [e.__name__ for e in kwargs["evaluators"]] == list(METRICS)
    assert kwargs["max_concurrency"] == 1 and kwargs["metadata"]["prompt_version"]
    report = json.loads(next(tmp_path.glob("*/report.json")).read_text(encoding="utf-8"))
    assert report["info"]["experiment_url"] == "https://smith.example/experiment" and report["examples_total"] == 2


class FakeExperiment:
    """Minimal stand-in for langsmith's ExperimentResults: iterable rows and a url."""

    def __init__(self, url: str, rows: list[dict[str, Any]]) -> None:
        self.url, self._rows = url, rows

    def __iter__(self) -> Any:
        return iter(self._rows)


def test_missing_langsmith_dataset_stops_before_any_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-placeholder-value")
    client = MagicMock()
    client.has_dataset.return_value = False
    with pytest.raises(SystemExit, match="does not exist"):
        cli.run_evaluation_main(
            ["--confirm"],
            client_factory=lambda: client,
            target_factory=lambda: never,
            judge_factory=lambda model, timeout: ScriptedJudge(),
            reports_dir=tmp_path,
        )
    client.evaluate.assert_not_called()
