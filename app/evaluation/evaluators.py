"""The four metrics, each a score from 0 to 1 with an explanation, or an evaluation ERROR (score None).

  faithfulness         claims in the answer supported by the retrieved contexts (LLM judge, given the contexts)
  answer_relevance     the answer responds to the question (LLM judge)
  retrieval_relevance  share of retrieved chunks that help answer the question (LLM judge picks the chunks;
                       the expected sources are given to it as supporting evidence, not as the rule)
  correctness          agreement with the verified reference answer and expected facts (LLM judge, given the
                       reference); for unanswerable questions: whether the chatbot abstains

Deterministic rules decide the cases a judge is not needed for (empty answer, empty retrieval, a clear
abstention). Errors (judge timeout or failure, invalid judge output, missing reference, failed target run)
are reported as errors, never as scores.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, model_validator

from app.evaluation import prompts
from app.evaluation.dataset import EvalExample, ExampleMetadata
from app.evaluation.judge import ChunkVerdict, Judge, JudgeError, ScoreVerdict

MetricKey = Literal["faithfulness", "answer_relevance", "retrieval_relevance", "correctness"]
METRICS: tuple[MetricKey, ...] = ("faithfulness", "answer_relevance", "retrieval_relevance", "correctness")
PASS_THRESHOLD = 0.7
MAX_CONTEXT_CHARS = 2000  # per chunk in judge prompts (chunks are ~800 characters)

# Phrases with which the chatbot says the documents do not answer (it is prompted to say exactly that).
_ABSTAIN = re.compile(
    r"\b(do(es)?n['’]?t|do(es)? not|did not|didn['’]?t) (cover|mention|say|state|contain|include|specify|provide)"
    r"|\bnot (covered|mentioned|stated|specified|included|found) (in|by)\b"
    r"|\b(no|not any) (information|details?|mention)\b"
    r"|\b(can(no|')t|could not|couldn['’]?t|unable to) find\b",
    re.IGNORECASE,
)


class MetricResult(BaseModel):
    key: MetricKey
    score: float | None = None
    explanation: str = ""
    error: str | None = None  # set => evaluation error; score is then None
    judged: bool = False  # True if an LLM judge was called

    @model_validator(mode="after")
    def _check(self) -> MetricResult:
        if self.error is None:
            if self.score is None or not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
                raise ValueError(f"{self.key}: a score from 0 to 1 is required unless there is an error")
        elif self.score is not None:
            raise ValueError(f"{self.key}: an errored metric has no score")
        return self

    @property
    def passed(self) -> bool | None:
        """None for errors (neither passed nor failed)."""
        return None if self.score is None else self.score >= PASS_THRESHOLD


def abstains(answer: str) -> bool:
    return bool(_ABSTAIN.search(answer))


def _error(key: MetricKey, message: str, judged: bool = False) -> MetricResult:
    return MetricResult(key=key, error=message, judged=judged)


def _ok(key: MetricKey, score: float, explanation: str, judged: bool = False) -> MetricResult:
    return MetricResult(key=key, score=score, explanation=explanation, judged=judged)


def _ask(judge: Judge, key: MetricKey, prompt: str) -> MetricResult:
    try:
        verdict = judge.ask(prompt, ScoreVerdict)
    except JudgeError as exc:
        return _error(key, f"judge {exc}", judged=True)
    return _ok(key, verdict.score, verdict.explanation, judged=True)


def _answer(outputs: dict[str, Any]) -> str:
    return str(outputs.get("answer") or "").strip()


def _contexts(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = outputs.get("contexts")
    if not isinstance(contexts, list) or not all(isinstance(c, dict) and "text" in c for c in contexts):
        raise ValueError("target output has no valid 'contexts' list")
    return contexts


def _format_contexts(contexts: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.get('source', '?')}) {str(c['text'])[:MAX_CONTEXT_CHARS]}" for i, c in enumerate(contexts, start=1)
    )


# --------------------------------------------------------------------------- the four metrics
def faithfulness(question: str, outputs: dict[str, Any], judge: Judge) -> MetricResult:
    key: MetricKey = "faithfulness"
    answer = _answer(outputs)
    if not answer:
        return _ok(key, 0.0, "The chatbot returned an empty answer.")
    contexts = _contexts(outputs)
    if not contexts:
        if abstains(answer):
            return _ok(key, 1.0, "Nothing was retrieved and the answer says the documents do not cover it.")
        return _ok(key, 0.0, "The answer makes claims, but no context was retrieved to support them.")
    prompt = prompts.FAITHFULNESS.format(question=question, context=_format_contexts(contexts), answer=answer)
    return _ask(judge, key, prompt)


def answer_relevance(question: str, outputs: dict[str, Any], judge: Judge) -> MetricResult:
    key: MetricKey = "answer_relevance"
    answer = _answer(outputs)
    if not answer:
        return _ok(key, 0.0, "The chatbot returned an empty answer.")
    return _ask(judge, key, prompts.ANSWER_RELEVANCE.format(question=question, answer=answer))


def retrieval_relevance(
    question: str, outputs: dict[str, Any], expected_sources: list[str], is_unanswerable: bool, judge: Judge
) -> MetricResult:
    key: MetricKey = "retrieval_relevance"
    contexts = _contexts(outputs)
    if not contexts:
        if is_unanswerable:
            return _ok(key, 1.0, "No chunk passed the relevance threshold, as expected for an unanswerable question.")
        return _ok(key, 0.0, f"Nothing was retrieved; expected content from {', '.join(expected_sources)}.")
    expectation = (
        prompts.RETRIEVAL_UNANSWERABLE
        if is_unanswerable
        else prompts.RETRIEVAL_EXPECTED.format(sources=", ".join(expected_sources))
    )
    prompt = prompts.RETRIEVAL_RELEVANCE.format(
        question=question, context=_format_contexts(contexts), expectation=expectation
    )
    try:
        verdict = judge.ask(prompt, ChunkVerdict)
    except JudgeError as exc:
        return _error(key, f"judge {exc}", judged=True)
    chosen = set(verdict.relevant_chunks)
    if not chosen <= set(range(1, len(contexts) + 1)):
        return _error(key, f"judge invalid_output: chunk numbers {sorted(chosen)} outside 1-{len(contexts)}", True)
    retrieved = sorted({str(c.get("source")) for c in contexts})
    missing = [s for s in expected_sources if s not in retrieved]
    note = f" Expected sources not retrieved: {', '.join(missing)}." if missing else ""
    explanation = f"{len(chosen)} of {len(contexts)} retrieved chunks are relevant. {verdict.explanation}{note}"
    return _ok(key, len(chosen) / len(contexts), explanation, judged=True)


def correctness(
    question: str,
    outputs: dict[str, Any],
    reference_answer: str | None,
    expected_facts: list[str],
    is_unanswerable: bool,
    judge: Judge,
) -> MetricResult:
    key: MetricKey = "correctness"
    if not (reference_answer or "").strip():
        return _error(key, "missing reference answer")
    answer = _answer(outputs)
    if not answer:
        return _ok(key, 0.0, "The chatbot returned an empty answer.")
    if is_unanswerable:
        if abstains(answer):
            return _ok(key, 1.0, "Correctly says the documents do not answer this question.")
        prompt = prompts.CORRECTNESS_UNANSWERABLE.format(question=question, reference=reference_answer, answer=answer)
        return _ask(judge, key, prompt)
    facts = "; ".join(expected_facts) or "(see reference)"
    prompt = prompts.CORRECTNESS.format(question=question, reference=reference_answer, facts=facts, answer=answer)
    return _ask(judge, key, prompt)


def evaluate_metric(key: MetricKey, example: EvalExample, outputs: dict[str, Any], judge: Judge) -> MetricResult:
    """One metric for one example (at most one judge call); a malformed target output is an error."""
    question, meta = example.inputs.question, example.metadata
    try:
        if key == "faithfulness":
            return faithfulness(question, outputs, judge)
        if key == "answer_relevance":
            return answer_relevance(question, outputs, judge)
        if key == "retrieval_relevance":
            return retrieval_relevance(question, outputs, meta.expected_sources, meta.is_unanswerable, judge)
        return correctness(
            question, outputs, example.outputs.reference_answer, meta.expected_facts, meta.is_unanswerable, judge
        )
    except ValueError as exc:  # malformed target output
        return _error(key, f"invalid target output: {exc}")


def evaluate_example(example: EvalExample, outputs: dict[str, Any], judge: Judge) -> dict[MetricKey, MetricResult]:
    """All four metrics for one example (at most four judge calls)."""
    return {key: evaluate_metric(key, example, outputs, judge) for key in METRICS}


# --------------------------------------------------------------------------- LangSmith adapters
def to_feedback(result: MetricResult) -> dict[str, Any]:
    """LangSmith feedback: a score with a comment, or an explicit error marker (value "error", no score)."""
    if result.error is not None:
        return {"key": result.key, "score": None, "value": "error", "comment": f"EVALUATION ERROR: {result.error}"}
    return {"key": result.key, "score": result.score, "comment": result.explanation}


def example_from_langsmith(
    inputs: dict[str, Any], outputs: dict[str, Any] | None, metadata: dict[str, Any] | None
) -> EvalExample:
    """Rebuild a validated example from a LangSmith Example (LangSmith adds its own metadata keys; ignored)."""
    known = set(ExampleMetadata.model_fields)
    return EvalExample.model_validate(
        {
            "inputs": inputs,
            "outputs": outputs or {},
            "metadata": {k: v for k, v in (metadata or {}).items() if k in known},
        }
    )


def langsmith_evaluators(judge: Judge) -> list[Callable[[Any, Any], dict[str, Any]]]:
    """One evaluator per metric, in the SDK's supported `(run, example)` form."""

    def make(key: MetricKey) -> Callable[[Any, Any], dict[str, Any]]:
        def evaluator(run: Any, example: Any) -> dict[str, Any]:
            if getattr(run, "error", None) or not getattr(run, "outputs", None):
                return to_feedback(_error(key, "the target run failed, so there is nothing to evaluate"))
            try:
                parsed = example_from_langsmith(example.inputs, example.outputs, example.metadata)
            except ValidationError as exc:
                return to_feedback(_error(key, f"invalid dataset example ({exc.error_count()} problem(s))"))
            return to_feedback(evaluate_metric(key, parsed, dict(run.outputs), judge))

        evaluator.__name__ = key
        return evaluator

    return [make(key) for key in METRICS]
