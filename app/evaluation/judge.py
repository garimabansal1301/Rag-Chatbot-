"""The LLM judge: one structured-output call per question to the judge model, with a hard timeout.

Failures are raised as JudgeError with a kind ("timeout", "llm_failure", "invalid_output"); the evaluators
turn them into evaluation ERRORS, never into scores. Scores outside 0-1 are invalid output, not clamped.
"""

from __future__ import annotations

import contextvars
import math
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Literal, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.evaluation.prompts import JUDGE_SYSTEM

JUDGE_TIMEOUT_SECONDS = 90.0
JudgeErrorKind = Literal["timeout", "llm_failure", "invalid_output"]
T = TypeVar("T", bound=BaseModel)


class JudgeError(RuntimeError):
    def __init__(self, kind: JudgeErrorKind, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class ScoreVerdict(BaseModel):
    """Structured judge output for faithfulness, answer relevance and correctness."""

    score: float = Field(ge=0.0, le=1.0, description="Score from 0 to 1.")
    explanation: str = Field(min_length=1, max_length=2000, description="Why this score, in 1-4 sentences.")

    @field_validator("score")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be a finite number")
        return value


class ChunkVerdict(BaseModel):
    """Structured judge output for retrieval relevance (the score is computed from it, not chosen by the judge)."""

    relevant_chunks: list[int] = Field(description="Numbers of the chunks that help answer the question.")
    explanation: str = Field(min_length=1, max_length=2000)


class Judge(Protocol):
    def ask(self, prompt: str, schema: type[T]) -> T:
        """Return the judge's answer to `prompt` as an instance of `schema`, or raise JudgeError."""
        ...


class LLMJudge:
    """A judge backed by a LangChain chat model with structured output (e.g. Gemini)."""

    def __init__(self, llm: BaseChatModel, timeout: float = JUDGE_TIMEOUT_SECONDS) -> None:
        self.llm = llm
        self.timeout = timeout

    def ask(self, prompt: str, schema: type[T]) -> T:
        runnable = self.llm.with_structured_output(schema)
        messages = [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]
        pool = ThreadPoolExecutor(max_workers=1)
        context = contextvars.copy_context()  # the caller's tracing context (e.g. tracing off) applies here too
        try:
            result = pool.submit(context.run, runnable.invoke, messages).result(timeout=self.timeout)
        except FutureTimeout:
            raise JudgeError("timeout", f"no answer within {self.timeout:g} s") from None
        except ValidationError as exc:
            raise JudgeError("invalid_output", f"{exc.error_count()} validation error(s)") from None
        except Exception as exc:  # provider errors, parsing errors, network errors
            kind: JudgeErrorKind = "invalid_output" if "OutputParser" in type(exc).__name__ else "llm_failure"
            raise JudgeError(kind, type(exc).__name__) from None
        finally:
            pool.shutdown(wait=False)  # a timed-out call is abandoned, not awaited
        if not isinstance(result, schema):
            raise JudgeError("invalid_output", f"expected {schema.__name__}, got {type(result).__name__}")
        return result


def build_judge(model: str | None = None, timeout: float = JUDGE_TIMEOUT_SECONDS) -> LLMJudge:
    """The judge for live runs: the app's configured LLM, or another model of the same provider (Gemini only).
    Building it makes no network call."""
    if model is None:
        from app.chat.llm import get_llm

        return LLMJudge(get_llm(), timeout)
    from app.config import get_settings

    if get_settings().llm_provider.strip().lower() != "gemini":
        raise ValueError("--judge-model is supported for LLM_PROVIDER=gemini only")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return LLMJudge(ChatGoogleGenerativeAI(model=model, request_timeout=timeout), timeout)
