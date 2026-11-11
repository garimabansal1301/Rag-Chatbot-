"""The local, version-controlled evaluation dataset (JSONL, one example per line).

Each line: {"inputs": {"question"}, "outputs": {"reference_answer"}, "metadata": {"example_id", "category",
"expected_sources", "expected_facts", "is_unanswerable", "evidence": [{"source", "quote"}]}}.

Reference answers are written from the verbatim `evidence` quotes; `check_evidence` proves every quote is in
its source document and every expected fact is in a quote, so the ground truth is traceable to the documents.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import PROJECT_ROOT

DATASET_NAME = "RAG-Chatbot-Evaluation-v1"
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / f"{DATASET_NAME}.jsonl"
DATASET_DESCRIPTION = (
    "Questions about the chatbot's knowledge-base documents with manually verified reference answers. "
    "Contains no document text beyond short evidence quotes and no chat-session data."
)
Category = Literal["direct_factual", "multi_chunk", "multi_document", "scenario", "careful_grounding", "unanswerable"]
_ID = re.compile(r"^q\d{2,3}$")


class DatasetError(ValueError):
    pass


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    quote: str = Field(min_length=10)


class ExampleInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=5, max_length=1000)


class ExampleOutputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_answer: str = Field(min_length=10, max_length=4000)


class ExampleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    category: Category
    expected_sources: list[str]
    expected_facts: list[str]
    is_unanswerable: bool
    evidence: list[Evidence]

    @model_validator(mode="after")
    def _check(self) -> ExampleMetadata:
        if not _ID.match(self.example_id):
            raise ValueError(f"example_id must look like q01, not {self.example_id!r}")
        if self.is_unanswerable != (self.category == "unanswerable"):
            raise ValueError("is_unanswerable must be true exactly for the 'unanswerable' category")
        if self.is_unanswerable and (self.expected_sources or self.expected_facts):
            raise ValueError("unanswerable examples have no expected sources or facts")
        if not self.is_unanswerable and (not self.expected_sources or not self.evidence):
            raise ValueError("answerable examples need expected sources and evidence quotes")
        if not {e.source for e in self.evidence} >= set(self.expected_sources):
            raise ValueError("every expected source needs at least one evidence quote")
        return self


class EvalExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: ExampleInputs
    outputs: ExampleOutputs
    metadata: ExampleMetadata

    @property
    def id(self) -> str:
        return self.metadata.example_id

    def to_langsmith(self) -> dict[str, Any]:
        """The example as sent to LangSmith: question, reference answer and metadata (no document text
        beyond the short evidence quotes)."""
        return self.model_dump(mode="json")


def normalize(text: str) -> str:
    """Lower-case, Markdown emphasis removed, whitespace collapsed (PDF text has a newline after every word)."""
    return " ".join(text.replace("*", "").replace("`", "").replace("’", "'").split()).casefold()


def parse_lines(lines: Iterable[str]) -> list[EvalExample]:
    """Parse and validate JSONL lines. Raises DatasetError listing every problem (line numbers included)."""
    examples: list[EvalExample] = []
    problems: list[str] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            examples.append(EvalExample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            problems.append(f"line {number}: {_first_error(exc)}")
    problems += duplicate_problems(examples)
    if problems:
        raise DatasetError("Invalid evaluation dataset:\n  " + "\n  ".join(problems))
    if not examples:
        raise DatasetError("The evaluation dataset is empty.")
    return examples


def _first_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        error = exc.errors()[0]
        location = ".".join(str(part) for part in error["loc"]) or "record"
        return f"{location}: {error['msg']}"
    return f"not valid JSON ({exc})"


def duplicate_problems(examples: list[EvalExample]) -> list[str]:
    """Duplicate ids, and duplicate questions (compared after normalising case and whitespace)."""
    problems: list[str] = []
    ids: set[str] = set()
    questions: dict[str, str] = {}
    for example in examples:
        if example.id in ids:
            problems.append(f"duplicate example_id {example.id}")
        ids.add(example.id)
        key = normalize(example.inputs.question)
        if key in questions:
            problems.append(f"{example.id} repeats the question of {questions[key]}")
        questions.setdefault(key, example.id)
    return problems


def load_dataset(path: Path = DATASET_PATH) -> list[EvalExample]:
    try:
        return parse_lines(path.read_text(encoding="utf-8").splitlines())
    except OSError as exc:
        raise DatasetError(f"Cannot read the evaluation dataset ({type(exc).__name__})") from None


def check_evidence(examples: list[EvalExample], documents: dict[str, str]) -> list[str]:
    """Problems found when checking the ground truth against the documents ({file name: text}).

    Every evidence quote must appear in its source document, and every expected fact in one of the
    example's quotes. An empty list means the dataset is consistent with the documents.
    """
    normalized = {name: normalize(text) for name, text in documents.items()}
    problems = []
    for example in examples:
        quotes = [normalize(e.quote) for e in example.metadata.evidence]
        for evidence, quote in zip(example.metadata.evidence, quotes, strict=True):
            if evidence.source not in normalized:
                problems.append(f"{example.id}: unknown source document {evidence.source!r}")
            elif quote not in normalized[evidence.source]:
                problems.append(f"{example.id}: quote not found in {evidence.source}: {evidence.quote[:60]!r}")
        for fact in example.metadata.expected_facts:
            if not any(normalize(fact) in quote for quote in quotes):
                problems.append(f"{example.id}: expected fact {fact!r} is not in any evidence quote")
    return problems


def select(examples: list[EvalExample], ids: list[str] | None = None, limit: int | None = None) -> list[EvalExample]:
    """A subset: the given example ids (all must exist), then at most `limit` examples."""
    chosen = examples
    if ids:
        known = {e.id for e in examples}
        unknown = sorted(set(ids) - known)
        if unknown:
            raise DatasetError(f"Unknown example id(s): {', '.join(unknown)}")
        chosen = [e for e in examples if e.id in set(ids)]
    if limit is not None:
        if limit < 1:
            raise DatasetError("--limit must be at least 1")
        chosen = chosen[:limit]
    return chosen
