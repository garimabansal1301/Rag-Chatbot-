"""Tool and tool-loop guarantees the M8 contract tests do not check. Fake LLMs only."""

from __future__ import annotations

import ast
import importlib
import inspect
import time
from datetime import datetime
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.chat.engine import MAX_TOOL_ROUNDS, SYSTEM_PROMPT, TOOL_LIMIT_PROMPT, get_engine
from app.tools import get_tools
from app.tools.calculator import calculate
from app.tools.current_datetime import current_datetime
from tests.fakes import TOOL_TRIGGER, FakeChatModel, StaticRetriever, make_chunk


# ---------------------------------------------------------------- calculator
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("-3 + +5", "2"),
        ("7 / 2", "3.5"),
        ("0.1 + 0.2", "0.3"),
        ("7 // 2", "3"),
        ("-7 % 3", "2"),
        ("2 ** 10", "1024"),
        ("2 ** -1", "0.5"),
        ("2 ** 100", str(2**100)),
        ("  (24 - 5) * 1.5 ", "28.5"),
    ],
)
def test_calculator_arithmetic(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "open('x')",
        "x + 1",
        "(1).__class__",
        "'a' * 3",
        "True + 1",
        "[1, 2]",
        "lambda: 1",
        "1 < 2",
        "2 ^ 3",
        "",
        "2 +",
        "1 / 0",
        "5 % 0",
        "10 ** 1001",
        "(-8) ** 0.5",
        "1e309",
        "1" * 201,
    ],
)
def test_calculator_rejects_anything_but_bounded_arithmetic(expression: str) -> None:
    assert calculate(expression).startswith("Error")


def test_calculator_refuses_huge_powers_quickly() -> None:
    start = time.perf_counter()
    assert calculate("9 ** 9 ** 9").startswith("Error")
    assert calculate("2 ** 2 ** 2 ** 2 ** 2").startswith("Error")
    assert time.perf_counter() - start < 1.0


def test_calculator_hints_at_power_operator() -> None:
    assert "**" in calculate("2 ^ 3")


def test_calculator_module_never_calls_eval_or_exec() -> None:
    # import_module, because the package attribute `app.tools.calculator` is the tool, not the module
    tree = ast.parse(inspect.getsource(importlib.import_module("app.tools.calculator")))
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not called & {"eval", "exec", "compile", "__import__"}


# ---------------------------------------------------------------- current_datetime and registry
def test_current_datetime_is_iso_8601_with_timezone() -> None:
    value = datetime.fromisoformat(current_datetime.invoke({}))
    assert value.tzinfo is not None
    assert abs((datetime.now().astimezone() - value).total_seconds()) < 5


def test_registry_has_exactly_the_required_tools_in_a_fresh_list() -> None:
    assert [t.name for t in get_tools()] == ["calculator", "current_datetime"]
    get_tools().clear()
    assert len(get_tools()) == 2


# ---------------------------------------------------------------- engine tool loop
@tool
def broken_lookup(query: str) -> str:
    """Look something up (always fails, for testing error handling)."""
    raise ConnectionError("upstream unavailable")


class LoopingModel(FakeChatModel):
    """Requests a tool on every call, unless bound with tool_choice {"type": "none"}."""

    answer_only: bool = False

    def bind_tools(self, tools: Any, **kwargs: Any) -> LoopingModel:  # type: ignore[override]
        super().bind_tools(tools)
        return self.model_copy(update={"answer_only": kwargs.get("tool_choice") == {"type": "none"}})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self.answer_only:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="FINAL ANSWER"))])
        call = {"name": "calculator", "args": {"expression": "1 + 1"}, "id": f"call_{len(self.calls)}"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


def test_tool_exception_becomes_an_error_result_not_a_crash() -> None:
    llm = FakeChatModel(reply="DONE", tool_name="broken_lookup", tool_args={"query": "x"})
    result = get_engine(llm=llm, retriever=None, tools=[broken_lookup]).invoke(f"{TOOL_TRIGGER} look up x", [], "s1")
    assert result.answer.startswith("DONE | tool said: Error: the tool failed (ConnectionError")


def test_tool_loop_is_bounded_and_ends_with_an_answer() -> None:
    llm = LoopingModel()
    result = get_engine(llm=llm, retriever=None, tools=get_tools()).invoke("keep calculating", [], "s1")
    assert result.answer == "FINAL ANSWER"
    tool_results = [m for m in llm.calls[-1] if isinstance(m, ToolMessage)]
    assert len(tool_results) == MAX_TOOL_ROUNDS and all(m.content == "2" for m in tool_results)
    assert TOOL_LIMIT_PROMPT in str(llm.calls[-1][0].content)


def test_tool_turn_keeps_retrieval_once_and_sources() -> None:
    retriever = StaticRetriever([make_chunk("Meals are capped at 50 USD.", source="expenses.md")])
    llm = FakeChatModel(reply="DONE", tool_name="calculator", tool_args={"expression": "50 * 3"})
    result = get_engine(llm=llm, retriever=retriever, tools=get_tools()).invoke(f"{TOOL_TRIGGER} 3 meals?", [], "s1")
    assert retriever.queries == [f"{TOOL_TRIGGER} 3 meals?"], "retrieve once per turn, not per tool round"
    assert "150" in result.answer and [s.source for s in result.sources] == ["expenses.md"]
    assert all("Meals are capped" in str(call[0].content) for call in llm.calls), "context in every LLM call"


def test_no_tools_binds_nothing() -> None:
    llm = FakeChatModel(reply="plain")
    result = get_engine(llm=llm, retriever=None, tools=[]).invoke(f"{TOOL_TRIGGER} hi", [], "s1")
    assert result.answer == "plain" and llm.bound_tools == []


# ---------------------------------------------------------------- tool guidance in the system prompt
def _system_prompt(tools: list[Any], retriever: StaticRetriever | None = None) -> str:
    llm = FakeChatModel()
    get_engine(llm=llm, retriever=retriever, tools=tools).invoke("What does a USB-C connector look like?", [], "s1")
    return str(llm.calls[0][0].content)


def test_without_tools_the_system_prompt_is_unchanged() -> None:
    assert _system_prompt([]) == SYSTEM_PROMPT


def test_tool_guidance_names_only_the_bound_tools() -> None:
    prompt = _system_prompt(get_tools())
    assert "calculator for arithmetic" in prompt and "current_datetime for the current date or time" in prompt
    assert "image_search" not in prompt, "image search is only mentioned when it is switched on"
    assert "instead of only saying that the documents do not cover it" in prompt


def test_tool_guidance_includes_image_search_when_enabled() -> None:
    from app.tools.image_search import make_image_search_tool

    class NoImages:
        def search(self, query: str, limit: int = 3) -> list[Any]:
            return []

    prompt = _system_prompt([*get_tools(), make_image_search_tool(NoImages())])
    assert "image_search when seeing what something looks like would help" in prompt


def test_tool_guidance_keeps_document_context_and_citation_rules() -> None:
    retriever = StaticRetriever([make_chunk("Meals are capped at 50 USD.", source="expenses.md")])
    prompt = _system_prompt(get_tools(), retriever)
    assert prompt.startswith(SYSTEM_PROMPT)
    assert "cite them like [1]" in prompt and "Meals are capped at 50 USD." in prompt
    assert prompt.index("</documents>") < prompt.index("You can call tools"), "tool guidance follows the documents"
    assert "Still cite document excerpts like [1]" in prompt
