"""Milestone 8 — Tools and the tool loop."""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool

from app.chat.engine import get_engine
from app.tools import get_tools
from tests.fakes import TOOL_TRIGGER, FakeChatModel

pytestmark = pytest.mark.contract


def _tool(name: str) -> BaseTool:
    tools = {t.name: t for t in get_tools()}
    assert name in tools, f"get_tools() must include {name!r}"
    return tools[name]


def test_registry() -> None:
    tools = get_tools()
    assert tools and all(isinstance(t, BaseTool) for t in tools)
    assert len({t.name for t in tools}) == len(tools)
    assert all(len(t.description) >= 20 for t in tools), "describe each tool for the LLM"


def test_calculator() -> None:
    calc = _tool("calculator")
    assert "14" in str(calc.invoke({"expression": "2 + 3 * 4"}))
    assert str(calc.invoke({"expression": "__import__('os').getcwd()"})).startswith("Error")


def test_engine_runs_tool_then_answers() -> None:
    llm = FakeChatModel(reply="DONE", tool_name="calculator", tool_args={"expression": "6 * 7"})
    engine = get_engine(llm=llm, retriever=None, tools=get_tools())
    result = engine.invoke(message=f"{TOOL_TRIGGER} what is 6*7?", history=[], session_id="s1")
    assert "calculator" in llm.bound_tools, "tools must be bound to the LLM"
    assert result.answer.startswith("DONE | tool said:") and "42" in result.answer
