"""Milestone 1 — LangGraph engine with an LLM. Run: uv run pytest -m contract tests/contract/test_m1_engine.py"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.chat.engine import get_engine
from app.models import EngineResult, Message
from tests.fakes import FakeChatModel

pytestmark = pytest.mark.contract


def test_engine_has_compiled_graph_and_answers() -> None:
    llm = FakeChatModel(reply="Hello from the fake model")
    engine = get_engine(llm=llm, retriever=None, tools=[])
    assert hasattr(engine.graph, "invoke"), "engine.graph should be the compiled LangGraph graph"
    result = engine.invoke(message="hi", history=[], session_id="s1")
    assert isinstance(result, EngineResult)
    assert result.answer == "Hello from the fake model"


def test_engine_sends_history_and_new_message_to_llm() -> None:
    llm = FakeChatModel()
    engine = get_engine(llm=llm, retriever=None, tools=[])
    history = [Message(role="user", content="My name is Priya"), Message(role="assistant", content="Hi Priya")]
    engine.invoke(message="What is my name?", history=history, session_id="s1")
    seen = llm.prompt_text()
    assert "My name is Priya" in seen, "earlier history must reach the LLM"
    assert any(isinstance(m, HumanMessage) and m.content == "What is my name?" for m in llm.calls[-1])
