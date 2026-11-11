"""Provider selection in get_llm and Gemini tool calling. Dummy keys and a fake Gemini client only: no network."""

from __future__ import annotations

from typing import Any

import pytest
from google.genai import types
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.chat.engine import MAX_TOOL_ROUNDS, GraphEngine
from app.chat.llm import LLM_TIMEOUT_SECONDS, get_llm
from app.config import get_settings
from app.tools import get_tools

DUMMY_KEY = "dummy-key-not-real"  # never a real key: these tests must not reach the API
KEY_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI")


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Select a provider/model; every key variable starts unset, hiding any real key loaded from .env."""
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)

    def select(name: str, model: str) -> None:
        monkeypatch.setenv("LLM_PROVIDER", name)
        monkeypatch.setenv("LLM_MODEL", model)
        get_settings.cache_clear()

    yield select
    get_settings.cache_clear()


# ---------------------------------------------------------------- get_llm
def test_gemini_provider_builds_chat_google_model(provider: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", DUMMY_KEY)
    provider(" Gemini ", "gemini-3.8-flash")
    llm = get_llm()
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == "gemini-3.8-flash" and llm.timeout == LLM_TIMEOUT_SECONDS


@pytest.mark.parametrize("variable", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
def test_gemini_reads_either_key_variable(provider: Any, monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.setenv(variable, DUMMY_KEY)
    provider("gemini", "gemini-3.8-flash")
    assert isinstance(get_llm(), ChatGoogleGenerativeAI)


def test_gemini_prefers_google_api_key(provider: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-google")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini")
    provider("gemini", "gemini-3.8-flash")
    llm = get_llm()
    assert isinstance(llm, ChatGoogleGenerativeAI) and llm.google_api_key is not None
    assert llm.google_api_key.get_secret_value() == "dummy-google"


def test_gemini_without_a_key_fails_clearly(provider: Any) -> None:
    provider("gemini", "gemini-3.8-flash")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY/GEMINI_API_KEY"):
        get_llm()


def test_anthropic_provider_still_works(provider: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    provider("anthropic", "claude-sonnet-5-5")
    assert isinstance(get_llm(), ChatAnthropic)


def test_unknown_provider_and_missing_model_are_rejected(provider: Any) -> None:
    provider("palm", "x")
    with pytest.raises(ValueError, match="'anthropic', 'openai' or 'gemini'"):
        get_llm()
    provider("gemini", " ")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        get_llm()


# ---------------------------------------------------------------- answer-only round per provider
def test_answer_only_round_uses_each_providers_format(provider: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    provider("gemini", "gemini-3.8-flash")
    gemini = GraphEngine(llm=get_llm(), retriever=None, tools=get_tools())
    provider("anthropic", "claude-sonnet-5-5")
    claude = GraphEngine(llm=get_llm(), retriever=None, tools=get_tools())
    assert gemini.llm_answer_only.kwargs["tool_choice"] == "none"  # type: ignore[attr-defined]
    assert claude.llm_answer_only.kwargs["tool_choice"] == {"type": "none"}  # type: ignore[attr-defined]
    assert "tool_choice" not in gemini.llm_with_tools.kwargs  # type: ignore[attr-defined]


def test_gemini_requests_keep_tools_and_switch_mode_to_none(provider: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", DUMMY_KEY)
    provider("gemini", "gemini-3.8-flash")
    llm = get_llm()
    engine = GraphEngine(llm=llm, retriever=None, tools=get_tools())
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="2+2?"),
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "c1"}]),
        ToolMessage(content="4", tool_call_id="c1", name="calculator"),
    ]
    for binding, mode in (
        (engine.llm_with_tools, None),
        (engine.llm_answer_only, types.FunctionCallingConfigMode.NONE),
    ):
        config = llm._prepare_request(messages, **binding.kwargs)["config"]  # type: ignore[attr-defined]
        names = [d.name for t in config.tools for d in t.function_declarations]
        assert names == ["calculator", "current_datetime"]
        assert (config.tool_config.function_calling_config.mode if config.tool_config else None) == mode


# ---------------------------------------------------------------- full tool loop with a fake Gemini client
def _response(*parts: types.Part) -> types.GenerateContentResponse:
    content = types.Content(role="model", parts=list(parts))
    candidate = types.Candidate(content=content, finish_reason=types.FinishReason.STOP)
    return types.GenerateContentResponse(candidates=[candidate])


class FakeGeminiModels:
    """Stands in for `client.models`: asks for the calculator until it sees a tool result or mode NONE."""

    def __init__(self, loop_forever: bool = False) -> None:
        self.loop_forever = loop_forever
        self.requests: list[dict[str, Any]] = []

    def generate_content(self, **request: Any) -> types.GenerateContentResponse:
        self.requests.append(request)
        tool_config = request["config"].tool_config
        answer_only = tool_config is not None and tool_config.function_calling_config.mode == "NONE"
        last = request["contents"][-1].parts[-1]
        if answer_only or (last.function_response is not None and not self.loop_forever):
            result = last.function_response.response if last.function_response else None
            return _response(types.Part(text=f"The answer is {result}"))
        call = types.FunctionCall(name="calculator", args={"expression": "6 * 7"}, id=f"c{len(self.requests)}")
        return _response(types.Part(function_call=call, thought_signature=b"sig"))


def _gemini_engine(provider: Any, monkeypatch: pytest.MonkeyPatch, fake: FakeGeminiModels) -> GraphEngine:
    monkeypatch.setenv("GOOGLE_API_KEY", DUMMY_KEY)
    provider("gemini", "gemini-3.8-flash")
    llm = get_llm()
    assert isinstance(llm, ChatGoogleGenerativeAI) and llm.client is not None
    monkeypatch.setattr(llm.client.models, "generate_content", fake.generate_content)
    return GraphEngine(llm=llm, retriever=None, tools=get_tools())


def test_gemini_tool_call_runs_the_tool_and_answers(provider: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGeminiModels()
    result = _gemini_engine(provider, monkeypatch, fake).invoke("What is 6 times 7?", [], "s1")
    assert "42" in result.answer
    assert len(fake.requests) == 2
    second = fake.requests[1]["contents"]
    call_part = next(p for c in second for p in c.parts if p.function_call)
    assert call_part.thought_signature == b"sig", "Gemini 3 needs the signature sent back with the call"
    assert second[-1].parts[-1].function_response.response == {"output": 42}  # LangChain JSON-parses "42"


def test_gemini_tool_loop_is_bounded(provider: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGeminiModels(loop_forever=True)
    result = _gemini_engine(provider, monkeypatch, fake).invoke("keep calculating", [], "s1")
    assert result.answer.startswith("The answer is")
    assert len(fake.requests) == MAX_TOOL_ROUNDS + 1
    assert fake.requests[-1]["config"].tool_config.function_calling_config.mode == "NONE"
