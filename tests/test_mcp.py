"""MCP: the local FastMCP server, tool discovery over stdio, the graph calling MCP tools, and the app lifecycle.

Local only: the real server runs as a stdio subprocess (no network port); the LLM is FakeChatModel or a fake
Gemini client with a dummy key. No Gemini, LangSmith or other network calls.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langgraph.prebuilt import ToolNode
from mcp.client.stdio import get_default_environment

from app import deps
from app.chat.engine import GraphEngine, PlaceholderEngine, get_engine
from app.chat.llm import get_llm
from app.chat.review import ReviewPolicy
from app.config import PROJECT_ROOT, Settings, get_settings, load_settings
from app.main import create_app
from app.mcp import server as mcp_server
from app.mcp.client import MCP_TAG, MCPToolset, MCPUnavailableError, server_connection
from app.tools import get_tools
from app.tools.calculator import calculator as local_calculator
from tests.fakes import TOOL_TRIGGER, FakeChatModel

SERVER_FILE = PROJECT_ROOT / "app" / "mcp" / "server.py"


def connection(*args: str) -> StdioConnection:
    """A stdio connection running this interpreter with `args`, like the app's own (same minimal env)."""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": list(args),
        "env": get_default_environment(),
        "cwd": str(PROJECT_ROOT),
    }


def script(body: str) -> StdioConnection:
    return connection("-c", body)


# A real server, started via runpy so it can first write its process id to the file named by argv[1].
PID_SERVER = (
    "import os, runpy, sys; open(sys.argv[1], 'w').write(str(os.getpid())); "
    "runpy.run_module('app.mcp.server', run_name='__main__')"
)
FAKE_SERVER = """
import sys, time
from mcp.server.fastmcp import FastMCP
server = FastMCP("fake")
@server.tool(name="calculator", description="slow calculator")
def calculator(expression: str) -> str:
    time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0)
    return "slow result"
@server.tool(name="read_file", description="must never be exposed by the app")
def read_file(path: str) -> str:
    return "secret"
server.run(transport="stdio")
"""


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_until_dead(pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.2)
    return False


def thread_stopped(toolset: MCPToolset) -> bool:
    """The toolset's own event-loop thread has ended (other toolsets, e.g. the module fixture's, may still run)."""
    thread: threading.Thread | None = toolset._thread
    return thread is None or not thread.is_alive()


class CountingClient(MultiServerMCPClient):
    """Counts how many sessions (= server processes) are opened."""

    sessions = 0

    def session(self, *args: Any, **kwargs: Any) -> Any:
        CountingClient.sessions += 1
        return super().session(*args, **kwargs)


@pytest.fixture(scope="module")
def toolset() -> Iterator[MCPToolset]:
    """One real stdio server for the discovery and graph tests (it takes a few seconds to start)."""
    CountingClient.sessions = 0
    started = MCPToolset(client_factory=CountingClient)
    started.start()
    yield started
    started.close()


def by_name(toolset: MCPToolset) -> dict[str, Any]:
    return {t.name: t for t in toolset.tools}


# ---------------------------------------------------------------- the server
def test_server_registers_exactly_the_two_tools_with_schemas() -> None:
    tools = {t.name: t for t in asyncio.run(mcp_server.server.list_tools())}
    assert sorted(tools) == ["calculator", "current_datetime"] == sorted(mcp_server.TOOL_NAMES)
    calc = tools["calculator"].inputSchema
    assert calc["required"] == ["expression"] and calc["properties"]["expression"]["type"] == "string"
    assert calc["properties"]["expression"]["maxLength"] == 200 and calc["properties"]["expression"]["minLength"] == 1
    assert tools["current_datetime"].inputSchema["properties"] == {}
    assert tools["calculator"].description == local_calculator.description, "the LLM sees the same description"
    for tool in tools.values():
        assert tool.annotations is not None and tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False and tool.annotations.openWorldHint is False


def test_server_tools_reuse_the_safe_app_logic() -> None:
    def call(name: str, args: dict[str, Any]) -> str:
        result: Any = asyncio.run(mcp_server.server.call_tool(name, args))
        content, _ = result  # (content blocks, structured output)
        return str(content[0].text)

    assert call("calculator", {"expression": "(24 - 5) * 1.5"}) == "28.5"
    assert call("calculator", {"expression": "__import__('os').system('whoami')"}).startswith("Error")
    assert call("calculator", {"expression": "9 ** 9 ** 9"}).startswith("Error")
    assert call("calculator", {"expression": "1 / 0"}) == "Error: division by zero"
    assert datetime.fromisoformat(call("current_datetime", {})).tzinfo is not None
    with pytest.raises(Exception, match="at most 200 characters"):
        asyncio.run(mcp_server.server.call_tool("calculator", {"expression": "1+" * 150}))
    with pytest.raises(Exception, match="Unknown tool"):
        asyncio.run(mcp_server.server.call_tool("read_file", {"path": ".env"}))


def test_server_module_has_no_eval_exec_or_system_access() -> None:
    tree = ast.parse(SERVER_FILE.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"eval", "exec", "open", "compile", "__import__"}
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {str(n.module).split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"__future__", "typing", "mcp", "pydantic", "langchain_core", "app"}, "no os, subprocess, socket ..."
    assert 'transport="stdio"' in SERVER_FILE.read_text(encoding="utf-8")


def test_the_app_starts_only_its_own_server_with_a_keyless_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-real")
    monkeypatch.setenv("LANGSMITH_API_KEY", "dummy-not-real")
    conn = server_connection()
    assert conn["transport"] == "stdio" and conn["command"] == sys.executable
    assert conn["args"] == ["-m", "app.mcp.server"] and Path(str(conn["cwd"])) == PROJECT_ROOT
    env = conn.get("env") or {}
    assert "dummy-not-real" not in env.values()
    assert not [k for k in env if any(word in k.upper() for word in ("KEY", "TOKEN", "SECRET", "LANGSMITH"))]


# ---------------------------------------------------------------- discovery over stdio
def test_tools_are_discovered_over_stdio(toolset: MCPToolset) -> None:
    tools = by_name(toolset)
    assert sorted(tools) == ["calculator", "current_datetime"] and toolset.running
    calc = tools["calculator"]
    assert calc.tags == [MCP_TAG] and calc.metadata["mcp_server"] == mcp_server.SERVER_NAME
    assert calc.metadata["readOnlyHint"] is True
    assert set(calc.args) == set(local_calculator.args) == {"expression"}
    assert calc.description == local_calculator.description


def test_tool_calls_reuse_one_session(toolset: MCPToolset) -> None:
    tools = by_name(toolset)
    assert tools["calculator"].invoke({"expression": "6 * 7"}) == "42"
    assert tools["calculator"].invoke({"expression": "2 ** 10"}) == "1024"
    assert datetime.fromisoformat(tools["current_datetime"].invoke({})).tzinfo is not None
    assert CountingClient.sessions == 1, "one server process for every call, not one per call"


def test_invalid_input_is_rejected_by_the_server(toolset: MCPToolset) -> None:
    calc = by_name(toolset)["calculator"]
    assert calc.invoke({"expression": "open('.env').read()"}).startswith("Error")
    assert "at most 200 characters" in calc.invoke({"expression": "1+" * 150})


# ---------------------------------------------------------------- through the LangGraph workflow
def test_fake_model_calls_an_mcp_tool_through_the_graph(toolset: MCPToolset) -> None:
    llm = FakeChatModel(reply="DONE", tool_name="calculator", tool_args={"expression": "6 * 7"})
    engine = get_engine(llm=llm, retriever=None, tools=toolset.tools, review_policy=ReviewPolicy())
    assert isinstance(engine, GraphEngine)
    assert "tools" in engine.graph.nodes and isinstance(engine.graph.nodes["tools"].bound, ToolNode)
    result = engine.invoke(f"{TOOL_TRIGGER} what is 6 times 7?", [], "session-mcp")
    assert result.answer == "DONE | tool said: 42" and result.review is None
    assert llm.bound_tools == ["calculator", "current_datetime", "request_human_review"], "HITL tool still bound"
    tool_messages = [m for call in llm.calls for m in call if isinstance(m, ToolMessage)]
    assert tool_messages[-1].name == "calculator" and tool_messages[-1].content == "42"


def test_mcp_tool_failures_become_tool_results(toolset: MCPToolset) -> None:
    llm = FakeChatModel(reply="DONE", tool_name="calculator", tool_args={"expression": "x" * 300})
    engine = get_engine(llm=llm, retriever=None, tools=toolset.tools)
    result = engine.invoke(f"{TOOL_TRIGGER} compute", [], "session-mcp")
    assert result.answer.startswith("DONE | tool said:") and "at most 200 characters" in result.answer


def test_gemini_binds_the_mcp_tool_schemas(toolset: MCPToolset, monkeypatch: pytest.MonkeyPatch) -> None:
    """The discovered JSON schemas convert into Gemini function declarations (request built offline, not sent)."""
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key-not-real")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    get_settings.cache_clear()
    try:
        llm = get_llm()
        assert isinstance(llm, ChatGoogleGenerativeAI)
        engine = GraphEngine(llm=llm, retriever=None, tools=toolset.tools)
        messages = [SystemMessage(content="system"), HumanMessage(content="2+2?")]
        config = llm._prepare_request(messages, **engine.llm_with_tools.kwargs)["config"]  # type: ignore[attr-defined]
        declarations = {d.name: d for t in config.tools for d in t.function_declarations}
        assert sorted(declarations) == ["calculator", "current_datetime"]
        schema = declarations["calculator"].parameters_json_schema or declarations["calculator"].parameters
        assert "expression" in str(schema)
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------- tool registration in the app
def captured_tools(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(deps, "get_engine", lambda **kw: captured.append(kw))
    deps.build_components(**kwargs)
    return captured[0]["tools"]


def test_mcp_tools_replace_the_local_ones_without_duplicates(
    toolset: MCPToolset, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = captured_tools(monkeypatch, mcp_tools=toolset.tools)
    assert [t.name for t in tools] == ["calculator", "current_datetime"]
    assert all(MCP_TAG in (t.tags or []) for t in tools), "the MCP versions, not the local ones"


def test_mcp_disabled_keeps_the_original_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    assert captured_tools(monkeypatch) is None, "unchanged: the engine's default tools"
    assert captured_tools(monkeypatch, mcp_tools=[]) is None
    assert [t.name for t in get_tools()] == ["calculator", "current_datetime"]
    assert get_tools()[0] is local_calculator


def test_mcp_enabled_flag_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCP_ENABLED", raising=False)
    assert load_settings().mcp_enabled is False
    monkeypatch.setenv("MCP_ENABLED", "true")
    assert load_settings().mcp_enabled is True
    monkeypatch.setenv("MCP_ENABLED", "maybe")
    with pytest.raises(ValueError, match="MCP_ENABLED"):
        load_settings()


# ---------------------------------------------------------------- startup failures and cleanup
def test_a_server_that_exits_at_startup_is_reported_and_cleaned_up() -> None:
    toolset = MCPToolset(script("import sys; sys.exit(3)"), startup_timeout=15)
    with pytest.raises(MCPUnavailableError, match="MCP server unavailable"):
        toolset.start()
    assert not toolset.running and toolset.tools == [] and thread_stopped(toolset)


def test_a_server_that_never_answers_times_out_and_is_terminated(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    hang = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(120)"
    toolset = MCPToolset(connection("-c", hang, str(pid_file)), startup_timeout=2)
    started = time.monotonic()
    with pytest.raises(MCPUnavailableError, match="timed out|TimeoutError"):
        toolset.start()
    assert time.monotonic() - started < 15
    assert wait_until_dead(int(pid_file.read_text())), "the hanging server process was terminated"
    assert thread_stopped(toolset)


def test_close_ends_the_server_process_and_further_calls_fail(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    toolset = MCPToolset(connection("-c", PID_SERVER, str(pid_file)))
    tools = {t.name: t for t in toolset.start()}
    calc = tools["calculator"]
    pid = int(pid_file.read_text())
    assert pid_alive(pid) and calc.invoke({"expression": "1 + 1"}) == "2"
    toolset.close()
    toolset.close()  # idempotent
    assert wait_until_dead(pid) and not toolset.running and thread_stopped(toolset)
    with pytest.raises(ToolException, match="not running"):
        calc.invoke({"expression": "1 + 1"})


def test_slow_tools_time_out_and_unexpected_tools_are_ignored() -> None:
    toolset = MCPToolset(connection("-c", FAKE_SERVER, "5"), call_timeout=0.5)  # each call sleeps 5 s
    try:
        tools = toolset.start()
        assert [t.name for t in tools] == ["calculator"], "read_file is not in ALLOWED_TOOLS"
        with pytest.raises(ToolException, match="did not answer within 0.5 s"):
            tools[0].invoke({"expression": "1"})
    finally:
        toolset.close()
    assert thread_stopped(toolset)


# ---------------------------------------------------------------- app lifecycle
@pytest.fixture()
def mcp_settings(settings: Settings) -> Settings:
    return replace(settings, mcp_enabled=True)


def test_app_with_mcp_disabled_starts_no_server(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def never() -> MCPToolset:
        raise AssertionError("MCP must not start when MCP_ENABLED is false")

    monkeypatch.setattr(deps, "_open_mcp_toolset", never)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").json()["mcp"] == "disabled"
        assert app.state.mcp_toolset is None and app.state.mcp_error is None


def test_app_starts_the_server_once_and_stops_it_at_shutdown(mcp_settings: Settings) -> None:
    app = create_app(mcp_settings)
    with TestClient(app) as client:
        toolset = app.state.mcp_toolset
        assert isinstance(toolset, MCPToolset) and toolset.running
        assert client.get("/health").json()["mcp"] == "connected"
        assert client.get("/health").json()["mcp"] == "connected" and app.state.mcp_toolset is toolset
    assert not toolset.running and app.state.mcp_toolset is None and thread_stopped(toolset)


def test_app_survives_an_mcp_startup_failure(mcp_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def broken() -> MCPToolset:
        attempts.append(1)
        raise MCPUnavailableError("MCP server unavailable (FileNotFoundError: no such file)")

    built: list[dict[str, Any]] = []

    def build(**kwargs: Any) -> deps.Components:
        built.append(kwargs)
        return deps.Components(engine=PlaceholderEngine())

    monkeypatch.setattr(deps, "_open_mcp_toolset", broken)
    monkeypatch.setattr(deps, "build_components", build)
    app = create_app(mcp_settings)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "tracing": False, "mcp": "unavailable"}
        components = deps._build_app_components(app.state)
        assert built[-1]["mcp_tools"] is None, "the local tools are used instead"
        assert components.missing["mcp"].startswith("MCP server unavailable")
    assert attempts == [1], "tried once at start-up, not on every request"
