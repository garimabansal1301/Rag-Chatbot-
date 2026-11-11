"""MCPToolset: connects the chat engine to the local MCP server (app/mcp/server.py) over stdio.

Why not simply `MultiServerMCPClient.get_tools()`: its tools open a NEW session, i.e. start a new server
process, for every call, and they are async-only while the chat graph runs synchronously. Instead:

  - one background thread runs an asyncio loop that owns ONE session for the app's lifetime
    (`client.session(...)` + `load_mcp_tools(session)`, the adapter's supported way to reuse a session);
  - the session is opened and closed inside the same task, as the MCP SDK's anyio streams require;
  - each discovered tool is wrapped in a synchronous StructuredTool that hands the call to that loop and waits
    (with a timeout), so ToolNode calls it like any other tool and the FastAPI event loop is never blocked.

Only the server command below can be started (no command comes from settings or requests), with the MCP SDK's
minimal environment (PATH, SYSTEMROOT, ...; no API keys), and only the tools in ALLOWED_TOOLS are used.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import get_default_environment

from app.config import PROJECT_ROOT
from app.mcp.server import SERVER_NAME, TOOL_NAMES

logger = logging.getLogger(__name__)

ALLOWED_TOOLS = frozenset(TOOL_NAMES)
STARTUP_TIMEOUT_SECONDS = 30.0  # start the process, initialize the session and list its tools
CALL_TIMEOUT_SECONDS = 30.0
CLOSE_TIMEOUT_SECONDS = 10.0
MCP_TAG = "mcp"


class MCPUnavailableError(RuntimeError):
    """The MCP server could not be started or its tools could not be discovered."""


def server_connection() -> StdioConnection:
    """The only server the app starts: this interpreter running app.mcp.server, from the project folder."""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "app.mcp.server"],
        "env": get_default_environment(),  # an allow-list without API keys or app settings
        "cwd": str(PROJECT_ROOT),
    }


def _text(result: Any) -> str:
    """Tool output as text: the adapter returns a string, or a list of content blocks."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in result]
        return "\n".join(p for p in parts if p)
    return str(result)


class MCPToolset:
    """One MCP stdio session and its tools. `start()` once (blocking), use `.tools`, `close()` once."""

    def __init__(
        self,
        connection: StdioConnection | None = None,
        *,
        startup_timeout: float = STARTUP_TIMEOUT_SECONDS,
        call_timeout: float = CALL_TIMEOUT_SECONDS,
        client_factory: Callable[[dict[str, Any]], MultiServerMCPClient] = MultiServerMCPClient,
    ) -> None:
        self.connection = connection or server_connection()
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self.client_factory = client_factory
        self.tools: list[BaseTool] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._serving: concurrent.futures.Future[None] | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._serving is not None and not self._serving.done()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> list[BaseTool]:
        """Start the server process, open the session and discover the tools. Raises MCPUnavailableError (after
        cleaning up the process and the thread) if that fails or takes longer than `startup_timeout`."""
        with self._lock:
            if self._loop is not None:
                raise RuntimeError("MCPToolset.start() was already called")
            loop = self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=loop.run_forever, name="mcp-client", daemon=True)
            self._thread.start()
            ready: concurrent.futures.Future[list[BaseTool]] = concurrent.futures.Future()
            self._serving = asyncio.run_coroutine_threadsafe(self._serve(ready), loop)
        try:
            remote = ready.result(timeout=self.startup_timeout + CLOSE_TIMEOUT_SECONDS)
        except Exception as exc:
            self.close()
            reason = "timed out" if isinstance(exc, TimeoutError) else f"{type(exc).__name__}: {exc}"
            raise MCPUnavailableError(f"MCP server unavailable ({reason})"[:300]) from None
        unexpected = sorted(t.name for t in remote if t.name not in ALLOWED_TOOLS)
        if unexpected:
            logger.warning("Ignoring unexpected MCP tools: %s", ", ".join(unexpected))
        self.tools = [self._wrap(t) for t in remote if t.name in ALLOWED_TOOLS]
        logger.info("MCP tools available: %s", ", ".join(t.name for t in self.tools) or "none")
        return self.tools

    async def _serve(self, ready: concurrent.futures.Future[list[BaseTool]]) -> None:
        """Open the session, report the tools, wait for close(), then close the session (one task throughout).
        Leaving the `async with` ends the session and terminates the server process."""
        self._stop = asyncio.Event()
        client = self.client_factory({SERVER_NAME: self.connection})
        try:
            async with asyncio.timeout(self.startup_timeout) as startup:
                async with client.session(SERVER_NAME) as session:
                    tools = await load_mcp_tools(session, server_name=SERVER_NAME)
                    startup.reschedule(None)  # the timeout covers start-up only
                    ready.set_result(tools)
                    await self._stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception) else MCPUnavailableError(type(exc).__name__))
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning("The MCP session ended with an error (%s)", type(exc).__name__)
            if not isinstance(exc, Exception):
                raise

    def close(self) -> None:
        """End the session (terminating the server process) and stop the thread. Safe to call more than once."""
        with self._lock:
            loop, thread, serving = self._loop, self._thread, self._serving
            self._serving = None
            self.tools = []
        if loop is None or thread is None:
            return
        if serving is not None and not serving.done():
            stop = self._stop
            if stop is not None:
                loop.call_soon_threadsafe(stop.set)
            try:
                serving.result(timeout=CLOSE_TIMEOUT_SECONDS)
            except Exception:  # an error the session already logged, or a timeout: cancel what is left
                serving.cancel()
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=CLOSE_TIMEOUT_SECONDS)
        if not thread.is_alive() and not loop.is_closed():
            loop.close()

    # ------------------------------------------------------------------ tools
    def _wrap(self, remote: BaseTool) -> BaseTool:
        """A synchronous tool with the remote tool's name, description and JSON schema, tagged "mcp"."""

        def run(callbacks: Callbacks = None, **arguments: Any) -> str:
            return self._call(remote, arguments, callbacks)

        return StructuredTool(
            name=remote.name,
            description=remote.description,
            args_schema=remote.args_schema if remote.args_schema is not None else {"type": "object", "properties": {}},
            func=run,
            tags=[MCP_TAG],
            metadata={**(remote.metadata or {}), "mcp_server": SERVER_NAME},
        )

    def _call(self, remote: BaseTool, arguments: dict[str, Any], callbacks: Callbacks) -> str:
        loop = self._loop
        if loop is None or not self.running:
            raise ToolException("The MCP tool server is not running.")
        # `callbacks` (from the calling tool run) nests the remote call under it in LangSmith traces.
        future = asyncio.run_coroutine_threadsafe(remote.ainvoke(arguments, config={"callbacks": callbacks}), loop)
        try:
            return _text(future.result(timeout=self.call_timeout))
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise ToolException(f"The MCP tool {remote.name} did not answer within {self.call_timeout:g} s.") from None
