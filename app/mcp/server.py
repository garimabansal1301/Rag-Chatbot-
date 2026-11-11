"""The local MCP server: exposes calculator and current_datetime over stdio, and nothing else.

  uv run python -m app.mcp.server

It speaks MCP on stdin/stdout (so it must never print), opens no network port, and reuses the app's own tool
logic (app/tools/), so both paths compute exactly the same results. It reads no settings, no `.env` and no
files: the app starts it with a minimal environment that contains no API keys (app/mcp/client.py).
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import StructuredTool
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from app.tools.calculator import MAX_EXPRESSION_LENGTH, calculate
from app.tools.calculator import calculator as _calculator_tool
from app.tools.current_datetime import current_datetime as _current_datetime_tool

SERVER_NAME = "supportpilot-tools"
TOOL_NAMES = ("calculator", "current_datetime")  # the only tools this server exposes
_PURE = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

server = FastMCP(
    SERVER_NAME,
    instructions="Safe arithmetic and the current date and time. No file, network or environment access.",
    log_level="WARNING",
)


@server.tool(name="calculator", description=_calculator_tool.description, annotations=_PURE)
def calculator(
    expression: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_EXPRESSION_LENGTH,
            description='Arithmetic only: numbers, + - * / // % ** and parentheses, e.g. "(24 - 5) * 1.5".',
        ),
    ],
) -> str:
    """Parsed with `ast` (never eval/exec); invalid input returns a message starting with "Error"."""
    return calculate(expression)


@server.tool(
    name="current_datetime",
    description=_current_datetime_tool.description,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False),
)
def current_datetime() -> str:
    # The plain function, not .invoke(): no LangChain run, so nothing is ever traced from this process.
    assert isinstance(_current_datetime_tool, StructuredTool) and _current_datetime_tool.func is not None
    result: str = _current_datetime_tool.func()
    return result


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
