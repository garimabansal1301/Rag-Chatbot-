"""current_datetime tool: the server's local date and time."""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def current_datetime() -> str:
    """Return the current local date and time in ISO 8601 format with UTC offset, e.g. 2026-10-28T15:04:05+05:30.

    Use it whenever the answer depends on today's date, the day of the week or the current time.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")
