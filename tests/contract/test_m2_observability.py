"""Milestone 2 — LangSmith tracing switch."""

from __future__ import annotations

import pytest

from app.observability import configure_tracing

pytestmark = pytest.mark.contract

ENV = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


def test_off_by_default() -> None:
    assert configure_tracing() is False


def test_on_with_flag_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
    assert configure_tracing() is True


def test_flag_without_key_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert configure_tracing() is False
