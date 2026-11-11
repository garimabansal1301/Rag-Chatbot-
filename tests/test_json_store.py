from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import Message
from app.storage.json_store import DEFAULT_TITLE, SessionNotFoundError, SessionStore


def test_create_writes_one_json_file(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create("Hello")
    path = tmp_path / f"{session.id}.json"
    assert path.exists()
    assert json.loads(path.read_text())["title"] == "Hello"


def test_rename_delete_and_list(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    a, b = store.create("A"), store.create("B")
    store.rename(a.id, "Renamed")
    assert {s.title for s in store.list()} == {"Renamed", "B"}
    store.delete(b.id)
    assert [s.id for s in store.list()] == [a.id]
    with pytest.raises(SessionNotFoundError):
        store.get(b.id)


def test_append_messages_sets_title_from_first_question(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create()
    assert session.title == DEFAULT_TITLE
    store.append_messages(session.id, Message(role="user", content="How many leave days?"))
    loaded = store.get(session.id)
    assert loaded.title == "How many leave days?"
    assert loaded.messages[0].content == "How many leave days?"


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "abc", "Z" * 32])
def test_rejects_unsafe_ids(tmp_path: Path, bad_id: str) -> None:
    with pytest.raises(SessionNotFoundError):
        SessionStore(tmp_path).get(bad_id)
