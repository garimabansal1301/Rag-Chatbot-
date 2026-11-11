from __future__ import annotations

from fastapi.testclient import TestClient

from app.chat.engine import PLACEHOLDER_ANSWER


def test_session_crud(client: TestClient) -> None:
    created = client.post("/api/sessions", json={"title": "First"})
    assert created.status_code == 201
    sid = created.json()["id"]

    assert [s["id"] for s in client.get("/api/sessions").json()] == [sid]
    assert client.patch(f"/api/sessions/{sid}", json={"title": "Renamed"}).json()["title"] == "Renamed"
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_send_message_uses_placeholder_until_engine_exists(client: TestClient) -> None:
    sid = client.post("/api/sessions").json()["id"]
    turn = client.post(f"/api/sessions/{sid}/messages", json={"content": "hi"}).json()
    assert turn["user_message"]["content"] == "hi"
    assert turn["assistant_message"]["role"] == "assistant"
    messages = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_placeholder_text_is_exported() -> None:
    assert "app/chat/engine.py" in PLACEHOLDER_ANSWER


def test_unknown_session_is_404(client: TestClient) -> None:
    missing = "0" * 32
    assert client.get(f"/api/sessions/{missing}").status_code == 404
    assert client.post(f"/api/sessions/{missing}/messages", json={"content": "x"}).status_code == 404
