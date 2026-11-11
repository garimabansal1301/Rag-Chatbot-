from __future__ import annotations

from fastapi.testclient import TestClient


def test_home_renders_without_sessions(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "New chat" in response.text


def test_ui_flow_create_rename_chat_delete(client: TestClient) -> None:
    created = client.post("/ui/sessions", follow_redirects=False)
    assert created.status_code == 303
    url = created.headers["location"]
    sid = url.rsplit("/", 1)[-1]

    assert client.get(url).status_code == 200
    client.post(f"/ui/sessions/{sid}/rename", data={"title": "My chat"})
    assert "My chat" in client.get(url).text

    partial = client.post(f"/ui/sessions/{sid}/messages", data={"content": "hello there"})
    assert partial.status_code == 200
    assert "hello there" in partial.text

    client.post(f"/ui/sessions/{sid}/delete")
    assert client.get(url).status_code == 404
