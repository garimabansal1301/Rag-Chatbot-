from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings


def test_upload_saves_markdown(client: TestClient, settings: Settings) -> None:
    response = client.post("/api/documents", files={"file": ("notes.md", b"# Notes\nhello", "text/markdown")})
    assert response.status_code == 201
    assert (settings.documents_dir / "notes.md").read_text() == "# Notes\nhello"
    assert client.get("/api/documents").json() == ["notes.md"]


def test_upload_rejects_other_types(client: TestClient) -> None:
    response = client.post("/api/documents", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert response.status_code == 400


def test_upload_strips_path_from_filename(client: TestClient, settings: Settings) -> None:
    client.post("/api/documents", files={"file": ("../../escape.md", b"x", "text/markdown")})
    assert (settings.documents_dir / "escape.md").exists()
