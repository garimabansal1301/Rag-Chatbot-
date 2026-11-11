"""Phase 1 of web/image search: backward-compatible models and settings. Nothing here calls a provider."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import IMAGE_SEARCH_PROVIDERS, Settings, load_settings
from app.models import EngineResult, ImageResult, Message, Session, WebGrounding, WebSource
from app.storage.json_store import SessionStore

SESSION_ID = "0123456789abcdef0123456789abcdef"
OLD_SESSION: dict[str, Any] = {  # the shape saved before web search existed: no "web" / "images" keys
    "id": SESSION_ID,
    "title": "Leave questions",
    "created_at": "2026-10-28T10:00:00+00:00",
    "updated_at": "2026-10-28T10:01:00+00:00",
    "messages": [
        {"role": "user", "content": "How many leave days?", "timestamp": "2026-10-28T10:00:30+00:00", "sources": []},
        {
            "role": "assistant",
            "content": "25 days [1].",
            "timestamp": "2026-10-28T10:01:00+00:00",
            "sources": [{"source": "leave_policy.md", "chunk_id": "abc:0", "score": 0.81, "snippet": "25 days"}],
        },
    ],
}

IMAGE: dict[str, Any] = {
    "title": "Router front panel",
    "thumbnail_url": "https://api.openverse.org/v1/images/1b2c/thumb/",
    "source_page_url": "https://www.flickr.com/photos/someone/123",
    "creator": "Someone",
    "creator_url": "https://www.flickr.com/photos/someone",
    "license_name": "CC BY 2.0",
    "license_url": "https://creativecommons.org/licenses/by/2.0/",
    "attribution": '"Router front panel" by Someone is licensed under CC BY 2.0.',
}
WEB_SOURCE = {
    "title": "Example article",
    "url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123",
    "domain": "example.com",
}


# ---------------------------------------------------------------- old data keeps working
def _write_old_session(folder: Path) -> SessionStore:
    store = SessionStore(folder)
    (folder / f"{SESSION_ID}.json").write_text(json.dumps(OLD_SESSION), encoding="utf-8")
    return store


def test_old_session_files_load_with_empty_new_fields(tmp_path: Path) -> None:
    store = _write_old_session(tmp_path)
    session = store.get(SESSION_ID)
    assert [m.content for m in session.messages] == ["How many leave days?", "25 days [1]."]
    assert all(m.web is None and m.images == [] for m in session.messages)
    assert session.messages[1].sources[0].source == "leave_policy.md"
    assert [s.id for s in store.list()] == [SESSION_ID]


def test_old_session_still_renders_and_accepts_new_messages(client: TestClient, settings: Settings) -> None:
    _write_old_session(settings.sessions_dir)
    page = client.get(f"/sessions/{SESSION_ID}")
    assert page.status_code == 200 and "25 days [1]." in page.text and "leave_policy.md" in page.text
    turn = client.post(f"/api/sessions/{SESSION_ID}/messages", json={"content": "And sick leave?"}).json()
    assert turn["assistant_message"]["web"] is None and turn["assistant_message"]["images"] == []
    assert len(client.get(f"/api/sessions/{SESSION_ID}").json()["messages"]) == 4


def test_new_fields_round_trip_through_the_session_file(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create()
    web = WebGrounding(
        answer="Grounded text.", sources=[WebSource(**WEB_SOURCE)], search_suggestions_html="<style></style>"
    )
    reply = Message(role="assistant", content="From the documents.", web=web, images=[ImageResult(**IMAGE)])
    store.append_messages(session.id, Message(role="user", content="q"), reply)
    assert store.get(session.id).messages[1] == reply


def test_engine_result_defaults_are_empty() -> None:
    result = EngineResult(answer="hi")
    assert result.sources == [] and result.web is None and result.images == []


# ---------------------------------------------------------------- settings
def test_tests_run_with_search_off_whatever_dotenv_says(settings: Settings) -> None:
    # conftest.search_features_off overrides values that `.env` loaded into the process
    assert os.environ["IMAGE_SEARCH_PROVIDER"] == "none" and os.environ["WEB_SEARCH_ENABLED"] == "false"
    assert settings.image_search_provider == "none" and settings.web_search_enabled is False


def test_a_test_can_still_switch_image_search_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_SEARCH_PROVIDER", "openverse")
    assert load_settings().image_search_provider == "openverse"


def test_new_settings_default_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("IMAGE_SEARCH_PROVIDER", raising=False)
    settings = load_settings()
    assert settings.web_search_enabled is False and settings.image_search_provider == "none"
    assert Settings.web_search_enabled is False and Settings.image_search_provider == "none"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), (" TRUE ", True), ("1", True), ("yes", True), ("on", True)]
    + [("false", False), ("0", False), ("No", False), ("off", False), ("", False)],
)
def test_web_search_flag_values(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("WEB_SEARCH_ENABLED", raw)
    assert load_settings().web_search_enabled is expected


@pytest.mark.parametrize("raw", ["ture", "enabled", "2", "y e s"])
def test_invalid_web_search_flag_is_rejected(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("WEB_SEARCH_ENABLED", raw)
    with pytest.raises(ValueError, match="WEB_SEARCH_ENABLED must be true or false"):
        load_settings()


@pytest.mark.parametrize(("raw", "expected"), [("openverse", "openverse"), (" OpenVerse ", "openverse"), ("", "none")])
def test_image_provider_values(monkeypatch: pytest.MonkeyPatch, raw: str, expected: str) -> None:
    monkeypatch.setenv("IMAGE_SEARCH_PROVIDER", raw)
    assert load_settings().image_search_provider == expected


@pytest.mark.parametrize("raw", ["brave", "google", "openverse,brave"])
def test_invalid_image_provider_is_rejected(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("IMAGE_SEARCH_PROVIDER", raw)
    with pytest.raises(ValueError, match="IMAGE_SEARCH_PROVIDER must be one of none, openverse"):
        load_settings()


def test_invalid_provider_is_rejected_when_settings_are_copied(settings: Settings) -> None:
    assert IMAGE_SEARCH_PROVIDERS == ("none", "openverse")
    with pytest.raises(ValueError, match="IMAGE_SEARCH_PROVIDER"):
        replace(settings, image_search_provider="brave")


# ---------------------------------------------------------------- untrusted provider data
def test_valid_image_and_web_source() -> None:
    image = ImageResult(**IMAGE)
    assert image.provider == "openverse" and image.thumbnail_url.startswith("https://")
    assert WebSource(**WEB_SOURCE).domain == "example.com"
    assert ImageResult(**{**IMAGE, "source_page_url": "http://example.org/page"}).source_page_url.startswith("http:")


@pytest.mark.parametrize(
    "url",
    [
        "http://api.openverse.org/thumb.jpg",  # thumbnails must be https
        "javascript:alert(1)",
        "data:image/png;base64,AAAA",
        "//cdn.example.com/x.jpg",
        "https://user:pass@example.com/x.jpg",
        "https://127.0.0.1/x.jpg",
        "https://[::1]/x.jpg",
        "https://10.0.0.5/x.jpg",
        "https://localhost/x.jpg",
        "https://printer.local/x.jpg",
        "https://intranet/x.jpg",
        "https://example.com:8443/x.jpg",
        "https://example.com:notaport/x.jpg",
        "https://exa mple.com/x.jpg",
        "https://example.com/x.jpg\n",
        "https://example.com/" + "a" * 2048,
    ],
)
def test_unsafe_thumbnail_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        ImageResult(**{**IMAGE, "thumbnail_url": url})


@pytest.mark.parametrize("field", ["source_page_url", "creator_url", "license_url"])
@pytest.mark.parametrize("url", ["ftp://example.com/x", "javascript:alert(1)", "https://192.168.1.1/", "file:///etc"])
def test_unsafe_link_urls_are_rejected(field: str, url: str) -> None:
    with pytest.raises(ValidationError):
        ImageResult(**{**IMAGE, field: url})


@pytest.mark.parametrize(
    "change",
    [
        {"url": "http://example.com/page"},  # web source links must be https
        {"url": "https://localhost/"},
        {"domain": "localhost"},
        {"domain": "10.0.0.1"},
        {"domain": "not a domain"},
    ],
)
def test_invalid_web_sources_are_rejected(change: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        WebSource(**{**WEB_SOURCE, **change})


def test_provider_text_is_cleaned_and_capped() -> None:
    image = ImageResult(**{**IMAGE, "title": "  Line\none\t\x00two  ", "attribution": "x" * 5000})
    assert image.title == "Line one two"
    assert len(image.attribution) == 1000 and image.attribution.endswith("…")
    assert ImageResult(**image.model_dump()) == image, "cleaning is stable when a saved session is reloaded"


def test_web_grounding_limits() -> None:
    with pytest.raises(ValidationError):
        WebGrounding(answer="")
    with pytest.raises(ValidationError):
        WebGrounding(answer="ok", search_suggestions_html="x" * 50_001)
    with pytest.raises(ValidationError):
        WebGrounding(answer="ok", sources=[WebSource(**WEB_SOURCE)] * 21)
    assert WebGrounding(answer="ok").search_suggestions_html is None


def test_unknown_image_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ImageResult(**{**IMAGE, "provider": "brave"})


def test_session_with_new_fields_validates_from_json() -> None:
    data = {**OLD_SESSION, "messages": [{**OLD_SESSION["messages"][1], "images": [IMAGE], "web": None}]}
    session = Session.model_validate_json(json.dumps(data))
    assert session.messages[0].images[0].license_name == "CC BY 2.0"
