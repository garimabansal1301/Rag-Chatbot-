"""Phase 2: Openverse image search, the image_search tool and image cards. Mocked HTTP and fake LLMs only."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

import app.search.images as images_module
from app.chat.engine import IMAGE_RULE, MAX_TOOL_ROUNDS, PlaceholderEngine, _remove_markdown_images, get_engine
from app.config import Settings
from app.deps import Components, _optional_tools, get_components
from app.main import create_app
from app.models import EngineResult, ImageResult, Message
from app.search.images import (
    MAX_IMAGES_PER_TURN,
    PAGE_SIZE,
    USER_AGENT,
    ImageSearchError,
    OpenverseImageSearch,
    parse_results,
    to_image,
)
from app.storage.json_store import SessionStore
from app.tools import get_tools
from app.tools.image_search import NO_IMAGE_LINKS, TOOL_NAME, make_image_search_tool, safe_image_query
from tests.fakes import TOOL_TRIGGER, FakeChatModel, StaticRetriever, make_chunk

UUID = "4bc43a04-ef46-4544-a0c1-63c63f56e276"


def openverse_item(n: int = 1, **changes: Any) -> dict[str, Any]:
    """An image result in the documented Openverse response format."""
    item = {
        "id": UUID[:-1] + str(n % 10),
        "title": f"Router front panel {n}",
        "foreign_landing_url": f"https://www.flickr.com/photos/someone/{n}",
        "url": f"https://live.staticflickr.com/{n}.jpg",
        "creator": "Some One",
        "creator_url": "https://www.flickr.com/photos/someone",
        "license": "by",
        "license_version": "2.0",
        "license_url": "https://creativecommons.org/licenses/by/2.0/",
        "provider": "flickr",
        "attribution": f'"Router front panel {n}" by Some One is licensed under CC BY 2.0.',
        "mature": False,
        "thumbnail": f"https://api.openverse.org/v1/images/{UUID[:-1]}{n % 10}/thumb/",
        "unstable__sensitivity": [],
    }
    item.update(changes)
    return item


def img(n: int) -> ImageResult:
    image = to_image(openverse_item(n))
    assert image is not None
    return image


# The fake LLM calls its tool when any user message contains TOOL_TRIGGER; putting the trigger in an
# earlier message keeps the current question free of the trigger's own words.
TRIGGER_HISTORY = [Message(role="user", content=TOOL_TRIGGER), Message(role="assistant", content="ok")]


def payload(*items: dict[str, Any]) -> dict[str, Any]:
    return {"result_count": len(items), "page_count": 1, "page_size": PAGE_SIZE, "page": 1, "results": list(items)}


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> OpenverseImageSearch:
    return OpenverseImageSearch(client=httpx.Client(transport=httpx.MockTransport(handler)))


# ---------------------------------------------------------------- provider: request and parsing
def test_search_sends_a_small_anonymous_safe_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload(openverse_item(1)))

    provider = OpenverseImageSearch(
        client=httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": USER_AGENT})
    )
    images = provider.search("  wifi   router  ")
    request = seen[0]
    assert request.url.host == "api.openverse.org" and request.url.path == "/v1/images/"
    assert dict(request.url.params) == {"q": "wifi router", "page_size": str(PAGE_SIZE), "mature": "false"}
    assert "authorization" not in request.headers and request.headers["user-agent"] == USER_AGENT
    assert images == [to_image(openverse_item(1))]


def test_default_client_has_timeout_user_agent_and_no_redirects() -> None:
    client = OpenverseImageSearch()._client
    assert client.timeout.connect == 3.0 and client.timeout.read == 12.0
    assert client.timeout.write == 6.0 and client.timeout.pool == 6.0
    assert client.headers["user-agent"] == USER_AGENT and client.follow_redirects is False


def test_empty_query_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert client_for(handler).search("   ") == []


def test_fields_are_mapped_with_attribution() -> None:
    image = to_image(openverse_item(1))
    assert image is not None
    assert image.title == "Router front panel 1" and image.creator == "Some One"
    assert image.thumbnail_url.startswith("https://api.openverse.org/v1/images/")
    assert image.source_page_url == "https://www.flickr.com/photos/someone/1"
    assert image.license_name == "CC BY 2.0" and image.license_url == "https://creativecommons.org/licenses/by/2.0/"
    assert image.attribution.endswith("is licensed under CC BY 2.0.") and image.provider == "openverse"


@pytest.mark.parametrize(
    ("license_code", "version", "name"),
    [("by-nc-sa", "3.0", "CC BY-NC-SA 3.0"), ("cc0", "1.0", "CC0 1.0"), ("pdm", "", "Public Domain Mark")],
)
def test_license_names(license_code: str, version: str, name: str) -> None:
    image = to_image(openverse_item(1, license=license_code, license_version=version))
    assert image is not None and image.license_name == name


@pytest.mark.parametrize(
    "changes",
    [
        {"mature": True},
        {"unstable__sensitivity": ["sensitive_text"]},
        {"thumbnail": "http://api.openverse.org/v1/images/x/thumb/"},  # not https
        {"thumbnail": "https://evil.example.com/v1/images/x/thumb/"},  # not Openverse's thumbnail host
        {"thumbnail": "https://api.openverse.org/admin/"},
        {"thumbnail": None},
        {"license": None},
        {"license": "<script>"},
        {"foreign_landing_url": "javascript:alert(1)", "id": "not-a-uuid"},
    ],
)
def test_unsafe_or_incomplete_results_are_dropped(changes: dict[str, Any]) -> None:
    assert to_image(openverse_item(1, **changes)) is None


def test_missing_optional_fields_are_tolerated() -> None:
    item = openverse_item(
        1,
        title=None,
        creator=42,
        creator_url="https://10.0.0.1/me",
        license_url="ftp://x",
        attribution=None,
        foreign_landing_url="https://localhost/x",
    )
    image = to_image(item)
    assert image is not None
    assert image.title == "Untitled image" and image.creator == "" and image.creator_url is None
    assert image.license_url is None and image.source_page_url == f"https://openverse.org/image/{item['id']}"
    assert image.attribution == '"Untitled image" is licensed under CC BY 2.0.'


def test_parse_keeps_at_most_three_valid_unique_results() -> None:
    items = [openverse_item(1, mature=True), openverse_item(2), openverse_item(2), *map(openverse_item, range(3, 9))]
    images = parse_results(payload(*items), limit=10)
    assert len(images) == MAX_IMAGES_PER_TURN == 3
    assert [i.title for i in images] == ["Router front panel 2", "Router front panel 3", "Router front panel 4"]


@pytest.mark.parametrize("body", [[], {"results": None}, {"results": "x"}, "text"])
def test_malformed_response_shapes_raise(body: Any) -> None:
    with pytest.raises(ImageSearchError, match="unexpected Openverse response format"):
        parse_results(body)


def test_non_dict_items_are_skipped() -> None:
    assert parse_results({"results": [None, "x", 3, openverse_item(1)]}) == [to_image(openverse_item(1))]


# ---------------------------------------------------------------- provider: failures
def _raise(exc: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (_raise(httpx.ConnectError("refused")), "could not reach Openverse (ConnectError)"),
        (lambda r: httpx.Response(429, json={"detail": "throttled"}), "rate limit reached"),
        (lambda r: httpx.Response(503, text="down"), "HTTP 503"),
        (lambda r: httpx.Response(301, headers={"Location": "https://elsewhere.example.com"}), "HTTP 301"),
        (lambda r: httpx.Response(200, text="<html>not json"), "invalid JSON"),
        (lambda r: httpx.Response(200, json={"detail": "x"}), "unexpected Openverse response format"),
        (lambda r: httpx.Response(200, content=b"[" + b"0," * 1_100_000 + b"0]"), "too large"),
    ],
)
def test_failures_raise_a_clean_error_without_the_query(
    handler: Callable[[httpx.Request], httpx.Response], message: str
) -> None:
    with pytest.raises(ImageSearchError, match=re.escape(message)) as info:
        client_for(handler).search("secret project falcon")
    assert "falcon" not in str(info.value) and "api.openverse.org" not in str(info.value)


@pytest.mark.parametrize("error", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout])
def test_timeouts_name_the_limit_that_expired(error: type[httpx.TimeoutException]) -> None:
    with pytest.raises(ImageSearchError) as info:
        client_for(_raise(error("timed out"))).search("secret project falcon")
    message = str(info.value)
    assert re.fullmatch(rf"Openverse did not answer in time \({error.__name__} after \d+\.\ds\)", message)
    assert "falcon" not in message and "openverse.org" not in message and "q=" not in message


def test_timeout_message_reports_elapsed_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([100.0, 112.04])
    monkeypatch.setattr(images_module.time, "monotonic", lambda: next(clock))
    with pytest.raises(ImageSearchError, match=re.escape("(ReadTimeout after 12.0s)")):
        client_for(_raise(httpx.ReadTimeout("timed out"))).search("router")


def test_each_search_is_one_request_without_retries() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(ImageSearchError):
        client_for(handler).search("router")
    assert calls == [1]


# ---------------------------------------------------------------- query safety
EXCERPTS = "[1] devices.md\nThe Falcon ZX-900 router is issued to engineers. Reset it with the pinhole button."


@pytest.mark.parametrize(
    ("requested", "user_texts", "expected"),
    [
        ("wifi router front panel", ["Show me what a wifi router front panel looks like"], "wifi router front panel"),
        ("the photo of a wifi router please", ["What does a wifi router look like?"], "wifi router"),
        ("Falcon ZX-900 router", ["What does our office router look like?"], "office router"),  # doc-only words
        ("falcon zx-900 router", ["We use the Falcon ZX-900 router. What does it look like?"], "falcon zx-900 router"),
        # "router" appears only in the documents here, so the query falls back to the user's own words
        ("falcon zx-900 router", ["We use the Falcon ZX-900. What does it look like?"], "use falcon zx-900"),
        ("pinhole button engineers", ["Show me a picture"], None),  # nothing safe left
        ("", ["What does a sea otter look like?"], "sea otter"),
        ("a b c d e f g h router", ["router"], "b c d e f g"),  # capped at 6 words
    ],
)
def test_safe_image_query(requested: str, user_texts: list[str], expected: str | None) -> None:
    assert safe_image_query(requested, user_texts, EXCERPTS) == expected


def test_query_never_contains_document_only_words() -> None:
    query = safe_image_query("Falcon pinhole engineers devices", ["How do I reset my router?"], EXCERPTS)
    assert query == "reset router"


# ---------------------------------------------------------------- tool in the LangGraph workflow
class FakeImages:
    def __init__(self, images: list[ImageResult] | None = None, error: Exception | None = None) -> None:
        self.images = images if images is not None else [img(1), img(2)]
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, limit: int = 3) -> list[ImageResult]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return list(self.images)


def image_llm(query: str = "wifi router") -> FakeChatModel:
    return FakeChatModel(reply="ANSWER", tool_name=TOOL_NAME, tool_args={"query": query})


def test_tool_schema_exposes_only_the_query() -> None:
    tool_schema = make_image_search_tool(FakeImages()).tool_call_schema
    assert isinstance(tool_schema, type) and issubclass(tool_schema, BaseModel)
    schema = tool_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}


def test_images_reach_the_engine_result_and_answer_still_comes_back() -> None:
    provider = FakeImages()
    tools = [*get_tools(), make_image_search_tool(provider)]
    result = get_engine(llm=image_llm(), retriever=None, tools=tools).invoke(
        f"{TOOL_TRIGGER} what does a wifi router look like?", [], "s1"
    )
    assert result.answer.startswith("ANSWER | tool said: 2 openly licensed image(s) will be shown")
    assert [i.title for i in result.images] == ["Router front panel 1", "Router front panel 2"]
    assert provider.queries == ["wifi router"]


def test_query_with_document_wording_is_replaced_by_the_users_words() -> None:
    provider = FakeImages()
    retriever = StaticRetriever([make_chunk(EXCERPTS, source="devices.md")])
    llm = image_llm(query="Falcon ZX-900 pinhole")
    engine = get_engine(llm=llm, retriever=retriever, tools=[make_image_search_tool(provider)])
    result = engine.invoke("show a home router", TRIGGER_HISTORY, "s1")
    assert provider.queries == ["home router"], "built from the user's own question"
    assert [s.source for s in result.sources] == ["devices.md"], "document citations unchanged"


def test_no_safe_query_skips_the_search() -> None:
    provider = FakeImages()
    retriever = StaticRetriever([make_chunk(EXCERPTS, source="devices.md")])
    engine = get_engine(llm=image_llm(query="Falcon"), retriever=retriever, tools=[make_image_search_tool(provider)])
    result = engine.invoke("Show me a picture", TRIGGER_HISTORY, "s1")
    assert provider.queries == [] and result.images == []
    assert "Image search skipped" in result.answer and result.answer.startswith("ANSWER")


@pytest.mark.parametrize("error", [ImageSearchError("Openverse returned HTTP 503"), RuntimeError("boom")])
def test_provider_failure_still_returns_the_answer(error: Exception) -> None:
    provider = FakeImages(error=error)
    result = get_engine(llm=image_llm(), retriever=None, tools=[make_image_search_tool(provider)]).invoke(
        f"{TOOL_TRIGGER} router", [], "s1"
    )
    assert result.images == [] and result.answer.startswith("ANSWER | tool said: Error")


def test_more_than_three_images_are_capped() -> None:
    provider = FakeImages(images=[img(n) for n in range(1, 6)])
    result = get_engine(llm=image_llm(), retriever=None, tools=[make_image_search_tool(provider)]).invoke(
        f"{TOOL_TRIGGER} router", [], "s1"
    )
    assert len(result.images) == 3


def test_questions_without_an_image_request_never_search() -> None:
    provider = FakeImages()
    engine = get_engine(llm=image_llm(), retriever=None, tools=[*get_tools(), make_image_search_tool(provider)])
    result = engine.invoke("How many leave days do I get?", [], "s1")
    assert provider.queries == [] and result.images == [] and result.answer == "ANSWER"


class ParallelImageCalls(FakeChatModel):
    """Asks for two image searches in one message, then answers."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if isinstance(messages[-1], ToolMessage):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="DONE"))])
        calls = [{"name": TOOL_NAME, "args": {"query": q}, "id": f"img_{q}"} for q in ("router", "modem")]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=calls))])


class RepeatingImageCalls(FakeChatModel):
    """Asks for an image search in every round (until the answer-only round)."""

    answer_only: bool = False

    def bind_tools(self, tools: Any, **kwargs: Any) -> RepeatingImageCalls:
        return self.model_copy(update={"answer_only": "tool_choice" in kwargs})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self.answer_only:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="FINAL"))])
        call = {"name": TOOL_NAME, "args": {"query": "router"}, "id": f"img_{len(self.calls)}"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


def test_parallel_image_calls_search_only_once() -> None:
    provider = FakeImages()
    result = get_engine(llm=ParallelImageCalls(), retriever=None, tools=[make_image_search_tool(provider)]).invoke(
        "router and modem pictures", [], "s1"
    )
    assert provider.queries == ["router"] and len(result.images) == 2 and result.answer == "DONE"


def test_repeated_image_calls_search_once_and_the_loop_stays_bounded() -> None:
    provider = FakeImages()
    llm = RepeatingImageCalls()
    result = get_engine(llm=llm, retriever=None, tools=[make_image_search_tool(provider)]).invoke("router", [], "s1")
    assert provider.queries == ["router"] and result.answer == "FINAL" and len(result.images) == 2
    refusals = [m for m in llm.calls[-1] if isinstance(m, ToolMessage) and "already used" in str(m.content)]
    assert len(refusals) == MAX_TOOL_ROUNDS - 1


def test_images_do_not_carry_over_to_the_next_turn() -> None:
    provider = FakeImages()
    engine = get_engine(llm=image_llm(), retriever=None, tools=[make_image_search_tool(provider)])
    assert engine.invoke(f"{TOOL_TRIGGER} router", [], "s1").images
    assert engine.invoke("thanks", [], "s1").images == []


# ---------------------------------------------------------------- settings wiring
def test_tool_is_added_only_when_openverse_is_configured(settings: Settings) -> None:
    from dataclasses import replace

    assert _optional_tools(settings) == []
    tools = _optional_tools(replace(settings, image_search_provider="openverse"))
    assert [t.name for t in tools] == [TOOL_NAME]
    assert [t.name for t in get_tools()] == ["calculator", "current_datetime"], "get_tools() unchanged"


# ---------------------------------------------------------------- chat UI
class ImageEngine:
    graph: Any = None

    def __init__(self, images: list[ImageResult]) -> None:
        self.images = images

    def invoke(self, message: str, history: list[Message], session_id: str) -> EngineResult:
        return EngineResult(answer="Here is what it looks like.", images=self.images)


@pytest.fixture()
def ui(settings: Settings) -> Iterator[tuple[TestClient, Settings]]:
    images = [img(1), img(2)]
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=ImageEngine(images))
    with TestClient(app) as client:
        yield client, settings


def test_chat_turn_renders_and_saves_image_cards(ui: tuple[TestClient, Settings]) -> None:
    client, settings = ui
    session_id = client.post("/api/sessions", json={}).json()["id"]
    html = client.post(f"/ui/sessions/{session_id}/messages", data={"content": "what does it look like?"}).text
    assert html.count('class="image-card"') == 2 and "Related images" in html
    assert 'loading="lazy"' in html and 'referrerpolicy="no-referrer"' in html
    assert 'alt="Router front panel 1 by Some One"' in html
    assert 'target="_blank" rel="noopener noreferrer"' in html and "View source" in html
    assert "not endorsed or certified by Openverse" in html and "not verified by Openverse" in html
    saved = SessionStore(settings.sessions_dir).get(session_id).messages[1]
    assert len(saved.images) == 2, "images are stored with the message and shown again on reload"
    assert client.get(f"/sessions/{session_id}").text.count('class="image-card"') == 2


def _page_with(settings: Settings, images: list[ImageResult]) -> str:
    store = SessionStore(settings.sessions_dir)
    session = store.create()
    store.append_messages(session.id, Message(role="assistant", content="answer", images=images))
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=PlaceholderEngine())
    with TestClient(app) as client:
        return client.get(f"/sessions/{session.id}").text


def test_untrusted_text_is_escaped_and_unsafe_links_are_not_rendered(settings: Settings) -> None:
    evil = to_image(
        openverse_item(1, title='<script>alert("x")</script>', creator="<b>me</b>", creator_url=None, license_url=None)
    )
    assert evil is not None
    html = _page_with(settings, [evil])
    assert "<script>alert" not in html and "&lt;script&gt;alert" in html and "&lt;b&gt;me&lt;/b&gt;" in html
    assert "by &lt;b&gt;me&lt;/b&gt;</span>" in html, "creator without a safe URL is plain text"
    assert ">CC BY 2.0</span>" in html, "license without a URL is plain text"


def test_at_most_three_cards_and_no_section_without_images(settings: Settings) -> None:
    many = [img(n) for n in range(1, 6)]
    assert _page_with(settings, many).count('class="image-card"') == 3
    assert "Related images" not in _page_with(settings, [])


def test_page_loads_the_broken_image_script(client: TestClient) -> None:
    session_id = client.post("/api/sessions", json={}).json()["id"]
    assert '<script src="http://testserver/static/app.js">' in client.get(f"/sessions/{session_id}").text
    script = client.get("/static/app.js")
    assert script.status_code == 200 and ".image-card" in script.text and 'style.display = "none"' in script.text
    assert ".remove()" not in script.text, "a failed thumbnail must not remove the card or the section"


def test_saved_message_json_keeps_document_sources_separate(settings: Settings) -> None:
    store = SessionStore(settings.sessions_dir)
    session = store.create()
    image = to_image(openverse_item(1))
    assert image is not None
    store.append_messages(session.id, Message(role="assistant", content="a", images=[image]))
    data = json.loads((settings.sessions_dir / f"{session.id}.json").read_text(encoding="utf-8"))
    assert data["messages"][0]["sources"] == [] and data["messages"][0]["images"][0]["license_name"] == "CC BY 2.0"


# ---------------------------------------------------------------- invented image links (not from image_search)
WIKI_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant_(1).jpg"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            f"Here is an elephant:\n\n![Elephant]({WIKI_IMAGE})\n\nElephants are large.",
            "Here is an elephant:\n\nElephants are large.",
        ),
        ('Look ![Elephant](https://x.example.org/e.jpg "An elephant") here.', "Look here."),
        ("[![Elephant](https://x.example.org/e.jpg)](https://x.example.org/page) Elephants.", "Elephants."),
        ("![a](https://x.example.org/1.jpg) ![b](https://x.example.org/2.jpg)\nText", "Text"),
        ("![](https://x.example.org/e.jpg)", ""),
    ],
)
def test_markdown_images_are_removed_from_answers(answer: str, expected: str) -> None:
    assert _remove_markdown_images(answer) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "See [Wikipedia](https://en.wikipedia.org/wiki/Elephant) for more. Leave is 25 days [1].",
        "Use 2 * 3 = 6.\n\n\n  Indented line  \n",
        "An exclamation! [1] and (parentheses).",
        "",
    ],
)
def test_answers_without_markdown_images_are_unchanged(answer: str) -> None:
    assert _remove_markdown_images(answer) == answer


def test_engine_removes_an_invented_image_and_keeps_the_text() -> None:
    llm = FakeChatModel(reply=f"An African elephant: ![Elephant]({WIKI_IMAGE}) It has large ears.")
    result = get_engine(llm=llm, retriever=None, tools=[]).invoke("Show me image of elephant", [], "s1")
    assert result.answer == "An African elephant: It has large ears."
    assert "upload.wikimedia.org" not in result.answer and result.images == []


def test_structured_images_survive_while_invented_ones_are_removed() -> None:
    class ImageThenMarkdown(FakeChatModel):
        """Calls image_search, then answers with an extra Markdown image of its own."""

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            self.calls.append(list(messages))
            if isinstance(messages[-1], ToolMessage):
                text = f"Elephants have trunks. ![Elephant]({WIKI_IMAGE})"
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
            call = {"name": TOOL_NAME, "args": {"query": "elephant"}, "id": "img_1"}
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])

    provider = FakeImages()
    result = get_engine(llm=ImageThenMarkdown(), retriever=None, tools=[make_image_search_tool(provider)]).invoke(
        "Show me image of elephant", [], "s1"
    )
    assert result.answer == "Elephants have trunks."
    assert [i.title for i in result.images] == ["Router front panel 1", "Router front panel 2"]
    assert all(i.thumbnail_url.startswith("https://api.openverse.org/") for i in result.images)
    assert result.images[0].attribution and result.images[0].source_page_url.startswith("https://")


def test_saved_and_rendered_answer_has_no_invented_image(settings: Settings) -> None:
    engine = get_engine(llm=FakeChatModel(reply=f"Elephant: ![Elephant]({WIKI_IMAGE})"), retriever=None, tools=[])
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=engine)
    with TestClient(app) as client:
        session_id = client.post("/api/sessions", json={}).json()["id"]
        html = client.post(f"/ui/sessions/{session_id}/messages", data={"content": "Show me an elephant"}).text
    assert "upload.wikimedia.org" not in html and "![" not in html and "Related images" not in html
    saved = SessionStore(settings.sessions_dir).get(session_id).messages[1]
    assert saved.content == "Elephant:" and saved.images == []


def test_image_rule_is_in_the_prompt_only_with_image_search() -> None:
    with_images, without = FakeChatModel(), FakeChatModel()
    get_engine(llm=with_images, retriever=None, tools=[*get_tools(), make_image_search_tool(FakeImages())]).invoke(
        "q", [], "s1"
    )
    get_engine(llm=without, retriever=None, tools=get_tools()).invoke("q", [], "s1")
    assert IMAGE_RULE in str(with_images.calls[0][0].content)
    assert IMAGE_RULE not in str(without.calls[0][0].content)


@pytest.mark.parametrize(
    ("provider", "history", "query"),
    [
        (
            FakeImages(error=ImageSearchError("Openverse did not answer in time (ReadTimeout after 12.0s)")),
            [],
            "router",
        ),
        (FakeImages(images=[]), [], "router"),  # no results
        (FakeImages(), "trigger-only", "Falcon"),  # skipped: no safe words
    ],
)
def test_no_image_results_tell_the_model_not_to_link_images(provider: FakeImages, history: Any, query: str) -> None:
    retriever = StaticRetriever([make_chunk(EXCERPTS, source="devices.md")]) if history else None
    question = "Show me a picture" if history else f"{TOOL_TRIGGER} router"
    engine = get_engine(llm=image_llm(query=query), retriever=retriever, tools=[make_image_search_tool(provider)])
    result = engine.invoke(question, TRIGGER_HISTORY if history else [], "s1")
    assert result.answer.endswith(NO_IMAGE_LINKS) and result.images == []


def test_repeated_calls_tell_the_model_not_to_link_images() -> None:
    llm = RepeatingImageCalls()
    get_engine(llm=llm, retriever=None, tools=[make_image_search_tool(FakeImages())]).invoke("router", [], "s1")
    refusals = [
        str(m.content) for m in llm.calls[-1] if isinstance(m, ToolMessage) and "already used" in str(m.content)
    ]
    assert refusals and all(r.endswith(NO_IMAGE_LINKS) for r in refusals)


# ---------------------------------------------------------------- outcome logs (no query, URL or document text)
SECRET_QUESTION = "Show me what the Falcon ZX-900 router looks like"


def _image_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "app.tools.image_search"]


def _assert_private(records: list[logging.LogRecord]) -> None:
    for record in records:
        text = record.getMessage().lower()
        assert not any(w in text for w in ("falcon", "zx-900", "router", "http", "pinhole", "engineers")), text


def test_success_is_logged_with_the_image_count(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.tools.image_search")
    engine = get_engine(
        llm=image_llm(query="falcon zx-900 router"), retriever=None, tools=[make_image_search_tool(FakeImages())]
    )
    assert len(engine.invoke(SECRET_QUESTION, TRIGGER_HISTORY, "s1").images) == 2
    records = _image_logs(caplog)
    assert [(r.levelname, r.getMessage()) for r in records] == [("INFO", "Image search returned 2 image(s)")]
    _assert_private(records)


def test_no_valid_results_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.tools.image_search")
    engine = get_engine(
        llm=image_llm(query="router"), retriever=None, tools=[make_image_search_tool(FakeImages(images=[]))]
    )
    engine.invoke(SECRET_QUESTION, TRIGGER_HISTORY, "s1")
    assert [r.getMessage() for r in _image_logs(caplog)] == ["Image search found no valid images"]
    _assert_private(_image_logs(caplog))


def test_unsafe_query_skip_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.tools.image_search")
    provider = FakeImages()
    retriever = StaticRetriever([make_chunk(EXCERPTS, source="devices.md")])
    engine = get_engine(
        llm=image_llm(query="Falcon pinhole"), retriever=retriever, tools=[make_image_search_tool(provider)]
    )
    engine.invoke("Show me a picture", TRIGGER_HISTORY, "s1")
    assert provider.queries == []
    assert [r.getMessage() for r in _image_logs(caplog)] == ["Image search skipped: no safe query words"]
    _assert_private(_image_logs(caplog))


def test_repeat_call_skip_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.tools.image_search")
    get_engine(llm=ParallelImageCalls(), retriever=None, tools=[make_image_search_tool(FakeImages())]).invoke(
        "router and modem pictures", [], "s1"
    )
    messages = sorted(r.getMessage() for r in _image_logs(caplog))
    assert messages == ["Image search returned 2 image(s)", "Image search skipped: already used for this question"]


def test_provider_failure_keeps_its_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.tools.image_search")
    error = ImageSearchError("Openverse did not answer in time (ReadTimeout after 12.0s)")
    engine = get_engine(
        llm=image_llm(query="router"), retriever=None, tools=[make_image_search_tool(FakeImages(error=error))]
    )
    result = engine.invoke(SECRET_QUESTION, TRIGGER_HISTORY, "s1")
    records = _image_logs(caplog)
    assert [(r.levelname, r.getMessage()) for r in records] == [
        ("WARNING", "Image search failed: Openverse did not answer in time (ReadTimeout after 12.0s)")
    ]
    assert result.images == [] and result.answer.startswith("ANSWER | tool said: Error")


# ---------------------------------------------------------------- broken thumbnails (static/app.js, run in Node)
APP_JS = Path(__file__).resolve().parent.parent / "static" / "app.js"
# A tiny stand-in DOM: just what app.js uses (closest, contains, style, classList, addEventListener).
DOM_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
class El {
  constructor(tag, cls, parent) {
    this.tag = tag; this.classes = new Set(cls ? [cls] : []); this.parent = parent || null;
    this.style = {}; this.removed = false; if (parent) (parent.children = parent.children || []).push(this);
    this.classList = { add: (c) => this.classes.add(c), contains: (c) => this.classes.has(c) };
  }
  matches(sel) { return sel.startsWith(".") ? this.classes.has(sel.slice(1)) : this.tag === sel; }
  closest(sel) { for (let n = this; n; n = n.parent) if (n.matches(sel)) return n; return null; }
  contains(other) { for (let n = other; n; n = n.parent) if (n === this) return true; return false; }
  remove() { this.removed = true; }
}
class HTMLImageElement extends El {}
const handlers = [], warnings = [];
globalThis.HTMLImageElement = HTMLImageElement;
globalThis.document = { addEventListener: (type, fn, capture) => handlers.push({ type, fn, capture }) };
console.warn = (...args) => warnings.push(args.join(" "));
vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"));

const section = new El("section", "related-images");
const grid = new El("div", "image-grid", section);
const cards = [0, 1, 2].map(() => {
  const card = new El("figure", "image-card", grid);
  const wrapper = new El("a", null, card);
  const img = new HTMLImageElement("img", null, wrapper);
  const caption = new El("figcaption", null, card);
  const links = ["creator", "license", "view-source"].map(() => new El("a", null, caption));
  return { card, wrapper, img, caption, links };
});
const loose = new HTMLImageElement("img", null, null);  // an image outside any card
const fire = (target) => handlers.filter((h) => h.type === "error").forEach((h) => h.fn({ target }));
const failing = Number(process.argv[2]);
cards.slice(0, failing).forEach(({ img }) => fire(img));
fire(loose); fire(new El("script", null, null));
console.log(JSON.stringify({
  capture: handlers.every((h) => h.capture === true),
  cards: cards.map(({ card, wrapper, img, caption, links }) => ({
    removed: card.removed, wrapperHidden: wrapper.style.display === "none", imgDisplay: img.style.display || "",
    marked: card.classList.contains("thumbnail-unavailable"), captionVisible: caption.style.display !== "none",
    linksVisible: links.every((l) => l.style.display !== "none" && !l.removed),
  })),
  sectionRemoved: section.removed, looseDisplay: loose.style.display || "", warnings,
}));
"""


def _run_app_js(failing_thumbnails: int) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; the browser script cannot be exercised")
    done = subprocess.run(
        [node, "-e", DOM_HARNESS, str(APP_JS), str(failing_thumbnails)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result: dict[str, Any] = json.loads(done.stdout)
    return result


def test_one_failed_thumbnail_hides_only_its_preview() -> None:
    result = _run_app_js(failing_thumbnails=1)
    first, *others = result["cards"]
    assert result["capture"] is True, "capturing listener, so HTMX-inserted messages are covered too"
    assert first == {
        "removed": False,
        "wrapperHidden": True,
        "imgDisplay": "",
        "marked": True,
        "captionVisible": True,
        "linksVisible": True,
    }, "card, title, creator, license and View source stay; only the picture (and its link) is hidden"
    assert all(not c["removed"] and not c["wrapperHidden"] and not c["marked"] for c in others)
    assert result["sectionRemoved"] is False and result["looseDisplay"] == ""


def test_all_failed_thumbnails_keep_every_card_and_the_section() -> None:
    result = _run_app_js(failing_thumbnails=3)
    assert all(not c["removed"] and c["wrapperHidden"] and c["linksVisible"] for c in result["cards"])
    assert result["sectionRemoved"] is False


def test_failed_thumbnail_warning_is_concise_and_private() -> None:
    warnings = _run_app_js(failing_thumbnails=2)["warnings"]
    assert warnings == ["Image preview could not be loaded; showing the card without it."] * 2
    assert not any("http" in w or "openverse" in w.lower() for w in warnings)
