"""Human-in-the-loop review: LangGraph interrupt/resume, persistence, session isolation, UI and API.

Fake LLMs and retrievers only (no API keys, no network); checkpoints and reviews live in temp folders.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.tools import tool
from langchain_core.tracers.context import collect_runs

from app.chat.checkpoints import close_checkpointer, memory_checkpointer, open_checkpointer
from app.chat.engine import (
    ACTION_REJECTED,
    ANSWER_REJECTED,
    GraphEngine,
    ReviewUnavailableError,
    _choice,
    get_engine,
)
from app.chat.review import REVIEW_TOOL_NAME, ReviewPolicy, mark_requires_approval, support_reasons
from app.config import Settings
from app.datasets.parsing import parse_csv
from app.datasets.store import DatasetStore
from app.deps import Components, get_components
from app.main import create_app
from app.models import ReviewDecision
from app.services import REVIEW_EXPIRED, WAITING_FOR_REVIEW
from app.tools import get_tools
from app.tools.dataset_tools import dataset_context, make_dataset_tools
from tests.fakes import TOOL_TRIGGER, FakeChatModel, StaticRetriever, make_chunk

LEAVE = make_chunk("Employees get 20 days of annual leave per year.", source="leave_policy.md")
CITED = "You get 20 days of annual leave [1]."
UNCITED = "You probably get some leave."
SESSION = "a" * 32


def engine(
    reply: str = CITED,
    chunks: list[Any] | None = None,
    tools: list[Any] | None = None,
    policy: ReviewPolicy | None = ReviewPolicy(),
    checkpointer: Any = None,
    **llm_options: Any,
) -> GraphEngine:
    built = get_engine(
        llm=FakeChatModel(reply=reply, **llm_options),
        retriever=StaticRetriever([LEAVE] if chunks is None else chunks),
        tools=tools or [],
        review_policy=policy,
        checkpointer=checkpointer,
    )
    assert isinstance(built, GraphEngine)
    return built


def threads(saver: Any) -> set[str]:
    return {c.config["configurable"]["thread_id"] for c in saver.list(None)}


executed: list[dict[str, Any]] = []


@tool
def update_customer_record(customer_id: str, field: str, value: str) -> str:
    """Change one field of a customer record (test stand-in for a consequential tool; no real system)."""
    executed.append({"customer_id": customer_id, "field": field, "value": value})
    return f"Updated {field} of {customer_id}."


mark_requires_approval(update_customer_record)
ACTION_ARGS = {"customer_id": "c-1", "field": "email", "value": "new@example.com"}


@pytest.fixture(autouse=True)
def reset_executed() -> None:
    executed.clear()


_opened_savers: list[Any] = []


def open_saver(path: Path) -> Any:
    """A SQLite saver on a temp file that is closed after the test (see close_savers)."""
    saver = open_checkpointer(path)
    _opened_savers.append(saver)
    return saver


@pytest.fixture(autouse=True)
def close_savers() -> Iterator[None]:
    yield
    while _opened_savers:
        close_checkpointer(_opened_savers.pop())  # closing twice is harmless


def action_engine(**options: Any) -> GraphEngine:
    return engine(tools=[update_customer_record], tool_name="update_customer_record", tool_args=ACTION_ARGS, **options)


# ---------------------------------------------------------------- 1. normal questions are not held
def test_supported_rag_answer_continues_and_leaves_no_checkpoint() -> None:
    saver = memory_checkpointer()
    result = engine(checkpointer=saver).invoke("How much annual leave do I get?", [], SESSION)
    assert result.review is None and result.answer == CITED
    assert [s.source for s in result.sources] == ["leave_policy.md"]
    assert threads(saver) == set(), "finished turns are deleted from the checkpointer"


def test_general_question_without_documents_continues() -> None:
    result = engine(reply="Paris is the capital of France.", chunks=[]).invoke("Capital of France?", [], SESSION)
    assert result.review is None and result.answer == "Paris is the capital of France."


def test_tool_based_answer_is_not_held_even_without_citations() -> None:
    built = engine(reply="The result", tools=get_tools(), tool_name="calculator", tool_args={"expression": "2+3"})
    result = built.invoke(f"{TOOL_TRIGGER} 2+3?", [], SESSION)
    assert result.review is None and result.answer.startswith("The result | tool said: 5")


def test_csv_and_excel_analysis_continue_without_approval(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    data = b"category,sales\nTools,10\nHome,5\nTools,2\n"
    store.save(SESSION, "sales.csv", parse_csv(data), len(data))
    built = get_engine(
        llm=FakeChatModel(
            reply="ANSWER",
            tool_name="query_dataset",
            tool_args={"group_by": ["category"], "aggregations": [{"column": "sales", "function": "sum"}]},
        ),
        retriever=StaticRetriever([LEAVE]),  # even with an excerpt in the prompt
        tools=[*get_tools(), *make_dataset_tools(store)],
        session_context=functools.partial(dataset_context, store),
        review_policy=ReviewPolicy(),
    )
    result = built.invoke(f"{TOOL_TRIGGER} total sales by category", [], SESSION)
    assert result.review is None and result.analysis[0].title == "Grouped by category"


def test_without_a_policy_answers_are_never_held() -> None:
    result = engine(reply=UNCITED, policy=None).invoke("I want a refund", [], SESSION)
    assert result.review is None and result.answer == UNCITED


# ---------------------------------------------------------------- 2. review-required answers pause the graph
def test_uncited_answer_pauses_with_a_real_interrupt() -> None:
    saver = memory_checkpointer()
    built = engine(reply=UNCITED, checkpointer=saver)
    result = built.invoke("How much annual leave do I get?", [], SESSION)
    review = result.review
    assert review is not None and result.answer == ""
    assert review.kind == "unsupported_answer" and review.proposed_answer == UNCITED
    assert review.question == "How much annual leave do I get?"
    assert "does not cite any of the 1 retrieved document excerpt(s)" in review.reasons[0]
    assert [s.source for s in review.sources] == ["leave_policy.md"]
    snapshot = built.graph.get_state({"configurable": {"thread_id": review.thread_id}})
    assert snapshot.next == ("human_review",) and snapshot.interrupts[0].value["kind"] == "unsupported_answer"
    assert threads(saver) == {review.thread_id}, "the paused turn is checkpointed"


def test_citation_checks_are_structural_not_scores() -> None:
    assert support_reasons("See [1].", 2) == []
    assert support_reasons("See [1, 2].", 2) == []
    assert "cites [7]" in support_reasons("See [7].", 2)[0]
    assert support_reasons("No citation.", 0) == [], "nothing retrieved: nothing to check"
    result = engine(reply="See [7].").invoke("Leave?", [], SESSION)
    assert result.review is not None and any("cites [7]" in r for r in result.review.reasons)
    assert not any("confidence" in r.lower() or "%" in r for r in result.review.reasons), "no invented scores"


@pytest.mark.parametrize(
    "question",
    ["I want a refund for my last order", "Please close my account today", "I will contact my lawyer about this"],
)
def test_sensitive_requests_pause_before_the_reply_is_shown(question: str) -> None:
    result = engine(reply=CITED).invoke(question, [], SESSION)
    assert result.review is not None and result.review.kind == "sensitive_request"
    assert result.review.proposed_answer == CITED and result.answer == ""


def test_model_can_request_a_review_but_not_approve_it() -> None:
    built = engine(
        reply="Proposed reply [1].",
        tool_name=REVIEW_TOOL_NAME,
        tool_args={"category": "sensitive_request", "reason": "Customer is upset about a delivery."},
    )
    assert REVIEW_TOOL_NAME in built.llm.bound_tools  # type: ignore[attr-defined]
    result = built.invoke(f"{TOOL_TRIGGER} my parcel never came", [], SESSION)
    assert result.review is not None and result.review.kind == "sensitive_request"
    assert "The assistant asked for a review: Customer is upset about a delivery." in result.review.reasons
    assert _choice({"choice": "approved!!"}) == "reject" and _choice("approve") == "reject", "fail closed"


# ---------------------------------------------------------------- 4-6. resume with approve / reject / edit
def paused(reply: str = UNCITED, **options: Any) -> tuple[GraphEngine, str]:
    built = engine(reply=reply, **options)
    result = built.invoke("How much annual leave do I get?", [], SESSION)
    assert result.review is not None
    return built, result.review.thread_id


def test_approval_resumes_and_shows_the_proposed_answer() -> None:
    saver = memory_checkpointer()
    built, thread = paused(checkpointer=saver)
    result = built.resume(thread, SESSION, ReviewDecision(choice="approve"))
    assert result.review is None and result.answer == UNCITED
    assert [s.source for s in result.sources] == ["leave_policy.md"]
    assert threads(saver) == set()
    with pytest.raises(ReviewUnavailableError):  # a finished turn cannot be resumed again
        built.resume(thread, SESSION, ReviewDecision(choice="approve"))


def test_rejection_replaces_the_answer() -> None:
    built, thread = paused()
    result = built.resume(thread, SESSION, ReviewDecision(choice="reject"))
    assert result.answer == ANSWER_REJECTED and UNCITED not in result.answer and result.sources == []


def test_edited_response_is_used_by_the_following_node() -> None:
    built, thread = paused()
    result = built.resume(thread, SESSION, ReviewDecision(choice="edit", response="Full-time staff get 20 days [1]."))
    assert result.answer == "Full-time staff get 20 days [1]."


def test_resume_checks_the_session_bound_to_the_thread() -> None:
    built, thread = paused()
    with pytest.raises(ReviewUnavailableError):
        built.resume(thread, "b" * 32, ReviewDecision(choice="approve"))
    assert built.resume(thread, SESSION, ReviewDecision(choice="approve")).answer == UNCITED, "still pending for A"


# ---------------------------------------------------------------- 3C. consequential actions need approval first
def test_consequential_tool_does_not_run_before_approval() -> None:
    built = action_engine()
    result = built.invoke(f"{TOOL_TRIGGER} update my email", [], SESSION)
    review = result.review
    assert review is not None and review.kind == "consequential_action"
    assert [a.model_dump() for a in review.actions] == [{"tool": "update_customer_record", "arguments": ACTION_ARGS}]
    assert executed == [], "nothing runs while the approval is pending"
    approved = built.resume(review.thread_id, SESSION, ReviewDecision(choice="approve"))
    assert executed == [ACTION_ARGS] and "Updated email of c-1." in approved.answer
    assert approved.review is None, "the approved action is not reviewed a second time"


def test_rejected_action_is_never_executed() -> None:
    built = action_engine()
    review = built.invoke(f"{TOOL_TRIGGER} update my email", [], SESSION).review
    assert review is not None
    result = built.resume(review.thread_id, SESSION, ReviewDecision(choice="reject"))
    assert executed == [] and result.answer == ACTION_REJECTED.format(names="update_customer_record")


def test_action_approval_applies_without_a_review_policy() -> None:
    result = action_engine(policy=None).invoke(f"{TOOL_TRIGGER} update my email", [], SESSION)
    assert result.review is not None and result.review.kind == "consequential_action" and executed == []


# ---------------------------------------------------------------- app: persistence, UI, API, isolation
@pytest.fixture()
def app_with(settings: Settings) -> Iterator[Any]:
    """create(engine) -> (TestClient, app); every app shares the temp settings (same files)."""
    clients: list[TestClient] = []

    def create(built: GraphEngine) -> tuple[TestClient, FastAPI]:
        app = create_app(settings)
        app.dependency_overrides[get_components] = lambda: Components(engine=built)
        client = TestClient(app)
        clients.append(client)
        return client, app

    yield create
    for client in clients:
        client.close()


def new_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={})
    assert response.status_code == 201
    return str(response.json()["id"])


def ask(client: TestClient, session_id: str, content: str = "How much annual leave do I get?") -> dict[str, Any]:
    response = client.post(f"/api/sessions/{session_id}/messages", json={"content": content})
    assert response.status_code == 200
    return dict(response.json())


def sqlite_engine(settings: Settings, reply: str = UNCITED, **options: Any) -> GraphEngine:
    return engine(reply=reply, checkpointer=open_saver(settings.checkpoint_path), **options)


def test_pending_review_is_persisted_and_survives_page_reload(app_with: Any, settings: Settings) -> None:
    client, _ = app_with(sqlite_engine(settings))
    sid = new_session(client)
    turn = ask(client, sid)
    message = turn["assistant_message"]
    assert message["content"] == WAITING_FOR_REVIEW["unsupported_answer"]
    review_id = message["review"]["id"]
    assert message["review"] == {"id": review_id, "kind": "unsupported_answer", "status": "pending"}
    stored = json.loads((settings.review_root / sid / f"{review_id}.json").read_text(encoding="utf-8"))
    assert stored["status"] == "pending" and stored["thread_id"]
    for _ in range(2):  # reload twice
        page = client.get(f"/sessions/{sid}").text
        assert "Pending reviews" in page and "Answer not supported by the documents" in page
        assert "How much annual leave do I get?" in page and UNCITED in page and "leave_policy.md" in page
        assert 'value="approve"' in page and 'value="reject"' in page and 'name="response"' in page
        assert "Submit decision" in page and "Waiting for review" in page


def test_server_restart_keeps_and_resumes_the_paused_turn(app_with: Any, settings: Settings) -> None:
    first = sqlite_engine(settings)
    client, _ = app_with(first)
    sid = new_session(client)
    review_id = ask(client, sid)["assistant_message"]["review"]["id"]
    close_checkpointer(first.checkpointer)  # type: ignore[arg-type]  # "server stops"
    client.close()

    second = sqlite_engine(settings)  # new process: new app, new connection to the same SQLite file
    client, _ = app_with(second)
    assert "Answer not supported by the documents" in client.get(f"/sessions/{sid}").text
    response = client.post(f"/api/sessions/{sid}/reviews/{review_id}/decision", json={"choice": "approve"})
    assert response.status_code == 200
    body = response.json()
    assert body["review"]["status"] == "approved" and body["message"]["content"] == UNCITED
    assert body["message"]["sources"][0]["source"] == "leave_policy.md"
    session = client.get(f"/api/sessions/{sid}").json()
    assert [m["content"] for m in session["messages"]] == ["How much annual leave do I get?", UNCITED]
    assert session["messages"][1]["review"]["status"] == "approved", "the waiting message was replaced in place"
    assert threads(second.checkpointer) == set()


def test_ui_decision_updates_message_and_card(app_with: Any, settings: Settings) -> None:
    client, _ = app_with(engine(reply=UNCITED))
    sid = new_session(client)
    reply = client.post(f"/ui/sessions/{sid}/messages", data={"content": "Leave days?"}, headers={"HX-Request": "1"})
    assert 'id="message-1"' in reply.text and 'hx-swap-oob="outerHTML"' in reply.text, "panel refreshed out of band"
    review_id = client.get(f"/api/sessions/{sid}/reviews").json()[0]["id"]
    response = client.post(
        f"/ui/sessions/{sid}/reviews/{review_id}/decide",
        data={"choice": "edit", "response": "Staff get 20 days [1].", "note": "checked policy"},
        headers={"HX-Request": "1"},
    )
    assert response.status_code == 200
    assert response.headers["HX-Retarget"] == "#message-1" and response.headers["HX-Reswap"] == "innerHTML"
    assert "Staff get 20 days [1]." in response.text and "Edited by a reviewer" in response.text
    assert "Edited and approved" in response.text and "Decision saved: edited." in response.text
    assert 'name="choice"' not in response.text, "a decided card has no form"
    review = client.get(f"/api/sessions/{sid}/reviews/{review_id}").json()
    assert review["status"] == "edited" and review["final_answer"] == "Staff get 20 days [1]."
    assert review["note"] == "checked policy"


def test_ui_rejection_and_invalid_decisions(app_with: Any) -> None:
    client, _ = app_with(engine(reply=UNCITED))
    sid = new_session(client)
    review_id = ask(client, sid)["assistant_message"]["review"]["id"]
    empty_edit = client.post(f"/ui/sessions/{sid}/reviews/{review_id}/decide", data={"choice": "edit", "response": " "})
    assert "Write the edited response" in empty_edit.text and "HX-Retarget" not in empty_edit.headers
    bogus = client.post(f"/ui/sessions/{sid}/reviews/{review_id}/decide", data={"choice": "auto-approve"})
    assert "That decision is not valid." in bogus.text
    assert client.get(f"/api/sessions/{sid}/reviews/{review_id}").json()["status"] == "pending", "still decidable"
    rejected = client.post(f"/ui/sessions/{sid}/reviews/{review_id}/decide", data={"choice": "reject"})
    assert ANSWER_REJECTED in rejected.text and UNCITED in rejected.text  # card still shows the proposal


def test_other_session_cannot_see_or_decide_the_review(app_with: Any) -> None:
    client, _ = app_with(engine(reply=UNCITED))
    owner, intruder = new_session(client), new_session(client)
    review_id = ask(client, owner)["assistant_message"]["review"]["id"]
    assert client.get(f"/api/sessions/{intruder}/reviews").json() == []
    assert client.get(f"/api/sessions/{intruder}/reviews/{review_id}").status_code == 404
    api = client.post(f"/api/sessions/{intruder}/reviews/{review_id}/decision", json={"choice": "approve"})
    ui = client.post(f"/ui/sessions/{intruder}/reviews/{review_id}/decide", data={"choice": "approve"})
    assert api.status_code == 404 and ui.status_code == 404
    assert client.get(f"/api/sessions/{owner}/reviews/{review_id}").json()["status"] == "pending"
    crafted = client.post(f"/api/sessions/{owner}/reviews/..%2F{review_id}/decision", json={"choice": "approve"})
    assert crafted.status_code == 404, "only the store's own id format is accepted"


def test_duplicate_and_replayed_decisions_are_rejected(app_with: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    built = engine(reply=UNCITED)
    resumes: list[str] = []
    original = built.resume

    def counting(*args: Any, **kwargs: Any) -> Any:
        resumes.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(built, "resume", counting)
    client, _ = app_with(built)
    sid = new_session(client)
    review_id = ask(client, sid)["assistant_message"]["review"]["id"]
    url = f"/api/sessions/{sid}/reviews/{review_id}/decision"
    assert client.post(url, json={"choice": "approve"}).status_code == 200
    again = client.post(url, json={"choice": "reject"})
    assert again.status_code == 409 and "already approved" in again.json()["detail"]
    ui_replay = client.post(f"/ui/sessions/{sid}/reviews/{review_id}/decide", data={"choice": "approve"})
    assert "already approved" in ui_replay.text and "HX-Retarget" not in ui_replay.headers
    assert len(resumes) == 1, "the graph was resumed exactly once"


def test_expired_review_is_refused_and_its_checkpoint_dropped(app_with: Any, settings: Settings) -> None:
    built = sqlite_engine(settings)
    client, _ = app_with(built)
    sid = new_session(client)
    review_id = ask(client, sid)["assistant_message"]["review"]["id"]
    path = settings.review_root / sid / f"{review_id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["created_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
    path.write_text(json.dumps(record), encoding="utf-8")
    response = client.post(f"/api/sessions/{sid}/reviews/{review_id}/decision", json={"choice": "approve"})
    assert response.status_code == 410
    assert client.get(f"/api/sessions/{sid}/reviews/{review_id}").json()["status"] == "expired"
    assert client.get(f"/api/sessions/{sid}").json()["messages"][1]["content"] == REVIEW_EXPIRED
    assert threads(built.checkpointer) == set()


def test_stale_reviews_expire_when_the_chat_is_opened(app_with: Any, settings: Settings) -> None:
    client, _ = app_with(engine(reply=UNCITED))
    sid = new_session(client)
    review_id = ask(client, sid)["assistant_message"]["review"]["id"]
    path = settings.review_root / sid / f"{review_id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["created_at"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(record), encoding="utf-8")
    page = client.get(f"/sessions/{sid}").text
    assert "Review expired" in page and 'name="choice"' not in page


def test_missing_checkpoint_is_handled_as_expired(app_with: Any) -> None:
    client, app = app_with(engine(reply=UNCITED))
    sid = new_session(client)
    review_id = ask(client, sid)["assistant_message"]["review"]["id"]
    fresh = engine(reply=UNCITED)  # e.g. the checkpoint file was deleted: a new, empty checkpointer
    app.dependency_overrides[get_components] = lambda: Components(engine=fresh)
    response = client.post(f"/api/sessions/{sid}/reviews/{review_id}/decision", json={"choice": "approve"})
    assert response.status_code == 410
    assert client.get(f"/api/sessions/{sid}/reviews/{review_id}").json()["status"] == "expired"


def test_action_reviews_cannot_be_edited_via_api(app_with: Any) -> None:
    client, _ = app_with(action_engine())
    sid = new_session(client)
    review_id = ask(client, sid, f"{TOOL_TRIGGER} update my email")["assistant_message"]["review"]["id"]
    url = f"/api/sessions/{sid}/reviews/{review_id}/decision"
    edit = client.post(url, json={"choice": "edit", "response": "done"})
    assert edit.status_code == 422 and executed == []
    assert client.post(url, json={"choice": "approve", "extra": "x"}).status_code == 422, "unknown fields refused"
    page = client.get(f"/sessions/{sid}").text
    assert "Proposed action (not performed yet)" in page and "update_customer_record" in page
    assert 'value="edit"' not in page
    assert client.post(url, json={"choice": "reject"}).json()["message"]["content"].startswith("I did not perform")
    assert executed == []


def test_thread_ids_never_reach_the_client(app_with: Any, settings: Settings) -> None:
    client, _ = app_with(engine(reply=UNCITED))
    sid = new_session(client)
    review_id = ask(client, sid)["assistant_message"]["review"]["id"]
    thread_id = json.loads((settings.review_root / sid / f"{review_id}.json").read_text(encoding="utf-8"))["thread_id"]
    exposed = [
        client.get(f"/api/sessions/{sid}/reviews").text,
        client.get(f"/api/sessions/{sid}/reviews/{review_id}").text,
        client.get(f"/sessions/{sid}").text,
        client.get(f"/api/sessions/{sid}").text,
        client.post(f"/api/sessions/{sid}/reviews/{review_id}/decision", json={"choice": "approve"}).text,
    ]
    assert all(thread_id not in text and "thread_id" not in text for text in exposed)
    assert all(str(settings.checkpoint_path) not in text for text in exposed)


def test_waiting_placeholders_are_not_sent_to_the_model_as_answers(app_with: Any) -> None:
    built = engine(reply=UNCITED)
    client, _ = app_with(built)
    sid = new_session(client)
    ask(client, sid)
    ask(client, sid, "And sick leave?")
    assert WAITING_FOR_REVIEW["unsupported_answer"] not in built.llm.prompt_text()  # type: ignore[attr-defined]


def test_deleting_a_session_removes_its_reviews_and_checkpoints(app_with: Any, settings: Settings) -> None:
    built = sqlite_engine(settings)
    client, _ = app_with(built)
    sid = new_session(client)
    ask(client, sid)
    assert threads(built.checkpointer)
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert not (settings.review_root / sid).exists() and threads(built.checkpointer) == set()


def test_checkpoint_file_is_created_by_startup_not_by_import(settings: Settings) -> None:
    app = create_app(settings)
    assert not settings.checkpoint_path.exists(), "building the app (as importing app.main does) opens nothing"
    with TestClient(app) as client:
        client.get("/health")
        assert settings.checkpoint_path.is_file(), "the lifespan start-up opens CHECKPOINT_DB"


def test_sqlite_checkpoints_restore_validated_models(settings: Settings) -> None:
    saver = open_saver(settings.checkpoint_path)
    built, thread = paused(checkpointer=saver)
    close_checkpointer(saver)
    reopened = engine(reply=UNCITED, checkpointer=open_saver(settings.checkpoint_path))
    result = reopened.resume(thread, SESSION, ReviewDecision(choice="approve"))
    assert result.sources[0].source == "leave_policy.md" and result.sources[0].score == 1.0


def test_app_wiring_uses_sqlite_checkpoints_and_the_review_policy(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.deps as deps

    captured: dict[str, Any] = {}

    def fake_get_engine(**kwargs: Any) -> GraphEngine:
        captured.update(kwargs)
        return engine(checkpointer=kwargs["checkpointer"], policy=kwargs.get("review_policy"))

    monkeypatch.setattr(deps, "get_engine", fake_get_engine)
    with TestClient(create_app(settings)) as client:
        sid = new_session(client)
        assert ask(client, sid, "I want a refund")["assistant_message"]["review"]["kind"] == "sensitive_request"
    assert isinstance(captured["review_policy"], ReviewPolicy)
    assert type(captured["checkpointer"]).__name__ == "SqliteSaver" and settings.checkpoint_path.is_file()


def test_human_review_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.deps as deps
    from app.config import get_settings

    captured: dict[str, Any] = {}
    monkeypatch.setattr(deps, "get_engine", lambda **kwargs: captured.update(kwargs) or engine(policy=None))
    monkeypatch.setenv("HUMAN_REVIEW", "false")
    get_settings.cache_clear()
    deps.build_components()
    assert "review_policy" not in captured


# ---------------------------------------------------------------- 13. LangSmith tracing metadata
def test_paused_and_resumed_runs_are_traced_and_correlated() -> None:
    built = engine(reply=UNCITED)
    with collect_runs() as paused_runs:
        review = built.invoke("I want a refund", [], SESSION).review
    assert review is not None
    with collect_runs() as resumed_runs:
        built.resume(
            review.thread_id, SESSION, ReviewDecision(choice="reject"), {"review_id": "r1", "review_kind": review.kind}
        )
    first = paused_runs.traced_runs[0].extra["metadata"]
    second = resumed_runs.traced_runs[0].extra["metadata"]
    assert first["session_id"] == second["session_id"] == SESSION
    assert first["thread_id"] == second["thread_id"] == review.thread_id, "both runs share the LangGraph thread"
    assert second["review_id"] == "r1" and second["review_kind"] == "sensitive_request"
    assert second["review_decision"] == "reject" and second["hitl"] == "resume"
    assert "hitl-resume" in (resumed_runs.traced_runs[0].tags or [])
    for metadata in (first, second):
        text = json.dumps(metadata, default=str)
        assert UNCITED not in text and "refund" not in text.lower(), "no answer or message text in metadata"
        assert LEAVE.text not in text, "no document contents in metadata"
        assert not any(word in key.lower() for key in metadata for word in ("key", "secret", "token", "password"))
        assert "lsv2" not in text and "AIza" not in text, "no API-key-like values"
