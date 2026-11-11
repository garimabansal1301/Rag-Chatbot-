"""Upload -> index -> ask: the document upload flow end to end. Temp folders, fake embedder and fake LLM only."""

from __future__ import annotations

import io
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.chat.engine import get_engine
from app.config import Settings
from app.deps import Components, get_components
from app.ingest.pipeline import document_id, load_document
from app.main import create_app
from app.services import UploadError, safe_filename, save_and_index_upload
from app.vectordb.chunking import get_chunker
from app.vectordb.retriever import VectorRetriever
from app.vectordb.store import VectorStore, get_vector_store
from tests.conftest import SAMPLE_DOCS
from tests.fakes import FakeChatModel, HashEmbedder

POLICY_NAME = "Work Engagement Policy.pdf"
POLICY_PAGES = (
    "Work Engagement Policy. Employees may work remotely up to three days per week.",
    "Core collaboration hours are 10:00 to 15:00 local time. Remote work requires manager approval.",
)


def make_pdf(*page_texts: str) -> bytes:
    """A minimal valid PDF, one page per text in Helvetica. An empty text gives a page with no text, like a scan."""
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for text in page_texts:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 11 Tf 40 760 Td ({escaped}) Tj ET".encode() if text else b"0 0 200 200 re f"
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            b"/Contents %d 0 R >>" % (len(objects))
        )
        kids.append(b"%d 0 R" % len(objects))
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(kids), len(kids))
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    out.write(b"".join(b"%010d 00000 n \n" % offset for offset in offsets))
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref))
    return out.getvalue()


def locked_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(user_password="secret", owner_password="owner", algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class FailingEmbedder(HashEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model unavailable")


@dataclass
class App:
    client: TestClient
    settings: Settings
    store: VectorStore
    llm: FakeChatModel

    def upload(self, name: str, data: bytes, api: str = "/api/documents") -> Any:  # the test client's Response
        return self.client.post(api, files={"file": (name, data, "application/octet-stream")})

    def ask(self, question: str) -> dict[str, object]:
        session_id = self.client.post("/api/sessions", json={}).json()["id"]
        response = self.client.post(f"/api/sessions/{session_id}/messages", json={"content": question})
        assert response.status_code == 200
        return dict(response.json()["assistant_message"])

    def saved_files(self) -> list[str]:
        folder = self.settings.documents_dir
        return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def _build(settings: Settings, embedder: HashEmbedder | None = None) -> Iterator[App]:
    embedder = embedder or HashEmbedder()
    store = get_vector_store(settings.vector_dir)
    retriever = VectorRetriever(store, HashEmbedder(), k=4)
    llm = FakeChatModel(reply="ANSWER")
    parts = Components(
        engine=get_engine(llm=llm, retriever=retriever, tools=[]),
        chunker=get_chunker("fixed", chunk_size=300, chunk_overlap=50),
        embedder=embedder,
        store=store,
        retriever=retriever,
    )
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: parts
    with TestClient(app) as client:
        yield App(client, settings, store, llm)


@pytest.fixture()
def rag(settings: Settings) -> Iterator[App]:
    yield from _build(settings)


# ---------------------------------------------------------------- successful uploads
@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("notes.txt", b"Badge photos are taken on the first Monday of each month."),
        ("guide.md", b"# Guide\n\nThe help desk is on the third floor, room 301."),
        (POLICY_NAME, make_pdf(*POLICY_PAGES)),
    ],
)
def test_supported_uploads_are_saved_and_indexed(rag: App, name: str, data: bytes) -> None:
    response = rag.upload(name, data)
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == name and body["indexed"] is True
    assert body["chunks_indexed"] == rag.store.count() > 0
    assert (rag.settings.documents_dir / name).read_bytes() == data
    assert rag.client.get("/api/documents").json() == [name]


def test_pdf_text_from_every_page_is_extracted(tmp_path: Path) -> None:
    path = tmp_path / POLICY_NAME
    path.write_bytes(make_pdf(POLICY_PAGES[0], "", POLICY_PAGES[1]))
    document = load_document(path)
    assert "three days per week" in document.text and "10:00 to 15:00" in document.text
    assert document.source == POLICY_NAME and document.metadata["pages"] == 3


def test_ui_upload_reports_filename_and_chunk_count(rag: App) -> None:
    html = rag.upload(POLICY_NAME, make_pdf(*POLICY_PAGES), api="/ui/documents").text
    assert 'class="ok"' in html and POLICY_NAME in html
    count = rag.store.count()
    assert f"indexed {count} chunk{'' if count == 1 else 's'}." in html and "ask about it now" in html


def test_upload_form_offers_pdf_and_shows_the_limit(rag: App) -> None:
    session_id = rag.client.post("/api/sessions", json={}).json()["id"]
    html = rag.client.get(f"/sessions/{session_id}").text
    assert 'accept=".md,.txt,.pdf"' in html and "max 5 MB" in html and "Uploading and indexing" in html


# ---------------------------------------------------------------- rejected uploads leave nothing behind
@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("tool.exe", b"MZ", "Unsupported file type"),
        ("noext", b"hello", "Unsupported file type"),
        ("empty.md", b"", "File is empty"),
        ("blank.txt", b"  \n\t ", "File is empty"),
        ("menu.txt", "caf\xe9".encode("latin-1"), "UTF-8"),
        ("scan.pdf", make_pdf("", ""), "no extractable text"),
        ("broken.pdf", b"%PDF-1.4\nthis is not really a PDF", "PDF"),
        ("locked.pdf", locked_pdf(), "password-protected"),
    ],
)
def test_bad_uploads_are_rejected_without_side_effects(rag: App, name: str, data: bytes, message: str) -> None:
    response = rag.upload(name, data)
    assert response.status_code == 400 and message in response.json()["detail"]
    assert rag.saved_files() == [] and rag.store.count() == 0


def test_ui_shows_errors_as_errors(rag: App) -> None:
    html = rag.upload("scan.pdf", make_pdf(""), api="/ui/documents").text
    assert 'class="error"' in html and "no extractable text" in html and "scanned" in html


def test_oversized_upload_is_rejected(settings: Settings) -> None:
    for app in _build(replace(settings, max_upload_bytes=1024)):
        response = app.upload("big.txt", b"a" * 1025)
        assert response.status_code == 400 and "too large" in response.json()["detail"]
        assert app.upload("ok.txt", b"a" * 1024).status_code == 201


def test_upload_limit_is_configurable_in_megabytes(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import load_settings

    monkeypatch.setenv("MAX_UPLOAD_MB", "12")
    assert load_settings().max_upload_bytes == 12 * 1024 * 1024


def test_indexing_failure_reports_an_error_and_saves_nothing(settings: Settings) -> None:
    for app in _build(settings, embedder=FailingEmbedder()):
        response = app.upload("notes.md", b"Some text worth indexing.")
        assert response.status_code == 400 and "indexing failed" in response.json()["detail"]
        assert app.saved_files() == [] and app.store.count() == 0


# ---------------------------------------------------------------- ask right after uploading
def test_question_after_upload_retrieves_the_new_document_with_citations(rag: App) -> None:
    assert rag.upload("leave_policy.md", (SAMPLE_DOCS / "leave_policy.md").read_bytes()).status_code == 201
    assert rag.upload(POLICY_NAME, make_pdf(*POLICY_PAGES)).status_code == 201

    answer = rag.ask("What are the core collaboration hours for remote work?")
    sources = answer["sources"]
    assert isinstance(sources, list) and sources[0]["source"] == POLICY_NAME
    assert sources[0]["chunk_id"].startswith(document_id(POLICY_NAME) + ":")
    assert "10:00 to 15:00" in rag.llm.prompt_text(), "the uploaded text must reach the LLM prompt"

    leave = rag.ask("How many days of annual leave do employees get?")["sources"]
    assert isinstance(leave, list) and leave[0]["source"] == "leave_policy.md", "older documents stay retrievable"


def test_reupload_replaces_chunks_without_duplicates(rag: App) -> None:
    first = rag.upload(POLICY_NAME, make_pdf(*POLICY_PAGES)).json()
    again = rag.upload(POLICY_NAME, make_pdf(*POLICY_PAGES)).json()
    assert again["chunks_indexed"] == first["chunks_indexed"] == rag.store.count()

    rag.upload(POLICY_NAME, make_pdf("Revised policy: core hours are now 11:00 to 14:00."))
    rag.ask("What are the core hours?")
    prompt = rag.llm.prompt_text()
    assert "11:00 to 14:00" in prompt and "10:00 to 15:00" not in prompt
    assert rag.saved_files() == [POLICY_NAME]


def test_failed_reupload_keeps_the_previous_copy(rag: App) -> None:
    good = make_pdf(*POLICY_PAGES)
    count = rag.upload(POLICY_NAME, good).json()["chunks_indexed"]
    assert rag.upload(POLICY_NAME, make_pdf("")).status_code == 400
    assert (rag.settings.documents_dir / POLICY_NAME).read_bytes() == good
    assert rag.store.count() == count


@pytest.mark.parametrize("other_name", ["Work Engagement Policy (1).pdf", "work engagement policy.pdf"])
def test_same_file_under_another_name_is_rejected(rag: App, other_name: str) -> None:
    rag.upload(POLICY_NAME, make_pdf(*POLICY_PAGES))
    count = rag.store.count()
    response = rag.upload(other_name, make_pdf(*POLICY_PAGES))
    assert response.status_code == 400 and POLICY_NAME in response.json()["detail"]
    assert rag.store.count() == count and rag.saved_files() == [POLICY_NAME]


# ---------------------------------------------------------------- file names and concurrency
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (POLICY_NAME, POLICY_NAME),
        ("../../escape.md", "escape.md"),
        ("..\\..\\windows\\escape.md", "escape.md"),
        ("C:\\Users\\me\\notes.txt", "notes.txt"),
        ("we<i>rd|na*me?.md", "we_i_rd_na_me_.md"),
        ("  spaced   out  .txt", "spaced out .txt"),
        ("con.md", "_con.md"),
        ("NUL.tar.txt", "_NUL.tar.txt"),
        ("../..", ""),
        ("", ""),
    ],
)
def test_safe_filename(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


def test_safe_filename_caps_length_and_keeps_suffix() -> None:
    name = safe_filename("x" * 300 + ".pdf")
    assert len(name) == 120 and name.endswith(".pdf")


def test_concurrent_uploads_and_searches_keep_the_index_consistent(settings: Settings) -> None:
    store = get_vector_store(settings.vector_dir)
    parts = Components(
        engine=get_engine(llm=FakeChatModel(), retriever=None, tools=[]),
        chunker=get_chunker("fixed", chunk_size=200, chunk_overlap=20),
        embedder=HashEmbedder(),
        store=store,
    )
    retriever = VectorRetriever(store, HashEmbedder(), k=4)
    files = {f"doc{i}.md": f"Document {i}. ".encode() + b"Topic words repeat here. " * 40 for i in range(8)}

    def upload(name: str) -> int:
        return save_and_index_upload(settings, parts, name, files[name]).chunks_indexed

    with ThreadPoolExecutor(max_workers=8) as pool:
        searches = [pool.submit(retriever.retrieve, "topic words") for _ in range(16)]
        counts = list(pool.map(upload, files))
        for search in searches:
            search.result()
    assert store.count() == sum(counts)
    assert get_vector_store(settings.vector_dir).count() == sum(counts), "what is on disk matches memory"
    assert sorted(p.name for p in settings.documents_dir.iterdir()) == sorted(files), "no .part files left"


def test_upload_error_is_a_value_error() -> None:
    assert issubclass(UploadError, ValueError)
