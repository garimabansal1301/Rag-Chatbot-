"""CSV datasets, phase 1: upload, validation, preview and per-session storage. Temp folders, no LLM calls."""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.chat.engine import get_engine
from app.config import PROJECT_ROOT, Settings, load_settings
from app.datasets import parsing
from app.datasets.parsing import DatasetError, is_formula_like, neutralize_formula, parse_csv
from app.datasets.store import DATA_FILE, DatasetNotFoundError, DatasetStore
from app.deps import Components, get_components
from app.main import create_app
from tests.fakes import FakeChatModel


def _sales_csv() -> bytes:
    """25 data rows; `units` is missing in 4 of them (rows 4, 11, 18, 25)."""
    regions_months = [(r, m) for r in ("North", "South") for m in range(1, 13)] + [("East", 1)]
    lines = ["region,month,revenue,units,launched,active"]
    for i, (region, month) in enumerate(regions_months):
        units = "" if i % 7 == 3 else str(10 + i)
        active = "true" if i % 2 else "false"
        lines.append(f"{region},2024-{month:02d}-01,{1000 + i * 12.5},{units},2024-{month:02d}-15,{active}")
    return ("\n".join(lines) + "\n").encode()


SALES = _sales_csv()


def upload_body(name: str, data: bytes, mime: str = "text/csv") -> dict[str, Any]:
    return {"files": {"file": (name, data, mime)}}


@pytest.fixture()
def app_client(settings: Settings) -> Iterator[tuple[TestClient, Settings, FakeChatModel]]:
    llm = FakeChatModel(reply="ANSWER")
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=get_engine(llm=llm, retriever=None, tools=[]))
    with TestClient(app) as client:
        yield client, settings, llm


def new_session(client: TestClient) -> str:
    session_id: str = client.post("/api/sessions", json={}).json()["id"]
    return session_id


# ---------------------------------------------------------------- parsing: valid files
def test_valid_csv_metadata_types_missing_and_preview() -> None:
    parsed = parse_csv(SALES)
    assert parsed.rows == 25 and parsed.encoding == "UTF-8" and parsed.delimiter == ","
    assert [(c.name, c.dtype) for c in parsed.columns] == [
        ("region", "text"),
        ("month", "date"),
        ("revenue", "decimal"),
        ("units", "integer"),
        ("launched", "date"),
        ("active", "boolean"),
    ]
    assert {c.name: c.missing for c in parsed.columns}["units"] == 4
    assert len(parsed.preview) == 10 and parsed.preview[0][:3] == ["North", "2024-01-01", "1000.0"]
    assert parsed.preview[3][3] is None, "missing cells are None in the preview"


@pytest.mark.parametrize(
    ("data", "delimiter", "encoding"),
    [
        (b"a;b\n1;x\n2;y\n", ";", "UTF-8"),
        (b"a\tb\n1\tx\n2\ty\n", "\t", "UTF-8"),
        (b"a|b\n1|x\n2|y\n", "|", "UTF-8"),
        (codecs.BOM_UTF8 + b"a,b\n1,x\n2,y\n", ",", "UTF-8"),
        ("a,b\n1,x\n2,y\n".encode("utf-16"), ",", "UTF-16"),
        (b"a,b\r\n1,x\r\n2,y\r\n", ",", "UTF-8"),
    ],
)
def test_delimiters_and_encodings(data: bytes, delimiter: str, encoding: str) -> None:
    parsed = parse_csv(data)
    assert (parsed.delimiter, parsed.encoding, parsed.rows) == (delimiter, encoding, 2)
    assert [c.name for c in parsed.columns] == ["a", "b"], "a UTF-8 BOM never ends up in the first column name"


def test_duplicate_and_blank_column_names_are_made_unique() -> None:
    parsed = parse_csv(b"Sales,sales, ,Sales\n1,2,3,4\n")
    assert [c.name for c in parsed.columns] == ["Sales", "sales_2", "column_3", "Sales_3"]
    assert any("Duplicate column names" in w for w in parsed.warnings)
    assert any("column_N" in w for w in parsed.warnings)


@pytest.mark.parametrize("marker", ["", "NA", "N/A", "null", "NaN", "None", "#N/A"])
def test_missing_value_markers(marker: str) -> None:
    parsed = parse_csv(f"x,y\n1,{marker}\n2,5\n".encode())
    assert parsed.columns[1].missing == 1 and parsed.preview[0][1] is None


def test_short_rows_and_empty_cells_count_as_missing() -> None:
    parsed = parse_csv(b"a,b,c\n1,2\n,,\n4,5,6\n")
    assert parsed.rows == 3 and [c.missing for c in parsed.columns] == [1, 1, 2]
    assert parsed.preview[0] == ["1", "2", None]


def test_quoted_commas_and_newlines() -> None:
    parsed = parse_csv(b'name,note\n"Smith, J","line one\nline two"\n')
    assert parsed.rows == 1 and parsed.preview[0] == ["Smith, J", "line one\nline two"]


def test_all_missing_column_is_empty() -> None:
    assert parse_csv(b"a,b\n1,\n2,\n").columns[1].dtype == "empty"


# ---------------------------------------------------------------- parsing: rejected files
@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "The file is empty."),
        (b"   \n\n  ", "The file is empty."),
        (b"a,b\n", "header row but no data rows"),
        (b"a,b\n\x00\x01\x02", "binary file"),
        (b"PK\x03\x04\x00\x00", "binary file"),
        ("caf\xe9,b\n1,2\n".encode("latin-1"), "not UTF-8"),
        (b"a,b\n1,2\n3,4,5\n", "line 3 has 3 values, but the header has 2 columns"),
        (b'a,b\n1,"never closed\n2,3\n', "Malformed CSV"),
    ],
)
def test_malformed_and_empty_files_are_rejected(data: bytes, message: str) -> None:
    with pytest.raises(DatasetError, match=re.escape(message)):
        parse_csv(data)


def test_row_and_column_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parsing, "MAX_ROWS", 3)
    monkeypatch.setattr(parsing, "MAX_COLUMNS", 2)
    assert parse_csv(b"a,b\n1,2\n3,4\n5,6\n").rows == 3
    with pytest.raises(DatasetError, match="Too many rows"):
        parse_csv(b"a,b\n1,2\n3,4\n5,6\n7,8\n")
    with pytest.raises(DatasetError, match="Too many columns: 3"):
        parse_csv(b"a,b,c\n1,2,3\n")


# ---------------------------------------------------------------- formula injection
@pytest.mark.parametrize(
    ("value", "neutralized"),
    [
        ("=1+1", "'=1+1"),
        ("+SUM(A1:A2)", "'+SUM(A1:A2)"),
        ("-2+3", "'-2+3"),
        ("@cmd", "'@cmd"),
        ("\t=1", "'\t=1"),
        ("\r=1", "'\r=1"),
        ("-5", "-5"),
        ("+3.25", "+3.25"),
        ("-1,5", "-1,5"),
        ("1e-3", "1e-3"),
        ("hello", "hello"),
        ("", ""),
    ],
)
def test_neutralize_formula(value: str, neutralized: str) -> None:
    assert neutralize_formula(value) == neutralized
    assert is_formula_like(value) == (value != neutralized)


def test_formula_like_cells_are_counted_and_kept_as_text() -> None:
    parsed = parse_csv(b'item,note\nx,"=HYPERLINK(""http://evil.example"",""x"")"\ny,-5\nz,@SUM(A1)\n')
    assert parsed.formula_like_cells == 2 and parsed.columns[1].dtype == "text"
    assert parsed.preview[0][1] == '=HYPERLINK("http://evil.example","x")', "kept as written, never evaluated"
    assert any("neutralised if the data is ever exported" in w for w in parsed.warnings)


# ---------------------------------------------------------------- store: ids, ownership, isolation
SESSION_A, SESSION_B = "a" * 32, "b" * 32


def test_store_saves_under_internal_ids(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path)
    info = store.save(SESSION_A, "../../evil name.csv", parse_csv(b"a,b\n1,2\n"), 8)
    assert re.fullmatch(r"[0-9a-f]{32}", info.id) and info.session_id == SESSION_A
    folder = tmp_path / SESSION_A / info.id
    assert sorted(p.name for p in folder.iterdir()) == [DATA_FILE, "meta.json"]
    assert store.data_path(SESSION_A, info.id).read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert store.get(SESSION_A, info.id) == info and store.list(SESSION_A) == [info]
    assert info.id != store.save(SESSION_A, "x.csv", parse_csv(b"a\n1\n"), 4).id


def test_dataset_is_not_reachable_through_another_session(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path)
    info = store.save(SESSION_A, "a.csv", parse_csv(b"a\n1\n"), 4)
    with pytest.raises(DatasetNotFoundError):
        store.get(SESSION_B, info.id)
    with pytest.raises(DatasetNotFoundError):
        store.data_path(SESSION_B, info.id)
    assert store.list(SESSION_B) == []
    # Even a copied folder is refused: the metadata names its owner.
    (tmp_path / SESSION_B).mkdir()
    (tmp_path / SESSION_A / info.id).rename(tmp_path / SESSION_B / info.id)
    with pytest.raises(DatasetNotFoundError):
        store.get(SESSION_B, info.id)


@pytest.mark.parametrize(
    ("session_id", "dataset_id"), [("../" + "a" * 29, "c" * 32), ("A" * 32, "c" * 32), ("a" * 32, "..\\x"), ("", "")]
)
def test_invalid_ids_are_refused(tmp_path: Path, session_id: str, dataset_id: str) -> None:
    with pytest.raises(DatasetNotFoundError):
        DatasetStore(tmp_path).get(session_id, dataset_id)


def test_per_session_limit_and_delete_session(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path, max_per_session=2)
    for name in ("one.csv", "two.csv"):
        store.save(SESSION_A, name, parse_csv(b"a\n1\n"), 4)
    with pytest.raises(DatasetError, match="already has 2 datasets"):
        store.save(SESSION_A, "three.csv", parse_csv(b"a\n1\n"), 4)
    kept = store.save(SESSION_B, "b.csv", parse_csv(b"a\n1\n"), 4)
    store.delete_session(SESSION_A)
    assert store.list(SESSION_A) == [] and store.list(SESSION_B) == [kept]
    assert not (tmp_path / SESSION_A).exists()


# ---------------------------------------------------------------- UI upload and preview
def test_ui_upload_shows_the_preview(app_client: tuple[TestClient, Settings, FakeChatModel]) -> None:
    client, _, _ = app_client
    session_id = new_session(client)
    html = client.post(f"/ui/sessions/{session_id}/datasets", **upload_body("Sales 2024.csv", SALES)).text
    assert 'class="dataset-status ok"' in html and "Sales 2024.csv: 25 rows × 6 columns uploaded and checked." in html
    assert "25 rows × 6 columns · UTF-8" in html and "First 10 of 25 rows" in html
    for name, dtype in [("region", "text"), ("month", "date"), ("revenue", "decimal"), ("active", "boolean")]:
        assert f'<th scope="col">{name}<span class="dtype">{dtype}' in html
    assert 'units<span class="dtype">integer · 4 missing' in html
    assert html.count("<tr>") == 1 + 10, "header row + exactly 10 preview rows"
    assert '<td class="missing" title="Missing value">—</td>' in html
    assert re.search(r'<details class="dataset( active)?" open>', html), "the new dataset is expanded"


@pytest.mark.parametrize(
    ("name", "data", "mime", "message"),
    [
        ("notes.txt", b"a,b\n1,2\n", "text/plain", "Upload a .csv file"),
        ("data.csv.exe", b"a,b\n1,2\n", "text/csv", "Upload a .csv file"),
        ("empty.csv", b"", "text/csv", "The file is empty."),
        ("fake.csv", b"\x89PNG\r\n\x1a\n\x00\x00", "text/csv", "binary file"),  # MIME type is not trusted
        ("bad.csv", b"a,b\n1,2,3\n", "text/csv", "line 2 has 3 values"),
        ("latin.csv", "caf\xe9,b\n1,2\n".encode("latin-1"), "text/csv", "not UTF-8"),
    ],
)
def test_ui_upload_errors_are_shown_and_nothing_is_saved(
    app_client: tuple[TestClient, Settings, FakeChatModel], name: str, data: bytes, mime: str, message: str
) -> None:
    client, settings, _ = app_client
    session_id = new_session(client)
    response = client.post(f"/ui/sessions/{session_id}/datasets", **upload_body(name, data, mime))
    assert response.status_code == 200 and 'class="dataset-status error"' in response.text
    assert message in response.text
    assert client.get(f"/api/sessions/{session_id}/datasets").json() == []
    assert not (settings.dataset_root / session_id).exists(), "a rejected upload writes nothing"


def test_uppercase_extension_and_generic_mime_are_accepted(
    app_client: tuple[TestClient, Settings, FakeChatModel],
) -> None:
    client, _, _ = app_client
    session_id = new_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/datasets", **upload_body("DATA.CSV", b"a,b\n1,2\n", "application/octet-stream")
    )
    assert response.status_code == 201 and response.json()["filename"] == "DATA.CSV"


def test_size_limit(settings: Settings) -> None:
    app = create_app(replace(settings, max_dataset_bytes=20))
    app.dependency_overrides[get_components] = lambda: Components(engine=get_engine(llm=FakeChatModel(), tools=[]))
    with TestClient(app) as client:
        session_id = new_session(client)
        too_big = client.post(f"/api/sessions/{session_id}/datasets", **upload_body("big.csv", b"a,b\n" + b"1,2\n" * 5))
        assert too_big.status_code == 400 and "too large (limit" in too_big.json()["detail"]
        assert (
            client.post(f"/api/sessions/{session_id}/datasets", **upload_body("ok.csv", b"a,b\n1,2\n")).status_code
            == 201
        )


def test_dataset_size_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_settings().max_dataset_bytes == 10 * 1024 * 1024
    monkeypatch.setenv("MAX_DATASET_MB", "2")
    assert load_settings().max_dataset_bytes == 2 * 1024 * 1024


def test_untrusted_cells_and_names_are_escaped(app_client: tuple[TestClient, Settings, FakeChatModel]) -> None:
    client, _, _ = app_client
    session_id = new_session(client)
    data = b"<b>col</b>,note\n<script>alert(1)</script>,\"=cmd|' /C calc'!A0\"\n"
    html = client.post(f"/ui/sessions/{session_id}/datasets", **upload_body("x.csv", data)).text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;col&lt;/b&gt;" in html and "neutralised if the data is ever exported" in html


# ---------------------------------------------------------------- JSON API and session isolation
def test_api_upload_list_and_get(app_client: tuple[TestClient, Settings, FakeChatModel]) -> None:
    client, _, _ = app_client
    session_id = new_session(client)
    created = client.post(f"/api/sessions/{session_id}/datasets", **upload_body("sales.csv", SALES))
    assert created.status_code == 201
    info = created.json()
    assert info["rows"] == 25 and len(info["columns"]) == 6 and len(info["preview"]) == 10
    assert client.get(f"/api/sessions/{session_id}/datasets").json() == [info]
    assert client.get(f"/api/sessions/{session_id}/datasets/{info['id']}").json() == info


def test_other_session_cannot_reach_the_dataset(app_client: tuple[TestClient, Settings, FakeChatModel]) -> None:
    client, _, _ = app_client
    owner, other = new_session(client), new_session(client)
    info = client.post(f"/api/sessions/{owner}/datasets", **upload_body("a.csv", b"a\n1\n")).json()
    assert client.get(f"/api/sessions/{other}/datasets/{info['id']}").status_code == 404
    assert client.get(f"/api/sessions/{other}/datasets").json() == []
    assert info["id"] not in client.get(f"/sessions/{other}").text


@pytest.mark.parametrize("method", ["get_list", "post", "get_one"])
def test_unknown_session_is_404(app_client: tuple[TestClient, Settings, FakeChatModel], method: str) -> None:
    client, _, _ = app_client
    unknown = "f" * 32
    if method == "get_list":
        response = client.get(f"/api/sessions/{unknown}/datasets")
    elif method == "post":
        response = client.post(f"/api/sessions/{unknown}/datasets", **upload_body("a.csv", b"a\n1\n"))
    else:
        response = client.get(f"/api/sessions/{unknown}/datasets/{'c' * 32}")
    assert response.status_code == 404
    assert client.post(f"/ui/sessions/{unknown}/datasets", **upload_body("a.csv", b"a\n1\n")).status_code == 404


@pytest.mark.parametrize("via", ["api", "ui"])
def test_deleting_a_session_removes_its_datasets(
    app_client: tuple[TestClient, Settings, FakeChatModel], via: str
) -> None:
    client, settings, _ = app_client
    doomed, kept = new_session(client), new_session(client)
    for session_id in (doomed, kept):
        client.post(f"/api/sessions/{session_id}/datasets", **upload_body("a.csv", b"a\n1\n"))
    if via == "api":
        assert client.delete(f"/api/sessions/{doomed}").status_code == 204
    else:
        client.post(f"/ui/sessions/{doomed}/delete")
    assert not (settings.dataset_root / doomed).exists()
    assert len(client.get(f"/api/sessions/{kept}/datasets").json()) == 1


# ---------------------------------------------------------------- page, isolation of storage, nothing sent to the LLM
def test_session_page_lists_saved_datasets(app_client: tuple[TestClient, Settings, FakeChatModel]) -> None:
    client, _, _ = app_client
    session_id = new_session(client)
    assert 'id="datasets"' in client.get(f"/sessions/{session_id}").text
    client.post(f"/api/sessions/{session_id}/datasets", **upload_body("sales.csv", SALES))
    page = client.get(f"/sessions/{session_id}").text
    assert 'Datasets <span class="muted">(1)</span>' in page and "sales.csv" in page and "First 10 of 25 rows" in page


def test_csv_content_is_not_sent_to_the_llm(app_client: tuple[TestClient, Settings, FakeChatModel]) -> None:
    client, _, llm = app_client
    session_id = new_session(client)
    client.post(f"/api/sessions/{session_id}/datasets", **upload_body("s.csv", b"code,value\nZQX-7731,42\n"))
    client.post(f"/api/sessions/{session_id}/messages", json={"content": "What is in my dataset?"})
    assert llm.calls and "ZQX-7731" not in llm.prompt_text() and "s.csv" not in llm.prompt_text()


def test_tests_never_write_to_the_real_data_folder(settings: Settings) -> None:
    real = PROJECT_ROOT / "data"
    for root in (settings.dataset_root, load_settings().dataset_root):  # fixture and DATASETS_DIR (conftest)
        assert root != real / "datasets" and real not in root.parents


def test_saved_metadata_is_json_without_the_full_file(app_client: tuple[TestClient, Settings, FakeChatModel]) -> None:
    client, settings, _ = app_client
    session_id = new_session(client)
    info = client.post(f"/api/sessions/{session_id}/datasets", **upload_body("sales.csv", SALES)).json()
    meta = json.loads((settings.dataset_root / session_id / info["id"] / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["preview"]) == 10 and meta["rows"] == 25 and "text" not in meta
