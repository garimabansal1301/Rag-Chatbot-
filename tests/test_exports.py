"""Phase 4: Excel exports of datasets and saved analysis results. Temp folders, fake LLMs, no network."""

from __future__ import annotations

import functools
import io
import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, LineChart, PieChart

import app.api.exports as exports_api
from app.chat.engine import get_engine
from app.config import Settings
from app.datasets import excel
from app.datasets.analysis import Query, load_frame, run_query
from app.datasets.excel import (
    XLSX_MEDIA_TYPE,
    ExportError,
    analysis_workbook,
    dataset_workbook,
    download_filename,
    excel_text,
    from_text,
    infer_kinds,
    safe_sheet_name,
    unique_headers,
)
from app.datasets.parsing import is_formula_like, neutralize_formula, parse_csv
from app.datasets.store import DatasetStore
from app.deps import Components, get_components
from app.main import create_app
from app.models import AnalysisTable, Message
from app.storage.json_store import SessionStore
from app.tools import get_tools
from app.tools.dataset_tools import dataset_context, make_dataset_tools
from tests.fakes import TOOL_TRIGGER, FakeChatModel, StaticRetriever

# Typed columns, missing values, a duplicate row, and hostile text cells (formula-like with and without
# leading whitespace/control characters) next to ordinary negative numbers.
ORDERS = (
    b"order_id,order_date,product,category,sales,units,returned,note\n"
    b"1,2024-01-05,Laptop,Electronics,1200.5,2,false,ok\n"
    b'2,2024-02-10,Chair,Furniture,-35.25,1,true,"=HYPERLINK(""http://evil.example"",""x"")"\n'
    b"3,2024-02-11,Pen,Office,,10,false,  =1+1\n"
    b"4,,Desk,Furniture,449,,false,@SUM(A1)\n"
    b"1,2024-01-05,Laptop,Electronics,1200.5,2,false,ok\n"
    b"5,2024-03-01,Phone,Electronics,800,-3,false,-5\n"
    b"6,2024-03-02,Lamp,Furniture,40,1,false,\"+cmd|' /C calc'!A0\"\n"
)
GROUPED_SALES = {"group_by": ["category"], "aggregations": [{"column": "sales", "function": "sum"}]}


def workbook(content: bytes) -> Workbook:
    return load_workbook(io.BytesIO(content))


def rows(sheet: Any) -> list[list[Any]]:
    return [[c.value for c in row] for row in sheet.iter_rows()]


@dataclass
class Saved:
    store: DatasetStore
    session_id: str
    info: Any
    frame: Any


@pytest.fixture()
def saved(tmp_path: Path) -> Saved:
    store = DatasetStore(tmp_path / "datasets")
    session_id = "a" * 32
    info = store.save(session_id, "orders.csv", parse_csv(ORDERS), len(ORDERS))
    return Saved(store, session_id, info, load_frame(store, session_id, info))


def result(saved: Saved, **spec: Any) -> AnalysisTable:
    return run_query(saved.frame, saved.info, Query.model_validate(spec))


# ---------------------------------------------------------------- formula injection
@pytest.mark.parametrize(
    ("value", "formula"),
    [
        ("=1+1", True),
        ("+SUM(A1)", True),
        ("-2+3", True),
        ("@cmd", True),
        ("  =1+1", True),
        (" =1+1", True),
        ("\x01=cmd", True),
        ("\n@x", True),
        ("\t", True),
        ("\rabc", True),
        ("-5", False),
        ("  -5", False),
        ("+3.25", False),
        ("-1e-3", False),
        ("-5abc", True),
        ("hello", False),
        ("a=b", False),
        ("", False),
    ],
)
def test_formula_detection(value: str, formula: bool) -> None:
    assert is_formula_like(value) is formula
    assert neutralize_formula(value) == ("'" + value if formula else value)


def test_excel_text_neutralises_strips_and_caps() -> None:
    assert excel_text("\x01=cmd") == "'=cmd", "control character removed after the check, apostrophe kept"
    assert excel_text("a\x00b\x07c") == "abc"
    assert excel_text("  =1+1") == "'  =1+1"
    assert len(excel_text("x" * 40_000)) == 32_767


def test_no_cell_in_any_export_is_a_formula(saved: Saved) -> None:
    table = result(saved, columns=["order_id", "sales", "note"])
    for content in (
        dataset_workbook(saved.info, saved.frame, summary=True, source=True, table=table),
        analysis_workbook(table),
    ):
        for sheet in workbook(content).worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    assert cell.data_type != "f", f"{sheet.title}!{cell.coordinate} became a formula"
                    if isinstance(cell.value, str):
                        assert not is_formula_like(cell.value), cell.value


def test_hostile_text_is_neutralised_and_negative_numbers_stay_numbers(saved: Saved) -> None:
    sheet = workbook(dataset_workbook(saved.info, saved.frame, summary=False, source=True))["Source Data"]
    data = rows(sheet)
    header = data[0]
    notes = [r[header.index("note")] for r in data[1:]]
    assert notes[1] == '\'=HYPERLINK("http://evil.example","x")'
    assert notes[2] == "'  =1+1" and notes[3] == "'@SUM(A1)" and notes[6] == "'+cmd|' /C calc'!A0"
    assert notes[5] == "-5", "a negative number written as text in a text column is left alone"
    sales = sheet.cell(row=3, column=header.index("sales") + 1)
    units = sheet.cell(row=7, column=header.index("units") + 1)
    assert sales.value == -35.25 and sales.data_type == "n", "a real negative number stays numeric"
    assert units.value == -3 and units.data_type == "n"


# ---------------------------------------------------------------- types, formats, sheets
def test_source_data_keeps_types_and_blanks(saved: Saved) -> None:
    book = workbook(dataset_workbook(saved.info, saved.frame, summary=False, source=True))
    assert book.sheetnames == ["Report", "Source Data"]
    sheet = book["Source Data"]
    header = [c.value for c in sheet[1]]
    assert header == ["order_id", "order_date", "product", "category", "sales", "units", "returned", "note"]
    assert all(c.font.b for c in sheet[1]) and sheet.freeze_panes == "A2" and sheet.auto_filter.ref == "A1:H8"
    first = {h: sheet.cell(row=2, column=i + 1) for i, h in enumerate(header)}
    assert first["order_id"].value == 1 and first["order_id"].number_format == "#,##0"
    assert first["order_date"].value == datetime(2024, 1, 5) and first["order_date"].number_format == "yyyy-mm-dd"
    assert first["sales"].value == 1200.5 and first["sales"].number_format == "#,##0.00"
    assert first["returned"].value is False and first["product"].value == "Laptop"
    desk = {h: sheet.cell(row=5, column=i + 1).value for i, h in enumerate(header)}
    assert desk["order_date"] is None and desk["units"] is None, "missing values are blank cells"
    whole = sheet.cell(row=5, column=header.index("sales") + 1)
    assert whole.value == 449 and whole.number_format == "#,##0.00", "one number format per decimal column"
    assert len(rows(sheet)) == 8


def test_report_has_summary_and_quality_sheets(saved: Saved) -> None:
    book = workbook(dataset_workbook(saved.info, saved.frame, summary=True))
    assert book.sheetnames == ["Report", "Dataset Summary", "Data Quality"]
    report = {r[0]: r[1] for r in rows(book["Report"]) if r and r[0]}
    assert report["Dataset"] == "orders.csv" and report["Rows"] == 7 and report["Columns"] == 8
    assert report["Duplicate rows"] == 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", report["Exported at"])
    summary = {r[0]: r for r in rows(book["Dataset Summary"])[1:]}
    assert summary["sales"][1:5] == ["decimal", 1, pytest.approx(1 / 7, abs=1e-6), 5]  # 1200.5 appears twice
    assert summary["sales"][5:] == [-35.25, pytest.approx((1200.5 - 35.25 + 449 + 1200.5 + 800 + 40) / 6), 1200.5]
    assert summary["order_date"][5] == datetime(2024, 1, 5) and summary["order_date"][6] is None
    pct = book["Dataset Summary"].cell(row=6, column=4)
    assert pct.number_format == "0.0%"
    quality = rows(book["Data Quality"])
    assert quality[0] == ["Column", "Missing values", "Missing %", "Observation"]
    facts = {r[0]: r[1] for r in quality if r and r[0] in ("Rows", "Duplicate rows", "Rows with no missing values")}
    assert facts == {"Rows": 7, "Duplicate rows": 1, "Rows with no missing values": 5}  # orders 3 and 4 have gaps


def test_report_with_source_and_saved_result(saved: Saved) -> None:
    table = result(saved, **GROUPED_SALES, chart={"type": "bar"})
    book = workbook(dataset_workbook(saved.info, saved.frame, summary=True, source=True, table=table))
    assert book.sheetnames == [
        "Report",
        "Dataset Summary",
        "Data Quality",
        "Grouped by category",
        "Chart",
        "Source Data",
    ]
    notes = [r[0] for r in rows(book["Report"]) if r and isinstance(r[0], str)]
    assert any("Source Data is the uploaded file as stored" in n for n in notes)
    assert any("missing value(s) in 'sales' were excluded" in n for n in notes)


def test_analysis_export_matches_the_saved_table(saved: Saved) -> None:
    table = result(
        saved, **GROUPED_SALES, sort_by=[{"column": "sum_sales", "descending": True}], limit=2, chart={"type": "pie"}
    )
    book = workbook(analysis_workbook(table))
    assert book.sheetnames == ["Report", "Grouped by category", "Chart"]
    data = rows(book["Grouped by category"])
    assert data[0] == table.columns == ["category", "records", "sum_sales"]
    assert data[1:] == [["Electronics", 3, 3201.0], ["Furniture", 3, 453.75]]
    sheet = book["Grouped by category"]
    assert (
        sheet.cell(row=2, column=2).number_format == "#,##0" and sheet.cell(row=2, column=3).number_format == "#,##0.00"
    )
    report = [r[0] for r in rows(book["Report"]) if r and isinstance(r[0], str)]
    assert any("had 3 rows; the 2 rows shown in the chat are exported" in n for n in report)


@pytest.mark.parametrize(("kind", "chart_class"), [("bar", BarChart), ("line", LineChart), ("pie", PieChart)])
def test_native_excel_charts(saved: Saved, kind: str, chart_class: type) -> None:
    spec = {"group_by_month": "order_date"} if kind == "line" else {"group_by": ["category"]}
    table = result(saved, **spec, aggregations=[{"column": "sales", "function": "sum"}], chart={"type": kind})
    sheet = workbook(analysis_workbook(table))["Chart"]
    assert len(sheet._charts) == 1 and isinstance(sheet._charts[0], chart_class)
    assert table.chart is not None
    data = rows(sheet)[1:]
    assert [r[0] for r in data] == table.chart.labels and [r[1] for r in data] == table.chart.values
    if kind == "pie":
        assert [round(r[2] * 100, 2) for r in data] == table.chart.percentages


def test_results_without_saved_types_are_inferred_safely() -> None:
    legacy = AnalysisTable(
        title="Old",
        dataset="x.csv",
        columns=["code", "count", "amount", "share", "day", "label"],
        rows=[["00123", "4", "1.5", "12.5%", "2024-01-02", "-5x"], ["00456", "6", "-2", "50%", None, "=1+1"]],
        total_rows=2,
    )
    assert infer_kinds(legacy) == ["text", "integer", "decimal", "percent", "date", "text"]
    data = rows(workbook(analysis_workbook(legacy))["Old"])
    assert data[1] == ["00123", 4, 1.5, 0.125, datetime(2024, 1, 2), "'-5x"]
    assert data[2] == ["00456", 6, -2.0, 0.5, None, "'=1+1"]


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        ("12801", "decimal", (12801.0, "#,##0.00")),
        ("12801", "integer", (12801, "#,##0")),
        ("4171.607143", "mixed", (4171.607143, "#,##0.00")),
        ("2024-01-05", "mixed", (date(2024, 1, 5), "yyyy-mm-dd")),
        ("12.5%", "percent", (0.125, "0.0%")),
        ("true", "boolean", (True, None)),
        ("2024-01", "text", ("2024-01", None)),
        ("inf", "decimal", ("inf", None)),
        (None, "decimal", (None, None)),
    ],
)
def test_saved_text_back_to_typed_cells(text: str | None, kind: str, expected: tuple[Any, Any]) -> None:
    assert from_text(text, kind) == expected


def test_empty_results_and_duplicate_headers() -> None:
    empty = AnalysisTable(title="Matching rows", dataset="x.csv", columns=["a", "A"], rows=[], total_rows=0)
    sheet = workbook(analysis_workbook(empty))["Matching rows"]
    assert rows(sheet) == [["a", "A (2)"]] and sheet.auto_filter.ref == "A1:B2"
    assert unique_headers(["x", "X", "x", ""]) == ["x", "X (2)", "x (3)", "column"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Grouped by category", "Grouped by category"),
        ("Sales [2024]: Q1/Q2?*\\", "Sales 2024 Q1 Q2"),
        ("x" * 40, "x" * 31),
        ("'quoted'", "quoted"),
        ("   ", "Sheet"),
        ("History", "History data"),
    ],
)
def test_safe_sheet_names(name: str, expected: str) -> None:
    assert safe_sheet_name(name, set()) == expected


def test_duplicate_sheet_names_stay_unique_and_short() -> None:
    used: set[str] = set()
    names = [safe_sheet_name("y" * 40, used) for _ in range(3)]
    assert names == ["y" * 31, "y" * 27 + " (2)", "y" * 27 + " (3)"] and all(len(n) <= 31 for n in names)


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("orders_sample", "report"), "orders_sample-report.xlsx"),
        (("../../etc/passwd", "data"), "etc_passwd-data.xlsx"),
        (("Ventes été", "Grouped by category"), "Ventes_t-Grouped_by_category.xlsx"),
        (("", ""), "export.xlsx"),
    ],
)
def test_download_filenames(parts: tuple[str, ...], expected: str) -> None:
    assert download_filename(*parts) == expected


def test_source_rows_are_bounded(saved: Saved, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(excel, "MAX_EXPORT_CELLS", 8 * 3)  # 8 columns -> 3 rows
    book = workbook(dataset_workbook(saved.info, saved.frame, summary=False, source=True))
    assert len(rows(book["Source Data"])) == 1 + 3
    notes = [r[0] for r in rows(book["Report"]) if r and isinstance(r[0], str)]
    assert any("first 3 of 7 rows (export size limit)" in n for n in notes)


def test_workbook_size_is_bounded(saved: Saved, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(excel, "MAX_WORKBOOK_BYTES", 1000)
    with pytest.raises(ExportError, match="too large"):
        dataset_workbook(saved.info, saved.frame)


# ---------------------------------------------------------------- routes: content, authorisation, errors
@pytest.fixture()
def app_client(settings: Settings) -> Iterator[tuple[TestClient, DatasetStore, SessionStore]]:
    app = create_app(settings)
    datasets: DatasetStore = app.state.dataset_store
    sessions: SessionStore = app.state.session_store
    llm = FakeChatModel(
        reply="ANSWER", tool_name="query_dataset", tool_args={**GROUPED_SALES, "chart": {"type": "bar"}}
    )
    engine = get_engine(
        llm=llm,
        retriever=StaticRetriever([]),
        tools=[*get_tools(), *make_dataset_tools(datasets)],
        session_context=functools.partial(dataset_context, datasets),
    )
    app.dependency_overrides[get_components] = lambda: Components(engine=engine)
    with TestClient(app) as client:
        yield client, datasets, sessions


def new_session(client: TestClient) -> str:
    session_id: str = client.post("/api/sessions", json={}).json()["id"]
    return session_id


def upload(client: TestClient, session_id: str, name: str = "orders.csv", data: bytes = ORDERS) -> str:
    dataset_id: str = client.post(
        f"/api/sessions/{session_id}/datasets", files={"file": (name, data, "text/csv")}
    ).json()["id"]
    return dataset_id


def ask_for_analysis(client: TestClient, session_id: str) -> str:
    return client.post(
        f"/ui/sessions/{session_id}/messages", data={"content": f"{TOOL_TRIGGER} sales by category"}
    ).text


def assert_xlsx(response: Any, filename: str) -> Workbook:
    assert response.status_code == 200 and response.headers["content-type"] == XLSX_MEDIA_TYPE
    assert response.headers["content-disposition"] == f'attachment; filename="{filename}"'
    assert response.headers["cache-control"] == "no-store" and response.headers["x-content-type-options"] == "nosniff"
    return workbook(response.content)


def test_dataset_data_and_report_downloads(app_client: tuple[TestClient, DatasetStore, SessionStore]) -> None:
    client, _, _ = app_client
    session_id = new_session(client)
    dataset_id = upload(client, session_id, "Sales 2024.csv")
    base = f"/api/sessions/{session_id}/datasets/{dataset_id}"
    data = assert_xlsx(client.get(f"{base}/data.xlsx"), "Sales_2024-data.xlsx")
    assert data.sheetnames == ["Report", "Source Data"]
    report = assert_xlsx(client.get(f"{base}/report.xlsx"), "Sales_2024-report.xlsx")
    assert report.sheetnames == ["Report", "Dataset Summary", "Data Quality"]
    with_source = workbook(client.get(f"{base}/report.xlsx?source=true").content)
    assert with_source.sheetnames[-1] == "Source Data"


def test_saved_result_export_via_the_link_in_the_chat(
    app_client: tuple[TestClient, DatasetStore, SessionStore],
) -> None:
    client, _, sessions = app_client
    session_id = new_session(client)
    dataset_id = upload(client, session_id)
    html = ask_for_analysis(client, session_id)
    links = re.findall(r'href="(/api/sessions/[^"]+/export\.xlsx)"', html)
    assert links == [f"/api/sessions/{session_id}/messages/1/analysis/0/export.xlsx"]
    book = assert_xlsx(client.get(links[0]), "orders-Grouped_by_category.xlsx")
    assert book.sheetnames == ["Report", "Grouped by category", "Chart"]
    saved_rows = sessions.get(session_id).messages[1].analysis[0].rows
    exported = rows(book["Grouped by category"])[1:]
    assert [r[0] for r in exported] == [r[0] for r in saved_rows]
    assert [(r[1], float(r[2])) for r in exported] == [(int(r[1] or 0), float(r[2] or 0)) for r in saved_rows]
    reloaded = client.get(f"/sessions/{session_id}").text
    assert links[0] in reloaded, "the result stays exportable after a reload"
    report = client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}/report.xlsx?message=1&table=0")
    assert "Grouped by category" in workbook(report.content).sheetnames


def test_follow_up_turn_links_point_at_the_right_message(
    app_client: tuple[TestClient, DatasetStore, SessionStore],
) -> None:
    client, _, _ = app_client
    session_id = new_session(client)
    upload(client, session_id)
    ask_for_analysis(client, session_id)
    second = ask_for_analysis(client, session_id)
    assert f"/api/sessions/{session_id}/messages/3/analysis/0/export.xlsx" in second
    assert client.get(f"/api/sessions/{session_id}/messages/3/analysis/0/export.xlsx").status_code == 200


def test_cross_session_and_invalid_ids_are_not_found(app_client: tuple[TestClient, DatasetStore, SessionStore]) -> None:
    client, _, _ = app_client
    owner, other = new_session(client), new_session(client)
    dataset_id = upload(client, owner)
    ask_for_analysis(client, owner)
    urls = [
        f"/api/sessions/{other}/datasets/{dataset_id}/data.xlsx",
        f"/api/sessions/{other}/datasets/{dataset_id}/report.xlsx",
        f"/api/sessions/{other}/messages/1/analysis/0/export.xlsx",
        f"/api/sessions/{owner}/datasets/{'c' * 32}/data.xlsx",
        f"/api/sessions/{'f' * 32}/datasets/{dataset_id}/data.xlsx",
        f"/api/sessions/{owner}/datasets/..%2F..%2F{other}/data.xlsx",
        f"/api/sessions/{owner}/datasets/{dataset_id.upper()}/data.xlsx",
        f"/api/sessions/{owner}/messages/0/analysis/0/export.xlsx",  # a user message
        f"/api/sessions/{owner}/messages/1/analysis/5/export.xlsx",
        f"/api/sessions/{owner}/messages/99/analysis/0/export.xlsx",
        f"/api/sessions/{owner}/messages/-1/analysis/0/export.xlsx",
    ]
    for url in urls:
        response = client.get(url)
        # "Not Found" (capital F) is FastAPI's own answer when a decoded "../" path matches no route at all.
        assert response.status_code == 404 and response.json()["detail"] in ("Not found", "Not Found"), url
        assert response.headers["content-type"].startswith("application/json"), url
    assert client.get(f"/api/sessions/{owner}/datasets/{dataset_id}/report.xlsx?message=0").status_code == 404
    traversal = client.get(f"/api/sessions/{owner}/datasets/{dataset_id}/data.xlsx?path=../../.env")
    assert traversal.status_code == 200 and "Source Data" in workbook(traversal.content).sheetnames, (
        "extra params ignored"
    )


def test_result_from_another_dataset_is_refused(app_client: tuple[TestClient, DatasetStore, SessionStore]) -> None:
    client, _, _ = app_client
    session_id = new_session(client)
    upload(client, session_id, "first.csv")
    client.post(f"/ui/sessions/{session_id}/datasets/{upload(client, session_id, 'second.csv')}/select")
    ask_for_analysis(client, session_id)
    first = [d["id"] for d in client.get(f"/api/sessions/{session_id}/datasets").json() if d["filename"] == "first.csv"]
    response = client.get(f"/api/sessions/{session_id}/datasets/{first[0]}/report.xlsx?message=1&table=0")
    assert response.status_code == 400 and "different dataset" in response.json()["detail"]


def test_exports_contain_nothing_from_outside_the_session(
    app_client: tuple[TestClient, DatasetStore, SessionStore], settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _ = app_client
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-key-123")
    owner, other = new_session(client), new_session(client)
    upload(client, other, "private.csv", b"secret_column,value\nTOPSECRET-77,1\n")
    dataset_id = upload(client, owner)
    content = client.get(f"/api/sessions/{owner}/datasets/{dataset_id}/report.xlsx?source=true").content
    text = " ".join(str(c.value) for s in workbook(content).worksheets for r in s.iter_rows() for c in r if c.value)
    for forbidden in ("secret-key-123", "GOOGLE_API_KEY", "TOPSECRET", "private.csv", str(settings.data_dir), other):
        assert forbidden not in text


def test_export_failures_are_safe(
    app_client: tuple[TestClient, DatasetStore, SessionStore], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, datasets, _ = app_client
    session_id = new_session(client)
    dataset_id = upload(client, session_id)

    def explode(*args: Any, **kwargs: Any) -> bytes:
        raise RuntimeError(f"boom at {datasets.root}")

    monkeypatch.setattr(exports_api, "dataset_workbook", explode)
    response = client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}/report.xlsx")
    assert response.status_code == 500 and response.json() == {"detail": "The workbook could not be created."}
    monkeypatch.setattr(
        exports_api, "dataset_workbook", lambda *a, **k: (_ for _ in ()).throw(ExportError("too large"))
    )
    assert client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}/data.xlsx").json() == {"detail": "too large"}
    (datasets.root / session_id / dataset_id / "data.csv").unlink()
    monkeypatch.undo()
    missing = client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}/data.xlsx")
    assert missing.status_code == 409 and str(datasets.root) not in missing.text


def test_exports_do_not_change_the_stored_files(app_client: tuple[TestClient, DatasetStore, SessionStore]) -> None:
    client, datasets, _ = app_client
    session_id = new_session(client)
    dataset_id = upload(client, session_id)
    folder = datasets.root / session_id / dataset_id
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    for suffix in ("data.xlsx", "report.xlsx", "report.xlsx?source=true"):
        client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}/{suffix}")
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


# ---------------------------------------------------------------- UI: links only where they belong
def test_export_links_only_for_saved_results(settings: Settings) -> None:
    sessions = SessionStore(settings.sessions_dir)
    session = sessions.create()
    table = AnalysisTable(title="Totals", dataset="x.csv", columns=["records"], rows=[["3"]], total_rows=1)
    sessions.append_messages(
        session.id,
        Message(role="user", content="hi"),
        Message(role="assistant", content="plain answer"),
        Message(role="assistant", content="with table", analysis=[table]),
    )
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=get_engine(llm=FakeChatModel(), tools=[]))
    with TestClient(app) as client:
        page = client.get(f"/sessions/{session.id}").text
    assert page.count('class="export-link"') == 1
    assert f"/api/sessions/{session.id}/messages/2/analysis/0/export.xlsx" in page
    assert "plain answer" in page and "with table" in page
    assert '<script src="http://testserver/static/exports.js" defer></script>' in page


def test_dataset_panel_offers_report_downloads(app_client: tuple[TestClient, DatasetStore, SessionStore]) -> None:
    client, _, _ = app_client
    session_id = new_session(client)
    assert "report.xlsx" not in client.get(f"/sessions/{session_id}").text
    dataset_id = upload(client, session_id)
    page = client.get(f"/sessions/{session_id}").text
    base = f"/api/sessions/{session_id}/datasets/{dataset_id}"
    for href in (f"{base}/report.xlsx", f"{base}/report.xlsx?source=true", f"{base}/data.xlsx"):
        assert f'href="{href}"' in page


# ---------------------------------------------------------------- static/exports.js in Node (fake DOM and fetch)
EXPORTS_JS = Path(__file__).resolve().parent.parent / "static" / "exports.js"
HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const scenario = JSON.parse(process.argv[1]);
const listeners = {}, clicked = [], revoked = [];
const status = { textContent: "", classes: new Set(),
  classList: { toggle: (c, on) => (on ? status.classes.add(c) : status.classes.delete(c)) } };
const answer = { textContent: "The analysis answer", removed: false };
const link = { href: "/api/sessions/x/export.xlsx", attrs: {}, parentElement: { querySelector: () => status },
  getAttribute(n) { return this.attrs[n] ?? null; }, setAttribute(n, v) { this.attrs[n] = v; },
  removeAttribute(n) { delete this.attrs[n]; }, closest(sel) { return sel === "a.export-link" ? this : null; } };
globalThis.document = {
  addEventListener: (type, fn) => { listeners[type] = fn; },
  createElement: () => ({ click() { clicked.push({ href: this.href, download: this.download }); }, remove() {} }),
  body: { appendChild: () => {} },
};
globalThis.URL = { createObjectURL: () => "blob:1", revokeObjectURL: (u) => revoked.push(u) };
globalThis.setTimeout = (fn) => fn();
globalThis.fetch = async () => {
  if (scenario.mode === "network") throw new Error("offline");
  return { ok: scenario.mode === "ok", status: scenario.mode === "ok" ? 200 : scenario.status,
    headers: { get: () => 'attachment; filename="orders-report.xlsx"' },
    blob: async () => "BLOB", json: async () => scenario.body };
};
vm.runInThisContext(fs.readFileSync(scenario.file, "utf8"));
let prevented = false;
listeners.click({ target: link, button: 0, defaultPrevented: false, preventDefault: () => { prevented = true; } });
setImmediate(() => setImmediate(() => console.log(JSON.stringify({
  prevented, clicked, revoked, status: status.textContent, error: status.classes.has("error"),
  busy: link.getAttribute("aria-busy"), answer: answer.textContent,
}))));
"""


def run_exports_js(mode: str, status: int = 200, body: Any = None) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; the browser script cannot be exercised")
    scenario = {"file": str(EXPORTS_JS), "mode": mode, "status": status, "body": body}
    done = subprocess.run(
        [node, "-e", HARNESS, json.dumps(scenario)], capture_output=True, text=True, timeout=60, check=True
    )
    output: dict[str, Any] = json.loads(done.stdout)
    return output


def test_js_successful_download_uses_the_server_filename() -> None:
    result = run_exports_js("ok")
    assert result["prevented"] and result["clicked"] == [{"href": "blob:1", "download": "orders-report.xlsx"}]
    assert result["status"] == "Downloaded." and not result["error"] and result["busy"] is None
    assert result["revoked"] == ["blob:1"]


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (404, {"detail": "Not found"}, "Not found"),
        (500, {"detail": "The workbook could not be created."}, "The workbook could not be created."),
        (422, {"detail": [{"loc": ["query"], "msg": "bad"}]}, "The export failed. The result is still shown above."),
    ],
)
def test_js_failed_download_shows_the_error_and_keeps_the_page(status: int, body: Any, message: str) -> None:
    result = run_exports_js("fail", status, body)
    assert result["clicked"] == [] and result["status"] == message and result["error"] is True
    assert result["answer"] == "The analysis answer" and result["busy"] is None


def test_js_network_error_is_reported() -> None:
    result = run_exports_js("network")
    assert result["status"] == "The export failed (network problem). Try again." and result["error"] is True
