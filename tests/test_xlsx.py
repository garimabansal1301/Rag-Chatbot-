"""Phase 5: Excel (.xlsx) workbook import into the dataset system. Generated workbooks, temp folders, no network."""

from __future__ import annotations

import datetime
import functools
import io
import json
import re
import zipfile
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook, load_workbook

from app.chat.engine import get_engine
from app.config import Settings
from app.datasets import parsing, xlsx
from app.datasets.analysis import Query, load_frame, resolve_dataset, run_query, summarize
from app.datasets.parsing import DatasetError
from app.datasets.store import DatasetNotFoundError, DatasetStore
from app.datasets.xlsx import import_sheet, list_sheets
from app.deps import Components, get_components
from app.main import create_app
from app.models import WorkbookSheet
from app.tools import get_tools
from app.tools.dataset_tools import dataset_context, make_dataset_tools
from tests.fakes import TOOL_TRIGGER, FakeChatModel, StaticRetriever

SESSION_A, SESSION_B = "a" * 32, "b" * 32
HEADER = ["id", "date", "product", "sales", "units", "total", "note"]


def make_xlsx(
    sheets: dict[str, list[list[Any]]],
    *,
    hidden: tuple[str, ...] = (),
    cached: dict[str, str] | None = None,
    errors: tuple[str, ...] = (),
    text_cells: dict[str, str] | None = None,
) -> bytes:
    """A small workbook. `cached` adds the value Excel would have saved for a formula cell (by reference, first
    sheet); `errors` turns cells of the first sheet into Excel error values; `text_cells` stores text as text
    even when it starts with '=' (openpyxl would otherwise write a formula)."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(row)
        if name in hidden:
            sheet.sheet_state = "hidden"
    first = workbook.worksheets[0]
    for ref in errors:
        first[ref].value = "#N/A"
        first[ref].data_type = "e"
    for ref, text in (text_cells or {}).items():
        first[ref].value = text
        first[ref].data_type = "s"
    buffer = io.BytesIO()
    workbook.save(buffer)
    data = buffer.getvalue()
    return patch_part(data, "xl/worksheets/sheet1.xml", _cache(cached)) if cached else data


def _cache(cached: dict[str, str]) -> Any:
    """Write saved formula results the way Excel does: numbers plain, text with t="str", errors with t="e"."""

    def edit(xml: bytes) -> bytes:
        for ref, value in cached.items():
            if value.startswith("#"):
                kind = b' t="e"'
            elif re.fullmatch(r"-?\d+(\.\d+)?", value):
                kind = b""
            else:
                kind = b' t="str"'
            pattern = rb'<c r="' + ref.encode() + rb'"([^>]*)><f>([^<]*)</f><v ?/>'
            replacement = rb'<c r="' + ref.encode() + rb'"\1' + kind + rb"><f>\2</f><v>" + value.encode() + rb"</v>"
            xml, count = re.subn(pattern, replacement, xml)
            assert count == 1, ref
        return xml

    return edit


def patch_part(
    data: bytes, part: str | None, edit: Any = None, add: dict[str, bytes] | None = None, drop: str = ""
) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            if item.filename == drop:
                continue
            content = source.read(item.filename)
            target.writestr(item, edit(content) if part and item.filename == part else content)
        for name, content in (add or {}).items():
            target.writestr(name, content)
    return out.getvalue()


ORDERS = [
    HEADER,
    [1, datetime.date(2024, 1, 5), "Laptop", 1200.5, 2, "=D2*E2", "ok"],
    [2, datetime.datetime(2024, 2, 10, 14, 30), "Chair", None, 1, "=D3*E3", "fine"],
    [],
    [3, datetime.date(2024, 3, 1), "Pen", 20, 10, "=D5*E5", "x"],
    [4, datetime.date(2024, 3, 2), "Desk", 449, 1, "=D6*E6", "y"],
]


def orders_workbook(**extra: Any) -> bytes:
    sheets = {"Orders": ORDERS, "Totals": [["region", "amount"], ["North", 10], ["South", 20]], **extra}
    return make_xlsx(sheets, cached={"F2": "2401", "F5": "200", "F6": "#DIV/0!"}, errors=("G6",), hidden=("Secret",))


# ---------------------------------------------------------------- worksheet listing and conversion
def test_list_sheets_with_visibility_and_size() -> None:
    data = orders_workbook(Secret=[["x"], [1]])
    assert list_sheets(data) == [
        WorkbookSheet(index=0, name="Orders", state="visible", rows=6, columns=7),
        WorkbookSheet(index=1, name="Totals", state="visible", rows=3, columns=2),
        WorkbookSheet(index=2, name="Secret", state="hidden", rows=2, columns=1),
    ]


def test_chart_sheets_are_not_listed() -> None:
    from openpyxl.chart import BarChart, Reference

    workbook = Workbook()
    workbook.active.append(["a"])
    workbook.active.append([1])
    chart = BarChart()
    chart.add_data(Reference(workbook.active, min_col=1, min_row=1, max_row=2), titles_from_data=True)
    workbook.create_chartsheet("Chart1").add_chart(chart)
    buffer = io.BytesIO()
    workbook.save(buffer)
    assert [s.name for s in list_sheets(buffer.getvalue())] == ["Sheet"]


def test_import_keeps_types_missing_values_and_flags_formulas() -> None:
    parsed, source = import_sheet(orders_workbook(), 0)
    assert [(c.name, c.dtype, c.missing) for c in parsed.columns] == [
        ("id", "integer", 0),
        ("date", "date", 0),
        ("product", "text", 0),
        ("sales", "decimal", 1),
        ("units", "integer", 0),
        ("total", "integer", 2),  # D3*E3 had no saved value, D6*E6 saved #DIV/0!
        ("note", "text", 1),  # G6 is an Excel error value
    ]
    assert parsed.rows == 4 and parsed.preview[0] == ["1", "2024-01-05", "Laptop", "1200.5", "2", "2401", "ok"]
    assert parsed.preview[1][1] == "2024-02-10T14:30:00" and parsed.preview[1][3] is None
    assert source.sheet == "Orders" and source.formula_cells == {"total": 2}
    assert source.preview_formula_cells == [(0, 5), (2, 5)] and source.uncached_formula_cells == 1
    assert source.error_cells == 2
    text = " ".join(parsed.warnings)
    assert "2 cell(s) in total hold results of Excel formulas" in text and "not recalculated" in text
    assert "1 formula cell(s) had no saved value" in text and "2 Excel error value(s)" in text
    assert "1 empty row(s) were skipped" in text
    assert "D2*E2" not in parsed.text and "=D" not in parsed.text, "formulas themselves are never imported"


def test_formula_values_are_never_recalculated() -> None:
    data = make_xlsx({"S": [["a", "b", "sum"], [1, 2, "=A2+B2"]]}, cached={"C2": "999"})
    parsed, source = import_sheet(data, 0)
    assert parsed.preview[0] == ["1", "2", "999"], "the saved value is used as is, even if it looks wrong"
    assert source.formula_cells == {"sum": 1}


def test_formula_in_the_header_row_gives_the_saved_name() -> None:
    data = make_xlsx({"S": [['="x"&"y"', "b"], [1, 2]]}, cached={"A1": "xy"})
    parsed, source = import_sheet(data, 0)
    assert [c.name for c in parsed.columns][0] == "A1" or parsed.columns[0].name == "xy"
    assert source.formula_cells == {}, "header cells are not data"


def test_text_that_looks_like_a_formula_stays_text() -> None:
    data = make_xlsx({"S": [["note", "n"], ["placeholder", 1], ["y", 2]]}, text_cells={"A2": '=HYPERLINK("x")'})
    parsed, source = import_sheet(data, 0)
    assert parsed.preview[0][0] == '=HYPERLINK("x")' and parsed.formula_like_cells == 1
    assert source.formula_cells == {}, "a text cell is not a formula cell"


def test_duplicate_and_blank_headers_and_extra_columns() -> None:
    data = make_xlsx({"S": [["Sales", "sales", None, "Sales"], [1, 2, 3, 4, 5]]})
    parsed, _ = import_sheet(data, 0)
    assert [c.name for c in parsed.columns] == ["Sales", "sales_2", "column_3", "Sales_3", "column_5"]
    assert any("Duplicate column names" in w for w in parsed.warnings)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "is empty"),
        ([["id", "x"]], "the first row holds the column names, but there are no data rows"),
        ([["  ", " "], [1, 2]], "has no column names in its first row"),
    ],
)
def test_empty_and_headerless_sheets_are_rejected(rows: list[list[Any]], message: str) -> None:
    with pytest.raises(DatasetError, match=re.escape(message)):
        import_sheet(make_xlsx({"S": rows}), 0)


def test_row_and_column_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parsing, "MAX_ROWS", 3)
    monkeypatch.setattr(parsing, "MAX_COLUMNS", 2)
    assert import_sheet(make_xlsx({"S": [["a", "b"], [1, 2], [3, 4], [5, 6]]}), 0)[0].rows == 3
    with pytest.raises(DatasetError, match="more than 3 data rows"):
        import_sheet(make_xlsx({"S": [["a", "b"], [1, 2], [3, 4], [5, 6], [7, 8]]}), 0)
    with pytest.raises(DatasetError, match="more than 2 columns"):
        import_sheet(make_xlsx({"S": [["a", "b", "c"], [1, 2, 3]]}), 0)


def test_unknown_sheet_index() -> None:
    with pytest.raises(DatasetError, match="does not exist"):
        import_sheet(orders_workbook(), 9)


# ---------------------------------------------------------------- invalid, encrypted, renamed, unsupported files
def zip_with_flag(data: bytes) -> bytes:
    """Mark the first central-directory entry as encrypted (general-purpose flag bit 0)."""
    raw = bytearray(data)
    position = raw.find(b"PK\x01\x02")
    raw[position + 8] |= 0x01
    return bytes(raw)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "The file is empty."),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100, "password-protected (encrypted) or an old .xls"),
        (b"id,name\n1,a\n", "not a valid .xlsx workbook"),
        (b"PK\x03\x04" + b"garbage" * 10, "damaged"),
        (orders_workbook()[:-200], "damaged"),
    ],
)
def test_invalid_files_are_rejected(data: bytes, message: str) -> None:
    with pytest.raises(DatasetError, match=re.escape(message)):
        list_sheets(data)


def test_renamed_macro_encrypted_and_bomb_packages() -> None:
    good = orders_workbook()
    with pytest.raises(DatasetError, match="another file type renamed"):
        list_sheets(patch_part(good, None, drop="xl/workbook.xml"))
    with pytest.raises(DatasetError, match="Macro-enabled"):
        list_sheets(patch_part(good, None, add={"xl/vbaProject.bin": b"\x00" * 10}))
    macro_types = patch_part(
        good, "[Content_Types].xml", lambda x: x.replace(b"sheet.main+xml", b"sheet.macroEnabled.main+xml")
    )
    with pytest.raises(DatasetError, match="Macro-enabled"):
        list_sheets(macro_types)
    with pytest.raises(DatasetError, match="encrypted"):
        list_sheets(zip_with_flag(good))
    with pytest.raises(DatasetError, match="too much data"):
        list_sheets(patch_part(good, None, add={"xl/media/big.bin": b"\x00" * (3 * 1024 * 1024)}))


def test_too_many_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xlsx, "MAX_ZIP_ENTRIES", 3)
    with pytest.raises(DatasetError, match="too many internal parts"):
        list_sheets(orders_workbook())


# ---------------------------------------------------------------- pending uploads in the store
def test_pending_workbooks_are_per_session_bounded_and_expire(tmp_path: Any) -> None:
    store = DatasetStore(tmp_path)
    sheets = [WorkbookSheet(index=0, name="S")]
    first = store.save_pending(SESSION_A, "a.xlsx", b"one", sheets)
    assert store.get_pending(SESSION_A, first.id)[1] == b"one" and store.latest_pending(SESSION_A) == first
    with pytest.raises(DatasetNotFoundError):
        store.get_pending(SESSION_B, first.id)
    with pytest.raises(DatasetNotFoundError):
        store.get_pending(SESSION_A, "../" + first.id[3:])
    for i in range(3):
        latest = store.save_pending(SESSION_A, f"{i}.xlsx", b"x", sheets)
    assert len(list((tmp_path / SESSION_A / ".pending").glob("*.json"))) == 3, "at most 3 per session"
    with pytest.raises(DatasetNotFoundError):
        store.get_pending(SESSION_A, first.id)
    meta = tmp_path / SESSION_A / ".pending" / f"{latest.id}.json"
    expired = latest.model_copy(update={"created_at": "2020-01-01T00:00:00+00:00"})
    meta.write_text(expired.model_dump_json(), encoding="utf-8")
    with pytest.raises(DatasetNotFoundError):
        store.get_pending(SESSION_A, latest.id)
    assert store.list(SESSION_A) == [], "pending uploads are not datasets"
    store.delete_session(SESSION_A)
    assert not (tmp_path / SESSION_A).exists()


# ---------------------------------------------------------------- UI and API workflow
@pytest.fixture()
def ui(settings: Settings) -> Iterator[tuple[TestClient, DatasetStore, FakeChatModel]]:
    app = create_app(settings)
    datasets: DatasetStore = app.state.dataset_store
    llm = FakeChatModel(
        reply="ANSWER",
        tool_name="query_dataset",
        tool_args={
            "aggregations": [{"column": "total", "function": "sum"}],
            "chart": None,
        },
    )
    engine = get_engine(
        llm=llm,
        retriever=StaticRetriever([]),
        tools=[*get_tools(), *make_dataset_tools(datasets)],
        session_context=functools.partial(dataset_context, datasets),
    )
    app.dependency_overrides[get_components] = lambda: Components(engine=engine)
    with TestClient(app) as client:
        yield client, datasets, llm


def new_session(client: TestClient) -> str:
    session_id: str = client.post("/api/sessions", json={}).json()["id"]
    return session_id


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def ui_upload(client: TestClient, session_id: str, name: str, data: bytes) -> str:
    return client.post(f"/ui/sessions/{session_id}/datasets", files={"file": (name, data, XLSX_MIME)}).text


def test_ui_two_step_import(ui: tuple[TestClient, DatasetStore, FakeChatModel]) -> None:
    client, datasets, _ = ui
    session_id = new_session(client)
    html = ui_upload(client, session_id, "Q1 sales.xlsx", orders_workbook(Secret=[["x"], [1]]))
    assert "Q1 sales.xlsx: choose the worksheet to import below." in html
    assert '<span class="sheet-name">Orders</span>' in html and "about 6 rows × 7 columns" in html
    assert html.count('<span class="dataset-badge sheet-hidden">hidden sheet</span>') == 1
    assert 'name="include_hidden" value="true"' in html and "Import hidden sheet" in html
    pending = datasets.latest_pending(session_id)
    assert pending is not None and datasets.list(session_id) == []
    html = client.post(f"/ui/sessions/{session_id}/workbooks/{pending.id}/import", data={"sheet": "0"}).text
    assert "Q1 sales.xlsx [Orders]: 4 rows × 7 columns imported and checked." in html
    assert "Excel sheet “Orders” of Q1 sales.xlsx" in html and "· 2 from formulas" in html
    assert html.count('<td class="formula"') == 2 and "not recalculated" in html
    assert "choose the worksheet" not in html, "the pending upload is gone after the import"
    info = datasets.list(session_id)[0]
    assert info.source is not None and info.source.workbook == "Q1 sales.xlsx" and info.source.sheet == "Orders"
    assert info.size_bytes > 0 and datasets.latest_pending(session_id) is None


def test_hidden_sheets_need_explicit_confirmation(ui: tuple[TestClient, DatasetStore, FakeChatModel]) -> None:
    client, _, _ = ui
    session_id = new_session(client)
    data = orders_workbook(Secret=[["x"], [1]])
    upload = client.post(f"/api/sessions/{session_id}/workbooks", files={"file": ("b.xlsx", data, XLSX_MIME)}).json()
    assert [s["state"] for s in upload["sheets"]] == ["visible", "visible", "hidden"]
    url = f"/api/sessions/{session_id}/workbooks/{upload['id']}/import"
    refused = client.post(url, json={"sheet": 2})
    assert refused.status_code == 400 and "hidden" in refused.json()["detail"]
    imported = client.post(url, json={"sheet": 2, "include_hidden": True})
    assert imported.status_code == 201 and imported.json()["source"]["sheet"] == "Secret"


def test_api_errors_and_session_isolation(ui: tuple[TestClient, DatasetStore, FakeChatModel]) -> None:
    client, datasets, _ = ui
    owner, other = new_session(client), new_session(client)
    upload = client.post(f"/api/sessions/{owner}/workbooks", files={"file": ("b.xlsx", orders_workbook(), XLSX_MIME)})
    upload_id = upload.json()["id"]
    assert client.get(f"/api/sessions/{other}/workbooks/{upload_id}").status_code == 404
    assert client.post(f"/api/sessions/{other}/workbooks/{upload_id}/import", json={"sheet": 0}).status_code == 404
    assert client.post(f"/ui/sessions/{other}/workbooks/{upload_id}/discard").status_code == 404
    assert "has expired" in client.post(f"/ui/sessions/{other}/workbooks/{upload_id}/import", data={"sheet": "0"}).text
    assert datasets.list(other) == []
    assert client.post(f"/api/sessions/{owner}/workbooks/{upload_id}/import", json={"sheet": 7}).status_code == 400
    assert client.post(f"/api/sessions/{owner}/workbooks/{upload_id}/import", json={"sheet": -1}).status_code == 422
    assert client.post(f"/api/sessions/{owner}/workbooks/{'c' * 32}/import", json={"sheet": 0}).status_code == 404
    assert (
        client.post(f"/api/sessions/{'f' * 32}/workbooks", files={"file": ("b.xlsx", b"x", XLSX_MIME)}).status_code
        == 404
    )
    csv_route = client.post(f"/api/sessions/{owner}/datasets", files={"file": ("b.xlsx", orders_workbook(), XLSX_MIME)})
    assert csv_route.status_code == 400 and "/workbooks" in csv_route.json()["detail"]
    assert client.delete(f"/api/sessions/{owner}/workbooks/{upload_id}").status_code == 204
    assert client.get(f"/api/sessions/{owner}/workbooks/{upload_id}").status_code == 404


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        (
            "old.xls",
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64,
            "Upload a .csv file or an Excel .xlsx workbook",
        ),
        ("macro.xlsm", b"PK\x03\x04", "Upload a .csv file or an Excel .xlsx workbook"),
        ("locked.xlsx", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64, "password-protected"),
        ("renamed.xlsx", b"id,name\n1,a\n", "not a valid .xlsx workbook"),
        ("broken.xlsx", b"PK\x03\x04broken", "damaged"),
    ],
)
def test_ui_rejects_unsupported_and_invalid_files(
    ui: tuple[TestClient, DatasetStore, FakeChatModel], name: str, data: bytes, message: str
) -> None:
    client, datasets, _ = ui
    session_id = new_session(client)
    html = ui_upload(client, session_id, name, data)
    assert 'class="dataset-status error"' in html and message in html
    assert datasets.latest_pending(session_id) is None and datasets.list(session_id) == []


def test_oversized_workbook_is_rejected(settings: Settings) -> None:
    app = create_app(replace(settings, max_dataset_bytes=1000))
    app.dependency_overrides[get_components] = lambda: Components(engine=get_engine(llm=FakeChatModel(), tools=[]))
    with TestClient(app) as client:
        session_id = new_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/workbooks", files={"file": ("big.xlsx", orders_workbook(), XLSX_MIME)}
        )
        assert response.status_code == 400 and "too large" in response.json()["detail"]


def test_malicious_names_are_cleaned_and_escaped(ui: tuple[TestClient, DatasetStore, FakeChatModel]) -> None:
    client, datasets, _ = ui
    session_id = new_session(client)
    data = make_xlsx({"<b>bad<i>": [["a"], [1]]})  # Excel forbids "/" in sheet names, so no closing tags
    html = ui_upload(client, session_id, "../../<script>x</script>.xlsx", data)
    assert "<script>x</script>" not in html and "<b>bad<i>" not in html and "&lt;b&gt;bad&lt;i&gt;" in html
    pending = datasets.latest_pending(session_id)
    assert pending is not None and "/" not in pending.filename and ".." not in pending.filename
    assert sorted(p.name for p in (datasets.root / session_id).iterdir()) == [".pending"]


def test_deleting_the_session_removes_pending_workbooks_and_datasets(
    ui: tuple[TestClient, DatasetStore, FakeChatModel],
) -> None:
    client, datasets, _ = ui
    session_id = new_session(client)
    upload = client.post(
        f"/api/sessions/{session_id}/workbooks", files={"file": ("b.xlsx", orders_workbook(), XLSX_MIME)}
    )
    client.post(f"/api/sessions/{session_id}/workbooks/{upload.json()['id']}/import", json={"sheet": 1})
    ui_upload(client, session_id, "c.xlsx", orders_workbook())
    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert not (datasets.root / session_id).exists()


# ---------------------------------------------------------------- analysis, LangGraph, charts and exports
@pytest.fixture()
def imported(
    ui: tuple[TestClient, DatasetStore, FakeChatModel],
) -> tuple[TestClient, DatasetStore, FakeChatModel, str, str]:
    client, datasets, llm = ui
    session_id = new_session(client)
    upload = client.post(
        f"/api/sessions/{session_id}/workbooks", files={"file": ("Q1.xlsx", orders_workbook(), XLSX_MIME)}
    )
    dataset_id = client.post(
        f"/api/sessions/{session_id}/workbooks/{upload.json()['id']}/import", json={"sheet": 0}
    ).json()["id"]
    return client, datasets, llm, session_id, dataset_id


def test_analysis_works_on_imported_sheets_and_mentions_formulas(
    imported: tuple[TestClient, DatasetStore, FakeChatModel, str, str],
) -> None:
    client, datasets, llm, session_id, dataset_id = imported
    info = datasets.get(session_id, dataset_id)
    frame = load_frame(datasets, session_id, info)
    totals = run_query(frame, info, Query.model_validate({"aggregations": [{"column": "sales", "function": "sum"}]}))
    assert totals.rows == [["4", "1669.5"]] and not any("Excel formulas" in n for n in totals.notes)
    formula = run_query(frame, info, Query.model_validate({"aggregations": [{"column": "total", "function": "sum"}]}))
    assert formula.rows == [["4", "2601"]] and any("saved by Excel formulas" in n for n in formula.notes)
    assert any("saved by Excel formulas" in n for n in summarize(frame, info).notes)
    for name in ("Orders", "Q1.xlsx", "q1.xlsx [orders]"):
        assert resolve_dataset(datasets, session_id, name).id == dataset_id
    turn = client.post(
        f"/api/sessions/{session_id}/messages", json={"content": f"{TOOL_TRIGGER} total of total"}
    ).json()
    assert turn["assistant_message"]["analysis"][0]["rows"] == [["4", "2601"]]
    prompt = llm.prompt_text()
    assert 'sheet "Orders" of Excel workbook "Q1.xlsx"' in prompt and "total (integer, Excel formula results)" in prompt
    assert "Laptop" not in prompt.split("tool said")[0], "no rows in the prompt"


def test_charts_work_on_imported_sheets(imported: tuple[TestClient, DatasetStore, FakeChatModel, str, str]) -> None:
    _, datasets, _, session_id, dataset_id = imported
    info = datasets.get(session_id, dataset_id)
    table = run_query(
        load_frame(datasets, session_id, info),
        info,
        Query.model_validate(
            {
                "group_by_month": "date",
                "aggregations": [{"column": "units", "function": "sum"}],
                "chart": {"type": "line"},
            }
        ),
    )
    assert table.chart is not None and table.chart.labels == ["2024-01", "2024-02", "2024-03"]
    assert table.chart.values == [2.0, 1.0, 11.0]


def test_excel_exports_of_imported_sheets(imported: tuple[TestClient, DatasetStore, FakeChatModel, str, str]) -> None:
    client, datasets, _, session_id, dataset_id = imported
    response = client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}/report.xlsx?source=true")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="Q1-Orders-report.xlsx"'
    book = load_workbook(io.BytesIO(response.content))
    facts = {r[0].value: r[1].value for r in book["Report"].iter_rows() if len(r) > 1 and r[0].value}
    assert facts["Excel workbook"] == "Q1.xlsx" and facts["Worksheet"] == "Orders"
    notes = [r[0].value for r in book["Report"].iter_rows() if r[0].value and isinstance(r[0].value, str)]
    assert any("hold values that Excel formulas computed" in n for n in notes)
    assert any("Excel error value" in n for n in notes)
    source = book["Source Data"]
    assert source["B2"].value == datetime.datetime(2024, 1, 5) and source["D2"].value == 1200.5
    assert source["F3"].value is None and all(c.data_type != "f" for r in source.iter_rows() for c in r)


def test_formula_like_text_is_neutralised_on_export(ui: tuple[TestClient, DatasetStore, FakeChatModel]) -> None:
    client, _, _ = ui
    session_id = new_session(client)
    data = make_xlsx({"S": [["note", "n"], ["placeholder", 1]]}, text_cells={"A2": "=cmd|' /C calc'!A0"})
    upload = client.post(f"/api/sessions/{session_id}/workbooks", files={"file": ("t.xlsx", data, XLSX_MIME)}).json()
    dataset_id = client.post(f"/api/sessions/{session_id}/workbooks/{upload['id']}/import", json={"sheet": 0}).json()[
        "id"
    ]
    book = load_workbook(io.BytesIO(client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}/data.xlsx").content))
    cell = book["Source Data"]["A2"]
    assert cell.value == "'=cmd|' /C calc'!A0" and cell.data_type == "s"


def test_csv_upload_is_unchanged(ui: tuple[TestClient, DatasetStore, FakeChatModel]) -> None:
    client, _, _ = ui
    session_id = new_session(client)
    html = client.post(
        f"/ui/sessions/{session_id}/datasets", files={"file": ("plain.csv", b"a,b\n1,2\n", "text/csv")}
    ).text
    assert "plain.csv: 1 rows × 2 columns uploaded and checked." in html and "Excel sheet" not in html
    info = client.get(f"/api/sessions/{session_id}/datasets").json()[0]
    assert info["source"] is None and json.dumps(info).count("formula") == 1  # only formula_like_cells
