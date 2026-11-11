"""Excel (.xlsx) exports of a session's dataset and of saved analysis results, built in memory with openpyxl.

What goes in comes only from trusted places: the session's stored, validated CSV (read through DatasetStore)
or an AnalysisTable saved with one of the session's messages. Nothing is recalculated by the LLM.

Safety and accuracy rules:
  - every text cell is neutralised against formula injection (`neutralize_formula`, which also looks past
    leading spaces and control characters), stripped of characters XML cannot hold, length-capped, and stored
    with the string type so a value like "=1+1" can never become a formula; numbers and dates are written
    as real numbers and dates (a negative number is never touched), missing values as blank cells;
  - workbooks are built in write-only mode and are bounded (cells per sheet, total size).
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from app.datasets.analysis import NUMERIC, column_profiles
from app.datasets.parsing import neutralize_formula
from app.models import AnalysisTable, ChartSpec, DatasetInfo

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MAX_EXPORT_CELLS = 2_000_000  # source-data cells per workbook (rows are cut beyond this, with a note)
MAX_WORKBOOK_BYTES = 50 * 1024 * 1024
MAX_SHEET_NAME = 31  # Excel's limit
MAX_CELL_TEXT = 32_767  # Excel's limit
MAX_COLUMN_WIDTH = 60
WIDTH_SAMPLE_ROWS = 200
FORMATS = {
    "integer": "#,##0",
    "decimal": "#,##0.00",
    "percent": "0.0%",
    "date": "yyyy-mm-dd",
    "datetime": "yyyy-mm-dd hh:mm",
}
_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_PERCENT = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)%$")
_LEADING_ZERO = re.compile(r"^[+-]?0\d")  # codes such as "00123" stay text
_BOLD = Font(bold=True)
_TITLE = Font(bold=True, size=14)

Cell = tuple[Any, str | None]  # (value for Excel, number format)


class ExportError(Exception):
    """The workbook cannot be produced as asked. The message is safe to show."""


# ---------------------------------------------------------------- safe text, names
def excel_text(value: str) -> str:
    """Text for a cell: formula-neutralised, without characters XML cannot hold, within Excel's length limit."""
    text = ILLEGAL_CHARACTERS_RE.sub("", neutralize_formula(value))
    return text[:MAX_CELL_TEXT]


def safe_sheet_name(name: str, used: set[str]) -> str:
    """A unique worksheet name: no []:*?/\\ characters, not 'History', at most 31 characters."""
    cleaned = ILLEGAL_CHARACTERS_RE.sub("", _INVALID_SHEET_CHARS.sub(" ", name))
    cleaned = " ".join(cleaned.split()).strip("'") or "Sheet"
    if cleaned.casefold() == "history":  # reserved by Excel
        cleaned = "History data"
    base, candidate, number = cleaned[:MAX_SHEET_NAME].rstrip(), cleaned[:MAX_SHEET_NAME].rstrip(), 2
    while candidate.casefold() in used:
        suffix = f" ({number})"
        candidate, number = base[: MAX_SHEET_NAME - len(suffix)].rstrip() + suffix, number + 1
    used.add(candidate.casefold())
    return candidate


def download_filename(*parts: str) -> str:
    """An ASCII file name such as 'orders_sample-report.xlsx' (no path, no quotes)."""
    stem = "-".join(p for p in (re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("._-") for part in parts) if p)
    return f"{(stem or 'export')[:100]}.xlsx"


def content_disposition(filename: str) -> str:
    return f'attachment; filename="{filename}"'


def unique_headers(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    headers = []
    for raw in names:
        name, number = raw or "column", 2
        candidate = name
        while candidate.casefold() in seen:
            candidate, number = f"{name} ({number})", number + 1
        seen.add(candidate.casefold())
        headers.append(candidate)
    return headers


# ---------------------------------------------------------------- typed cell values
def typed(value: Any, kind: str | None = None) -> Cell:
    """A pandas/Python value as an Excel value: numbers, dates and booleans keep their type; missing -> blank.

    `kind` ("integer"/"decimal", from the column type) keeps one number format for a whole column.
    """
    if value is None:
        return None, None
    try:
        if pd.isna(value):
            return None, None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, str | pd.Timestamp):  # numpy scalar
        value = value.item()
    if isinstance(value, bool):
        return value, None
    if isinstance(value, pd.Timestamp | date):
        return _temporal(value)
    if isinstance(value, int):
        return value, FORMATS["integer"]
    if isinstance(value, float):
        return _float(value, kind)
    return excel_text(str(value)), None


def _float(value: float, kind: str | None) -> Cell:
    if not math.isfinite(value):
        return None, None  # Excel cannot hold NaN or infinity: a blank cell
    if kind in ("integer", "decimal"):
        return value, FORMATS[kind]
    return value, FORMATS["integer"] if value.is_integer() else FORMATS["decimal"]


def _temporal(value: pd.Timestamp | date) -> Cell:
    """Dates as dates; date-times (without time zone, which Excel cannot store) as date-times."""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        value = value.replace(tzinfo=None)
        if value.time() == datetime.min.time():
            return value.date(), FORMATS["date"]
        return value, FORMATS["datetime"]
    return value, FORMATS["date"]


def from_text(text: str | None, kind: str) -> Cell:
    """A saved result cell (formatted text) back as a typed Excel value, using the column's type."""
    if text is None:
        return None, None
    if kind in ("integer", "decimal", "mixed") and _NUMBER.match(text.strip()):
        number = float(text)
        if math.isfinite(number):
            whole = number.is_integer() and abs(number) < 2**53
            if kind == "integer" and whole:
                return int(number), FORMATS["integer"]
            if kind == "mixed" and whole and "." not in text and "e" not in text.lower():
                return int(number), FORMATS["integer"]
            return number, FORMATS["decimal"]  # a decimal column keeps one format, e.g. 449 -> 449.00
    if kind == "percent" and _PERCENT.match(text.strip()):
        return float(text.strip()[:-1]) / 100, FORMATS["percent"]
    if kind in ("date", "mixed"):
        parsed = _iso_date(text)
        if parsed is not None:
            return parsed
    if kind == "boolean" and text in ("true", "false"):
        return text == "true", None
    return excel_text(text), None


def _iso_date(text: str) -> Cell | None:
    try:
        if len(text) == 10:
            return date.fromisoformat(text), FORMATS["date"]
        if len(text) > 10 and text[4] == "-" and "T" in text:
            return datetime.fromisoformat(text).replace(tzinfo=None), FORMATS["datetime"]
    except ValueError:
        return None
    return None


def infer_kinds(table: AnalysisTable) -> list[str]:
    """Column types for results saved before types were stored: numbers, percents or dates only when every
    value fits (codes with leading zeros stay text); everything else is text."""
    if table.column_types is not None:
        return list(table.column_types)
    kinds = []
    for index in range(len(table.columns)):
        values: list[str] = [v for row in table.rows if index < len(row) and (v := row[index]) is not None]
        if values and all(_NUMBER.match(v) and not _LEADING_ZERO.match(v) for v in values):
            whole = all("." not in v and "e" not in v.lower() for v in values)
            kinds.append("integer" if whole else "decimal")
        elif values and all(_PERCENT.match(v) for v in values):
            kinds.append("percent")
        elif values and all(_iso_date(v) is not None for v in values):
            kinds.append("date")
        else:
            kinds.append("text")
    return kinds


# ---------------------------------------------------------------- sheet writers
@dataclass
class _Book:
    workbook: Workbook
    used_names: set[str]

    def sheet(self, name: str) -> Any:
        return self.workbook.create_sheet(safe_sheet_name(name, self.used_names))


def _cell(ws: Any, value: Any, number_format: str | None = None, font: Font | None = None) -> WriteOnlyCell:
    cell = WriteOnlyCell(ws, value)
    if isinstance(value, str):
        cell.data_type = "s"  # never a formula
    if number_format:
        cell.number_format = number_format
    if font is not None:
        cell.font = font
    return cell


def _width(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, date):
        return 12
    if isinstance(value, float):
        return min(len(f"{value:,.2f}"), 20)
    return len(str(value))


def _write_table(
    ws: Any,
    headers: list[str],
    rows: Iterable[list[Cell]],
    *,
    count: int | None = None,
    sample: list[list[Cell]] | None = None,
) -> None:
    """Bold header, frozen first row, filter, readable column widths, typed cells.

    `rows` may be a generator (large source data is streamed); then `count` and a width `sample` are given.
    """
    if isinstance(rows, list):
        count, sample = len(rows), rows[:WIDTH_SAMPLE_ROWS]
    sample = sample or []
    headers = [excel_text(h) for h in unique_headers(headers)]
    for index, header in enumerate(headers, start=1):
        values = [row[index - 1][0] for row in sample if index - 1 < len(row)]
        width = max([_width(header), *(_width(v) for v in values)]) + 2
        ws.column_dimensions[get_column_letter(index)].width = max(10, min(width, MAX_COLUMN_WIDTH))
    ws.freeze_panes = "A2"
    if headers:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(count or 0, 1) + 1}"
    ws.append([_cell(ws, h, font=_BOLD) for h in headers])
    for row in rows:
        ws.append([_cell(ws, value, number_format) for value, number_format in row])


def _write_report(book: _Book, title: str, facts: list[tuple[str, Any]], notes: list[str]) -> None:
    ws = book.sheet("Report")
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 60
    ws.append([_cell(ws, excel_text(title), font=_TITLE)])
    ws.append([])
    for label, value in facts:
        cell_value, number_format = typed(value)
        ws.append([_cell(ws, excel_text(label), font=_BOLD), _cell(ws, cell_value, number_format)])
    if notes:
        ws.append([])
        ws.append([_cell(ws, "Notes", font=_BOLD)])
        for note in notes:
            ws.append([_cell(ws, excel_text(note))])


def _write_analysis(book: _Book, table: AnalysisTable) -> None:
    kinds = infer_kinds(table)
    rows = [
        [from_text(row[i] if i < len(row) else None, kinds[i]) for i in range(len(table.columns))] for row in table.rows
    ]
    _write_table(book.sheet(table.title or "Analysis"), list(table.columns), rows)
    if table.chart is not None:
        _write_chart(book, table.chart)


def _write_chart(book: _Book, chart: ChartSpec) -> None:
    """The chart's data on its own sheet, with a native Excel chart drawn from those cells."""
    ws = book.sheet("Chart")
    headers = [chart.label_name, chart.series_name] + (["percent of total"] if chart.percentages else [])
    rows: list[list[Cell]] = []
    for index, (label, value) in enumerate(zip(chart.labels, chart.values, strict=True)):
        row: list[Cell] = [(excel_text(label), None), typed(value)]
        if chart.percentages:
            row.append((round(chart.percentages[index] / 100, 6), FORMATS["percent"]))
        rows.append(row)
    _write_table(ws, headers, rows)
    count = len(rows)
    native: BarChart | LineChart | PieChart = {"bar": BarChart, "line": LineChart, "pie": PieChart}[chart.type]()
    native.title = ILLEGAL_CHARACTERS_RE.sub("", chart.title)
    native.add_data(Reference(ws, min_col=2, min_row=1, max_row=count + 1), titles_from_data=True)
    native.set_categories(Reference(ws, min_col=1, min_row=2, max_row=count + 1))
    if isinstance(native, BarChart | LineChart):
        native.x_axis.title = ILLEGAL_CHARACTERS_RE.sub("", chart.label_name)
        native.y_axis.title = ILLEGAL_CHARACTERS_RE.sub("", chart.series_name)
        native.legend = None
    native.width, native.height = 20, 10
    ws.add_chart(native, f"{get_column_letter(len(headers) + 2)}2")


def _save(book: _Book) -> bytes:
    buffer = io.BytesIO()
    book.workbook.save(buffer)
    content = buffer.getvalue()
    if len(content) > MAX_WORKBOOK_BYTES:
        raise ExportError("The workbook would be too large to download. Export a smaller result instead.")
    return content


def _new_book() -> _Book:
    return _Book(Workbook(write_only=True), set())


def _timestamp(moment: datetime | None) -> str:
    return (moment or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------- workbooks
def analysis_workbook(table: AnalysisTable, exported_at: datetime | None = None) -> bytes:
    """One saved result: Report (title, dataset, time, notes), the result table, and its chart if it had one."""
    book = _new_book()
    notes = list(table.notes) + (list(table.chart.notes) if table.chart else [])
    if table.total_rows > len(table.rows):
        notes.append(
            f"This result had {table.total_rows:,} rows; the {len(table.rows):,} rows shown in the chat are exported."
        )
    notes.append("Blank cells are missing values.")
    _write_report(
        book,
        f"Analysis result: {table.title}",
        [
            ("Dataset", table.dataset),
            ("Exported at", _timestamp(exported_at)),
            ("Rows in this export", len(table.rows)),
        ],
        notes,
    )
    _write_analysis(book, table)
    return _save(book)


def dataset_workbook(
    info: DatasetInfo,
    frame: pd.DataFrame,
    *,
    summary: bool = True,
    source: bool = False,
    table: AnalysisTable | None = None,
    exported_at: datetime | None = None,
) -> bytes:
    """A dataset report: Report, Dataset Summary, Data Quality, an optional saved Analysis result (and chart),
    and the Source Data only when asked for (bounded to MAX_EXPORT_CELLS)."""
    book = _new_book()
    total, columns = len(frame), len(info.columns)
    duplicates = int(frame.duplicated().sum()) if total else 0
    source_rows = min(total, MAX_EXPORT_CELLS // max(columns, 1)) if source else 0
    facts: list[tuple[str, Any]] = [
        ("Dataset", info.filename),
        ("Exported at", _timestamp(exported_at)),
        ("Rows", total),
        ("Columns", columns),
        ("Duplicate rows", duplicates),
    ]
    notes = ["Blank cells are missing values. Missing values are excluded from sums, means and other statistics."]
    origin = info.source
    if origin is not None:  # a worksheet imported from an Excel workbook
        facts[1:1] = [("Excel workbook", origin.workbook), ("Worksheet", origin.sheet)]
        if origin.formula_cells:
            notes.append(
                "Columns " + ", ".join(list(origin.formula_cells)[:20]) + " hold values that Excel formulas computed "
                "in the original workbook; they were imported as values and not recalculated."
            )
        notes.extend(w for w in info.warnings if "Excel error" in w or "no saved value" in w)
    if table is not None:
        facts.append(("Analysis result", table.title))
        notes.extend(table.notes)
    if source:
        facts.append(("Source data rows exported", source_rows))
        if source_rows < total:
            notes.append(f"Source Data holds the first {source_rows:,} of {total:,} rows (export size limit).")
        notes.append("Source Data is the uploaded file as stored; analysis sheets are calculated results.")
    _write_report(book, ("Dataset report: " if summary else "Dataset export: ") + info.filename, facts, notes)
    if summary:
        _write_summary(book, info, frame)
        _write_quality(book, info, frame, duplicates)
    if table is not None:
        _write_analysis(book, table)
    if source:
        _write_source(book, info, frame, source_rows)
    return _save(book)


def _write_summary(book: _Book, info: DatasetInfo, frame: pd.DataFrame) -> None:
    rows: list[list[Cell]] = []
    for profile in column_profiles(frame, info):
        rows.append(
            [
                (excel_text(profile.name), None),
                (profile.dtype, None),
                (profile.missing, FORMATS["integer"]),
                (round(profile.missing_pct / 100, 6), FORMATS["percent"]),
                (profile.unique, FORMATS["integer"]),
                typed(profile.low, profile.dtype),
                typed(profile.mean, "decimal") if profile.dtype in NUMERIC else (None, None),
                typed(profile.high, profile.dtype),
            ]
        )
    _write_table(
        book.sheet("Dataset Summary"),
        ["Column", "Type", "Missing values", "Missing %", "Unique values", "Min", "Mean", "Max"],
        rows,
    )


def _write_quality(book: _Book, info: DatasetInfo, frame: pd.DataFrame, duplicates: int) -> None:
    total = len(frame)
    complete = int(frame.notna().all(axis=1).sum()) if total else 0
    observation = {
        "empty": "No values",
        "mostly missing": "At least half missing",
        "single value": "A single repeated value",
        "": "",
    }
    rows: list[list[Cell]] = [
        [
            (excel_text(p.name), None),
            (p.missing, FORMATS["integer"]),
            (round(p.missing_pct / 100, 6), FORMATS["percent"]),
            (observation[p.quality], None),
        ]
        for p in column_profiles(frame, info)
    ]
    ws = book.sheet("Data Quality")
    _write_table(ws, ["Column", "Missing values", "Missing %", "Observation"], rows)
    ws.append([])
    for label, value, number_format in [
        ("Rows", total, FORMATS["integer"]),
        ("Duplicate rows", duplicates, FORMATS["integer"]),
        ("Duplicate rows %", duplicates / total if total else 0.0, FORMATS["percent"]),
        ("Rows with no missing values", complete, FORMATS["integer"]),
    ]:
        ws.append([_cell(ws, label, font=_BOLD), _cell(ws, value, number_format)])


def _write_source(book: _Book, info: DatasetInfo, frame: pd.DataFrame, limit: int) -> None:
    """The stored rows (first `limit`), streamed row by row so large exports stay light on memory."""
    names = [c.name for c in info.columns]
    kinds = [c.dtype for c in info.columns]
    data = frame[names].head(limit)

    def convert(record: tuple[Any, ...]) -> list[Cell]:
        return [typed(value, kind) for value, kind in zip(record, kinds, strict=True)]

    def rows() -> Iterable[list[Cell]]:
        for record in data.itertuples(index=False, name=None):
            yield convert(record)

    sample = [convert(record) for record in data.head(WIDTH_SAMPLE_ROWS).itertuples(index=False, name=None)]
    _write_table(book.sheet("Source Data"), names, rows(), count=len(data), sample=sample)
