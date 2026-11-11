"""Import one worksheet of an uploaded .xlsx workbook into the existing CSV dataset pipeline.

The worksheet is converted to canonical CSV text (numbers as numbers, dates as ISO dates, missing and error
cells as empty) and then validated by the same `parse_csv` as uploaded CSVs, so header clean-up, row/column
limits, type inference, preview and formula-text detection are shared.

Safety and resource rules:
  - the file must be a real .xlsx (zip) container: encrypted/legacy .xls (OLE), macro-enabled workbooks,
    renamed non-Excel files and zip bombs (too many entries, too much uncompressed data, extreme
    compression) are rejected before openpyxl reads anything;
  - openpyxl reads in read-only mode without external links; formulas are never evaluated and macros never run;
  - rows and columns are read only up to the dataset limits.

Formula policy: a formula cell is imported as the value Excel last saved for it (read with data_only=True).
That value was calculated by Excel, not here, so the dataset records which columns (and which preview cells)
hold such values, and the warnings, the model prompt, analysis notes and exports say so. A formula without
a saved value, and any Excel error value (#DIV/0!, #N/A, ...), is imported as a missing value.
"""

from __future__ import annotations

import csv
import io
import warnings
import zipfile
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time
from typing import Any

from openpyxl import load_workbook

from app.datasets import parsing
from app.datasets.parsing import DatasetError, ParsedCsv, parse_csv
from app.models import DatasetSource, WorkbookSheet

XLSX_SUFFIX = ".xlsx"
ZIP_SIGNATURE = b"PK\x03\x04"
OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # encrypted .xlsx and legacy .xls use this container
MAX_ZIP_ENTRIES = 2_000
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200  # per member larger than 1 MB
MAX_SHEETS_LISTED = 100
MAX_SHEET_NAME_LENGTH = 100
ERROR_VALUES = frozenset(
    {"#NULL!", "#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#NUM!", "#N/A", "#GETTING_DATA", "#SPILL!", "#CALC!"}
)


def is_xlsx_name(filename: str) -> bool:
    return filename.lower().endswith(XLSX_SUFFIX)


def check_container(data: bytes) -> None:
    """Reject anything that is not a plain, reasonably sized .xlsx package. Raises DatasetError."""
    if not data:
        raise DatasetError("The file is empty.")
    if data.startswith(OLE_SIGNATURE):
        raise DatasetError(
            "This workbook is password-protected (encrypted) or an old .xls file. Save it as a normal .xlsx "
            "workbook without a password and upload it again."
        )
    if not data.startswith(ZIP_SIGNATURE):
        raise DatasetError("This is not a valid .xlsx workbook.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        entries = archive.infolist()
    except (zipfile.BadZipFile, ValueError):
        raise DatasetError("This .xlsx workbook is damaged and cannot be read.") from None
    if len(entries) > MAX_ZIP_ENTRIES:
        raise DatasetError("This workbook has too many internal parts to be read safely.")
    if any(info.flag_bits & 0x1 for info in entries):
        raise DatasetError("This workbook is encrypted. Save it without a password and upload it again.")
    if sum(info.file_size for info in entries) > MAX_UNCOMPRESSED_BYTES or any(
        info.file_size > 1024 * 1024 and info.file_size > MAX_COMPRESSION_RATIO * max(info.compress_size, 1)
        for info in entries
    ):
        raise DatasetError("This workbook expands to too much data to be read safely.")
    names = {info.filename for info in entries}
    if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
        raise DatasetError("This is not a valid .xlsx workbook (it may be another file type renamed to .xlsx).")
    content_types = archive.read("[Content_Types].xml")
    if "xl/vbaProject.bin" in names or b"macroEnabled" in content_types:
        raise DatasetError("Macro-enabled workbooks are not supported. Save the data as a normal .xlsx workbook.")


def _open(data: bytes, data_only: bool) -> Any:
    try:
        with warnings.catch_warnings():  # openpyxl warns about unsupported extensions (data validation etc.)
            warnings.simplefilter("ignore")
            return load_workbook(io.BytesIO(data), read_only=True, data_only=data_only, keep_links=False)
    except Exception:  # openpyxl raises many kinds of errors for damaged files
        raise DatasetError("This .xlsx workbook is damaged or uses features that cannot be read.") from None


def list_sheets(data: bytes) -> list[WorkbookSheet]:
    """The workbook's worksheets (chart sheets are not data and are left out), with visibility and size."""
    check_container(data)
    workbook = _open(data, data_only=False)
    try:
        sheets = [
            WorkbookSheet(
                index=index,
                name=str(sheet.title)[:MAX_SHEET_NAME_LENGTH],
                state=sheet.sheet_state if sheet.sheet_state in ("hidden", "veryHidden") else "visible",
                rows=sheet.max_row,
                columns=sheet.max_column,
            )
            for index, sheet in enumerate(workbook.worksheets[:MAX_SHEETS_LISTED])
        ]
    except Exception:
        raise DatasetError("The worksheets of this workbook could not be read.") from None
    finally:
        workbook.close()
    if not sheets:
        raise DatasetError("This workbook has no worksheets with data.")
    return sheets


def import_sheet(data: bytes, sheet_index: int) -> tuple[ParsedCsv, DatasetSource]:
    """Convert one worksheet and validate it like an uploaded CSV. Raises DatasetError."""
    check_container(data)
    formulas = _open(data, data_only=False)
    values = _open(data, data_only=True)
    try:
        if not 0 <= sheet_index < min(len(formulas.worksheets), MAX_SHEETS_LISTED):
            raise DatasetError("That worksheet does not exist in this workbook.")
        name = str(formulas.worksheets[sheet_index].title)[:MAX_SHEET_NAME_LENGTH]
        grid = _read_grid(formulas.worksheets[sheet_index], values.worksheets[sheet_index], name)
    except DatasetError:
        raise
    except Exception:
        raise DatasetError(f"Worksheet {name!r} could not be read.") from None
    finally:
        formulas.close()
        values.close()
    try:
        parsed = parse_csv(grid.csv_text().encode("utf-8"), delimiter=",")
    except DatasetError as exc:  # the shared CSV checks speak of "the file"; name the worksheet instead
        message = str(exc).replace(
            "The file has a header row but no data rows.",
            "the first row holds the column names, but there are no data rows.",
        )
        raise DatasetError(f"Worksheet {name!r}: {message}") from None
    column_names = [c.name for c in parsed.columns]
    formula_cells = {column_names[j]: n for j, n in sorted(grid.formula_columns.items()) if j < len(column_names)}
    source = DatasetSource(
        workbook="",  # set by the caller (cleaned upload name)
        sheet=name,
        formula_cells=formula_cells,
        preview_formula_cells=sorted(grid.preview_formula_cells),
        uncached_formula_cells=grid.uncached,
        error_cells=grid.errors,
    )
    return replace(parsed, warnings=parsed.warnings + grid.notes(formula_cells)), source


class _Grid:
    """The worksheet's used cells as text rows, plus what was found about formulas and errors."""

    def __init__(self) -> None:
        self.rows: list[list[str]] = []
        self.formula_columns: Counter[int] = Counter()
        self.preview_formula_cells: set[tuple[int, int]] = set()
        self.uncached = 0
        self.errors = 0
        self.skipped_empty = 0

    def csv_text(self) -> str:
        width = max((len(row) for row in self.rows), default=0)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        for row in self.rows:
            writer.writerow(row + [""] * (width - len(row)))
        return buffer.getvalue()

    def notes(self, formula_cells: dict[str, int]) -> list[str]:
        notes = []
        if formula_cells:
            columns = ", ".join(list(formula_cells)[:10])
            notes.append(
                f"{sum(formula_cells.values()):,} cell(s) in {columns} hold results of Excel formulas: the values "
                "Excel last saved were imported as data. They were not recalculated here."
            )
        if self.uncached:
            notes.append(f"{self.uncached:,} formula cell(s) had no saved value and were imported as missing.")
        if self.errors:
            notes.append(f"{self.errors:,} Excel error value(s) such as #DIV/0! were imported as missing.")
        if self.skipped_empty:
            notes.append(f"{self.skipped_empty:,} empty row(s) were skipped.")
        return notes


def _read_grid(formula_sheet: Any, value_sheet: Any, name: str) -> _Grid:
    """Read used rows (bounded by the dataset limits) from the formula view and the saved-value view."""
    grid = _Grid()
    max_columns, max_rows = parsing.MAX_COLUMNS, parsing.MAX_ROWS
    formula_rows = formula_sheet.iter_rows(max_col=max_columns + 1)
    value_rows = value_sheet.iter_rows(max_col=max_columns + 1, values_only=True)
    for cells, saved in zip(formula_rows, value_rows, strict=False):
        data_row = len(grid.rows) - 1  # index among data rows once the header is in
        texts = [
            _cell_text(cell, saved[j] if j < len(saved) else None, j, data_row, grid) for j, cell in enumerate(cells)
        ]
        while texts and texts[-1] == "":
            texts.pop()
        if not texts:
            if grid.rows:
                grid.skipped_empty += 1
            continue
        if len(texts) > max_columns:
            raise DatasetError(f"Worksheet {name!r} has more than {max_columns} columns (the limit).")
        if len(grid.rows) > max_rows:  # header + max_rows data rows already read
            raise DatasetError(f"Worksheet {name!r} has more than {max_rows:,} data rows (the limit).")
        grid.rows.append(texts)
    if not grid.rows:
        raise DatasetError(f"Worksheet {name!r} is empty.")
    if not any(cell.strip() for cell in grid.rows[0]):
        raise DatasetError(f"Worksheet {name!r} has no column names in its first row.")
    return grid


def _cell_text(cell: Any, saved: Any, column: int, data_row: int, grid: _Grid) -> str:
    kind = getattr(cell, "data_type", None)
    if kind == "f":  # a formula: never evaluated; use the value Excel saved, if any
        if saved is None:
            grid.uncached += 1
            return ""
        if isinstance(saved, str) and saved in ERROR_VALUES:
            grid.errors += 1
            return ""
        if data_row >= 0:
            grid.formula_columns[column] += 1
            if data_row < parsing.PREVIEW_ROWS:
                grid.preview_formula_cells.add((data_row, column))
        return _text(saved)
    if kind == "e":
        grid.errors += 1
        return ""
    return _text(getattr(cell, "value", None))


def _text(value: Any) -> str:
    """One cell as CSV text that parse_csv types correctly."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return ""
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else repr(value)
    if isinstance(value, datetime):
        value = value.replace(tzinfo=None)
        return value.date().isoformat() if value.time() == time() else value.isoformat(timespec="seconds")
    if isinstance(value, date | time):
        return value.isoformat()
    return str(value)
