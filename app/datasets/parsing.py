"""Validate an uploaded CSV and describe it: size, encoding, structure, column types and a short preview.

The file's bytes are treated as untrusted data. Nothing in it is executed; the whole file is never sent
to the LLM. The first row is always the header.

Two parsers, each for what it is good at:
  - Python's `csv` module (strict mode) checks the structure row by row: quoting, rows with more values
    than the header, row/column limits, and keeps the first rows exactly as written for the preview.
  - pandas infers the column types and counts missing values.
"""

from __future__ import annotations

import codecs
import csv
import io
import re
import warnings
from dataclasses import dataclass, field

import pandas as pd
from pandas.api import types as pdt

from app.models import DatasetColumn, DatasetDtype

MAX_ROWS = 100_000
MAX_COLUMNS = 200
PREVIEW_ROWS = 10
MAX_COLUMN_NAME_LENGTH = 100
MAX_PREVIEW_CELL_LENGTH = 200
DELIMITERS = ",;\t|"
# Written in the file to mean "no value"; used both for the preview and for pandas.
MISSING_MARKERS = ("", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "#N/A")
_MISSING = frozenset(MISSING_MARKERS)
# A cell starting with one of these can run as a formula when a CSV is opened in a spreadsheet.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_FORMULA_TRIGGERS = ("=", "+", "-", "@")
# Leading characters some spreadsheet apps skip before reading a formula: whitespace and control characters.
_LEADING_IGNORED = re.compile(r"^[\s\x00-\x1f\x7f]+")
_PLAIN_NUMBER = re.compile(r"^[+-]?(\d+([.,]\d+)*|[.,]\d+)([eE][+-]?\d+)?%?$")


class DatasetError(ValueError):
    """The upload is not a usable CSV. The message is meant for the user."""


@dataclass(frozen=True)
class ParsedCsv:
    text: str  # the decoded file, stored as UTF-8
    encoding: str
    delimiter: str
    rows: int
    columns: list[DatasetColumn]
    preview: list[list[str | None]]
    warnings: list[str] = field(default_factory=list)
    formula_like_cells: int = 0


def is_formula_like(value: str) -> bool:
    """True for text a spreadsheet could run as a formula; plain numbers such as "-5" are fine.

    Formula-like: starting with tab or CR, or with =, + , - or @ once leading whitespace and control
    characters are skipped (e.g. "  =1+1", "\\x01=cmd").
    """
    if not value:
        return False
    if value[0] in ("\t", "\r"):
        return True
    rest = _LEADING_IGNORED.sub("", value)
    return rest.startswith(_FORMULA_TRIGGERS) and not _PLAIN_NUMBER.match(rest.strip())


def neutralize_formula(value: str) -> str:
    """Prefix formula-like text with an apostrophe so spreadsheets show it as text. Use for every export."""
    return "'" + value if is_formula_like(value) else value


def parse_csv(data: bytes, delimiter: str | None = None) -> ParsedCsv:
    """IN: the uploaded bytes (and the delimiter when it is known, e.g. for converted Excel sheets).
    OUT: ParsedCsv.  Raises DatasetError with a user-facing reason."""
    text, encoding = _decode(data)
    delimiter = delimiter or _sniff_delimiter(text)
    header, preview, rows, formula_cells = _read_structure(text, delimiter)
    names, name_warnings = _column_names(header)
    columns = _describe_columns(text, delimiter, names, rows)
    notes = list(name_warnings)
    if formula_cells:
        notes.append(
            f"{formula_cells} cell(s) start with =, +, - or @. They are kept as plain text and will be "
            "neutralised if the data is ever exported."
        )
    return ParsedCsv(text, encoding, delimiter, rows, columns, preview, notes, formula_cells)


def _decode(data: bytes) -> tuple[str, str]:
    if not data.strip():
        raise DatasetError("The file is empty.")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            text, encoding = data.decode("utf-16"), "UTF-16"
        except UnicodeDecodeError:
            raise DatasetError("The file says it is UTF-16 but is not valid UTF-16 text.") from None
    else:
        if b"\x00" in data:
            raise DatasetError("This looks like a binary file, not a CSV text file.")
        try:
            text, encoding = data.decode("utf-8-sig"), "UTF-8"  # also removes a UTF-8 BOM
        except UnicodeDecodeError:
            raise DatasetError(
                "Unsupported text encoding: the file is not UTF-8. Save it as 'CSV UTF-8' and upload it again."
            ) from None
    if "\x00" in text:
        raise DatasetError("This looks like a binary file, not a CSV text file.")
    if not text.strip():
        raise DatasetError("The file is empty.")
    return text, encoding


def _sniff_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:64_000], delimiters=DELIMITERS).delimiter
    except csv.Error:
        return ","  # e.g. a single-column file


def _read_structure(text: str, delimiter: str) -> tuple[list[str], list[list[str | None]], int, int]:
    """(header, preview rows, number of data rows, formula-like cells); raises DatasetError if malformed."""
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    header: list[str] | None = None
    preview: list[list[str | None]] = []
    rows = formula_cells = 0
    try:
        for record in reader:
            if not record:
                continue  # an empty line (pandas skips these too); ",," is a row of missing values
            if header is None:
                header = _check_header(record)
                continue
            rows += 1
            _check_row(record, len(header), rows, reader.line_num)
            formula_cells += sum(is_formula_like(cell) for cell in record)
            if len(preview) < PREVIEW_ROWS:
                padded = record + [""] * (len(header) - len(record))
                preview.append([_preview_cell(cell) for cell in padded])
    except csv.Error as exc:  # e.g. a quote that is never closed, or a field over the csv size limit
        raise DatasetError(f"Malformed CSV near line {reader.line_num}: {exc}.") from None
    if header is None:
        raise DatasetError("The file is empty.")
    if rows == 0:
        raise DatasetError("The file has a header row but no data rows.")
    return header, preview, rows, formula_cells


def _check_header(header: list[str]) -> list[str]:
    if len(header) > MAX_COLUMNS:
        raise DatasetError(f"Too many columns: {len(header)} (the limit is {MAX_COLUMNS}).")
    return header


def _check_row(record: list[str], columns: int, rows: int, line: int) -> None:
    if len(record) > columns:
        raise DatasetError(
            f"Malformed CSV: line {line} has {len(record)} values, but the header has {columns} columns."
        )
    if rows > MAX_ROWS:
        raise DatasetError(f"Too many rows: more than {MAX_ROWS:,} (the limit is {MAX_ROWS:,}).")


def _preview_cell(cell: str) -> str | None:
    if cell.strip() in _MISSING:
        return None
    return cell if len(cell) <= MAX_PREVIEW_CELL_LENGTH else cell[: MAX_PREVIEW_CELL_LENGTH - 1] + "…"


def _column_names(header: list[str]) -> tuple[list[str], list[str]]:
    """Clean header names, name blank ones column_N and make duplicates unique (case-insensitive)."""
    names: list[str] = []
    seen: set[str] = set()
    blank, renamed = 0, []
    for position, raw in enumerate(header, start=1):
        name = " ".join(raw.split())[:MAX_COLUMN_NAME_LENGTH]
        if not name:
            blank += 1
            name = f"column_{position}"
        unique, suffix = name, 2
        while unique.lower() in seen:
            unique, suffix = f"{name}_{suffix}", suffix + 1
        if unique != name:
            renamed.append(f"{name} → {unique}")
        seen.add(unique.lower())
        names.append(unique)
    notes = []
    if blank:
        notes.append(f"{blank} column(s) had no name and were named column_N by position.")
    if renamed:
        notes.append("Duplicate column names were made unique: " + ", ".join(renamed) + ".")
    return names, notes


def _describe_columns(text: str, delimiter: str, names: list[str], rows: int) -> list[DatasetColumn]:
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            header=0,
            names=names,
            na_values=list(MISSING_MARKERS),
            keep_default_na=False,
            skip_blank_lines=True,
            on_bad_lines="error",
            engine="c",
            low_memory=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DatasetError(f"Malformed CSV: {exc}") from None
    if len(frame.columns) != len(names) or len(frame) != rows:
        raise DatasetError("Malformed CSV: rows could not be read consistently.")
    return [DatasetColumn(name=name, dtype=_kind(frame[name]), missing=int(frame[name].isna().sum())) for name in names]


def _kind(series: pd.Series) -> DatasetDtype:
    values = series.dropna()
    if values.empty:
        return "empty"
    if pdt.is_bool_dtype(series) or all(isinstance(v, bool) for v in values):
        return "boolean"
    if pdt.is_integer_dtype(series):
        return "integer"
    if pdt.is_float_dtype(series):
        return "integer" if bool((values % 1 == 0).all()) else "decimal"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(values.astype(str), errors="coerce", format="ISO8601")
    return "date" if bool(parsed.notna().all()) else "text"
