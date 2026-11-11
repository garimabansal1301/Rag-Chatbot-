"""Deterministic, allowlisted pandas analysis of one stored CSV dataset (no LLM involved here).

Safety rules:
  - Only the session's own stored `data.csv` is read, through DatasetStore (which checks ownership).
  - Only the operations below exist; operators and aggregations are fixed allowlists, columns must exist in
    the dataset's metadata, and filter values are plain text converted per column type. Nothing is ever
    evaluated as Python, SQL, a regex or a pandas expression.
  - Every result is bounded (rows, columns, cell length) before it reaches the model or the UI.
  - Analysis never writes: the dataset files are only opened for reading.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pandas.api import types as pdt
from pydantic import BaseModel, Field, ValidationError

from app.datasets.parsing import MISSING_MARKERS
from app.datasets.store import DatasetNotFoundError, DatasetStore
from app.models import AnalysisTable, ChartSpec, DatasetColumn, DatasetInfo, ResultColumnType

# ---------------------------------------------------------------- limits (one place)
MAX_FILTERS = 5
MAX_GROUP_BY = 3
MAX_AGGREGATIONS = 5
MAX_SORT_KEYS = 3
MAX_RESULT_COLUMNS = 20
DEFAULT_LIMIT = 20
MAX_LIMIT = 100  # top-N / rows returned and shown
MAX_SUMMARY_ROWS = 100  # columns listed in a summary table
MAX_CELL_CHARS = 200
MAX_VALUE_CHARS = 200  # filter values
MODEL_ROWS = 20  # result rows passed to the model (the UI shows up to MAX_LIMIT)
MAX_MODEL_CHARS = 6000
MAX_PIE_SLICES = 12  # beyond this, the smallest slices are combined into one labelled "Other" slice
_ADDITIVE = frozenset({"sum", "count"})  # only these add up to a meaningful whole (pie charts)

Operator = Literal[
    "equals",
    "not_equals",
    "greater_than",
    "greater_or_equal",
    "less_than",
    "less_or_equal",
    "contains",
    "is_missing",
    "is_not_missing",
]
AggFunction = Literal["sum", "mean", "median", "min", "max", "count", "nunique"]
NUMERIC = frozenset({"integer", "decimal"})
_COMPARISONS = {
    "greater_than": "gt",
    "greater_or_equal": "ge",
    "less_than": "lt",
    "less_or_equal": "le",
}
_NUMERIC_ONLY = frozenset({"sum", "mean", "median"})
_ORDERED = frozenset({"min", "max"})  # numbers or dates
_TRUE, _FALSE = frozenset({"true", "yes", "1"}), frozenset({"false", "no", "0"})


class AnalysisError(ValueError):
    """The request cannot be answered as asked. The message is safe to show (no paths, no internals)."""


# ---------------------------------------------------------------- validated inputs
class Filter(BaseModel):
    column: str = Field(max_length=200, description="Column name exactly as in the dataset")
    operator: Operator
    value: str | None = Field(
        default=None,
        max_length=MAX_VALUE_CHARS,
        description="Value as text, e.g. '5000', '2024-01-31', 'Europe'; omit for is_missing / is_not_missing",
    )


class Aggregation(BaseModel):
    column: str = Field(max_length=200)
    function: AggFunction


class SortKey(BaseModel):
    column: str = Field(
        max_length=200, description="A dataset column, or a result column such as 'records' or 'sum_revenue'"
    )
    descending: bool = False


class ChartRequest(BaseModel):
    type: Literal["bar", "line", "pie"]
    value_column: str | None = Field(
        default=None,
        max_length=200,
        description="Result column to plot, e.g. 'sum_sales'; default: the first aggregation",
    )
    title: str | None = Field(default=None, max_length=120)


class Query(BaseModel):
    filters: list[Filter] = Field(default_factory=list, max_length=MAX_FILTERS)
    group_by: list[str] = Field(default_factory=list, max_length=MAX_GROUP_BY)
    group_by_month: str | None = Field(default=None, max_length=200)  # a date column, grouped by calendar month
    chart: ChartRequest | None = None
    aggregations: list[Aggregation] = Field(default_factory=list, max_length=MAX_AGGREGATIONS)
    sort_by: list[SortKey] = Field(default_factory=list, max_length=MAX_SORT_KEYS)
    columns: list[str] = Field(default_factory=list, max_length=MAX_RESULT_COLUMNS)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)


# ---------------------------------------------------------------- dataset resolution and loading
def resolve_dataset(store: DatasetStore, session_id: str, requested: str | None = None) -> DatasetInfo:
    """The dataset to analyse in this session: the one named, else the selected one, else the only one.

    `requested` is a file name (from the model); it is only matched against this session's own datasets.
    """
    try:
        datasets = store.list(session_id)
        selected = store.selected(session_id)
    except DatasetNotFoundError:
        datasets, selected = [], None
    if not datasets:
        raise AnalysisError(
            "No CSV dataset has been uploaded in this chat yet. Ask the user to upload one with the Datasets "
            "panel above the chat."
        )
    names = _list_names(d.filename for d in datasets)
    if requested and requested.strip():
        wanted = requested.strip().casefold()
        matches = [d for d in datasets if d.filename.casefold() in (wanted, f"{wanted}.csv")]
        if not matches:  # an imported worksheet may be named by its sheet or its workbook
            matches = [
                d for d in datasets if d.source and wanted in (d.source.sheet.casefold(), d.source.workbook.casefold())
            ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AnalysisError(
                f"Several datasets are named {requested[:100]!r}. Ask the user to select one in the Datasets panel."
            )
        raise AnalysisError(f"No dataset named {requested[:100]!r} in this chat. Datasets: {names}.")
    if selected is not None:
        return selected
    if len(datasets) == 1:
        return datasets[0]
    raise AnalysisError(
        f"This chat has several datasets ({names}) and none is selected. Ask the user which one to analyse, "
        "or to select one in the Datasets panel."
    )


def load_frame(store: DatasetStore, session_id: str, info: DatasetInfo) -> pd.DataFrame:
    """Read the stored CSV (read-only) and apply the column types detected at upload."""
    try:
        path = store.data_path(session_id, info.id)  # ownership check
        with path.open(encoding="utf-8", newline="") as handle:
            raw = pd.read_csv(
                handle,
                sep=info.delimiter,
                header=0,
                names=[c.name for c in info.columns],
                dtype=str,
                na_values=list(MISSING_MARKERS),
                keep_default_na=False,
                skip_blank_lines=True,
                on_bad_lines="error",
                engine="c",
            )
    except (DatasetNotFoundError, OSError, ValueError):
        raise AnalysisError("The dataset could not be read. Ask the user to upload it again.") from None
    return pd.DataFrame({c.name: _typed(raw[c.name], c.dtype) for c in info.columns})


def _typed(series: pd.Series, dtype: str) -> pd.Series:
    if dtype in NUMERIC:
        return pd.to_numeric(series, errors="coerce")
    if dtype == "boolean":
        return series.str.strip().str.lower().map({"true": True, "false": False})
    if dtype == "date":
        return pd.to_datetime(series, errors="coerce", format="ISO8601")
    return series


# ---------------------------------------------------------------- operations
@dataclass(frozen=True)
class ColumnProfile:
    """Raw per-column figures shared by the chat summary and the Excel report (no formatting here)."""

    name: str
    dtype: str
    missing: int
    missing_pct: float  # 0-100
    unique: int
    low: Any  # min, for numeric and date columns
    mean: Any  # numeric columns only
    high: Any  # max, for numeric and date columns
    quality: str  # "empty", "mostly missing", "single value" or ""


def column_profiles(frame: pd.DataFrame, info: DatasetInfo) -> list[ColumnProfile]:
    total = len(frame)
    profiles = []
    for column in info.columns:
        series = frame[column.name]
        missing = int(series.isna().sum())
        valid = series.dropna()
        unique = int(valid.nunique())
        low = high = mean = None
        if (column.dtype in NUMERIC or column.dtype == "date") and not valid.empty:
            low, high = valid.min(), valid.max()
        if column.dtype in NUMERIC and not valid.empty:
            mean = valid.mean()
        pct = missing / total * 100 if total else 0.0
        if valid.empty:
            quality = "empty"
        elif pct >= 50:
            quality = "mostly missing"
        elif unique == 1 and total > 1:
            quality = "single value"
        else:
            quality = ""
        profiles.append(ColumnProfile(column.name, column.dtype, missing, pct, unique, low, mean, high, quality))
    return profiles


def summarize(frame: pd.DataFrame, info: DatasetInfo) -> AnalysisTable:
    """Dimensions, types, missing values (count and %), basic statistics, duplicates and quality notes."""
    total = len(frame)
    profiles = column_profiles(frame, info)
    rows: list[list[str | None]] = [
        [
            p.name,
            p.dtype,
            str(p.missing),
            f"{p.missing_pct:.1f}%",
            str(p.unique),
            _cell(p.low),
            _cell(p.mean),
            _cell(p.high),
        ]
        for p in profiles
    ]
    duplicates = int(frame.duplicated().sum())
    notes = [f"{total:,} rows × {len(info.columns)} columns.", f"{duplicates:,} duplicate row(s)."]
    notes.extend(_formula_notes(info, [c.name for c in info.columns]))
    for quality, label in [
        ("empty", "Columns with no values: "),
        ("mostly missing", "Columns at least half missing: "),
        ("single value", "Columns with a single repeated value: "),
    ]:
        names = [p.name for p in profiles if p.quality == quality]
        if names:
            notes.append(label + _list_names(names) + ".")
    return AnalysisTable(
        title=f"Summary of {info.filename}",
        dataset=info.filename,
        columns=["column", "type", "missing", "missing %", "unique", "min", "mean", "max"],
        rows=rows[:MAX_SUMMARY_ROWS],
        total_rows=len(rows),
        notes=notes,
        column_types=["text", "text", "integer", "percent", "integer", "mixed", "mixed", "mixed"],
    )


def column_statistics(frame: pd.DataFrame, info: DatasetInfo, column_name: str) -> AnalysisTable:
    """Count, sum, mean, median, min, max and standard deviation of one numeric column."""
    column = _column(info, column_name)
    if column.dtype not in NUMERIC:
        raise AnalysisError(
            f"Column {column.name!r} is a {column.dtype} column; statistics need a numeric column (integer or "
            "decimal). For text, count values per group instead."
        )
    series = frame[column.name]
    valid = series.dropna()
    if valid.empty:
        raise AnalysisError(f"Column {column.name!r} has no values to calculate with.")
    stats: list[tuple[str, Any]] = [
        ("records", len(series)),
        ("non-missing values", len(valid)),
        ("missing values", int(series.isna().sum())),
        ("sum", valid.sum()),
        ("mean", valid.mean()),
        ("median", valid.median()),
        ("min", valid.min()),
        ("max", valid.max()),
    ]
    if len(valid) > 1:
        stats.append(("standard deviation (sample)", valid.std()))
    notes = []
    if len(valid) < len(series):
        notes.append(f"{len(series) - len(valid):,} missing value(s) were excluded from the calculations.")
    notes.extend(_formula_notes(info, [column.name]))
    return AnalysisTable(
        title=f"Statistics for {column.name}",
        dataset=info.filename,
        columns=["statistic", "value"],
        rows=[[name, _cell(value)] for name, value in stats],
        total_rows=len(stats),
        notes=notes,
        column_types=["text", "decimal"],
    )


def run_query(frame: pd.DataFrame, info: DatasetInfo, query: Query) -> AnalysisTable:
    """Filter, then either aggregate (optionally per group) or list rows; then sort and keep the top `limit`.

    With `query.chart`, a chart of the same computed result is attached (nothing is calculated twice).
    """
    data, notes = frame, []
    for condition in query.filters:
        data = data[_mask(data, _column(info, condition.column), condition)]
    if query.filters:
        notes.append(f"{len(data):,} of {len(frame):,} rows matched the filters.")
    keys: list[str] = []
    functions: dict[str, str] = {}
    if query.group_by or query.aggregations or query.group_by_month:
        result, title, keys, functions = _aggregate(data, info, query, notes)
        if query.sort_by:  # by group or result columns, e.g. 'records' or 'sum_revenue'
            result = _sort(result, query.sort_by)
    else:
        result, title = _rows(data, info, query, notes)  # sorts on dataset columns itself
    chart = _chart(result, query, keys, functions) if query.chart else None
    notes.extend(_formula_notes(info, _query_columns(info, query)))
    shown = result.head(query.limit)
    return AnalysisTable(
        title=title,
        dataset=info.filename,
        columns=[str(c) for c in shown.columns],
        rows=[[_cell(v) for v in row] for row in shown.itertuples(index=False, name=None)],
        total_rows=len(result),
        notes=notes[:10],
        chart=chart,
        column_types=[_result_type(shown[c]) for c in shown.columns],
    )


def _aggregate(
    data: pd.DataFrame, info: DatasetInfo, query: Query, notes: list[str]
) -> tuple[pd.DataFrame, str, list[str], dict[str, str]]:
    """(result, title, group key columns, function per output column)."""
    keys = list(dict.fromkeys(_column(info, name).name for name in query.group_by))
    if query.group_by_month:
        data, month_key = _add_month(data, info, query.group_by_month, keys, notes)
        keys.append(month_key)
    if len(keys) > MAX_GROUP_BY:
        raise AnalysisError(f"Group by at most {MAX_GROUP_BY} columns (counting the month).")
    specs: list[tuple[str, str, str]] = []  # (column, function, output name)
    for aggregation in query.aggregations:
        column = _column(info, aggregation.column)
        _check_aggregation(column, aggregation.function)
        output = f"{aggregation.function}_{column.name}"
        if output not in {s[2] for s in specs}:
            specs.append((column.name, aggregation.function, output))
            missing = int(data[column.name].isna().sum())
            if missing:
                notes.append(f"{missing:,} missing value(s) in {column.name!r} were excluded from {output}.")
    if keys:
        grouped = data.groupby(keys, dropna=False, sort=True)
        result = grouped.size().rename("records").to_frame()
        for column_name, function, output in specs:
            result[output] = grouped[column_name].agg(function)
        result = result.reset_index()
        title = "Grouped by " + ", ".join(keys)
    else:
        values: dict[str, Any] = {"records": len(data)}
        for column_name, function, output in specs:
            values[output] = data[column_name].agg(function) if len(data) else None
        result = pd.DataFrame([values])
        title = "Totals"
    notes.append("'records' counts rows; count_<column> counts non-missing values.")
    functions = {output: function for _, function, output in specs} | {"records": "count"}
    return result, title, keys, functions


def _add_month(
    data: pd.DataFrame, info: DatasetInfo, column_name: str, keys: list[str], notes: list[str]
) -> tuple[pd.DataFrame, str]:
    """Add a calendar-month key (a pandas Period, so months sort chronologically) for a date column.

    Policy: the column must be a date column (detected at upload; values are ISO dates). Rows whose date is
    missing are left out of the monthly groups, and the number left out is reported.
    """
    column = _column(info, column_name)
    if column.dtype != "date":
        dates = [c.name for c in info.columns if c.dtype == "date"]
        hint = f"Date columns: {_list_names(dates)}." if dates else "This dataset has no date column."
        raise AnalysisError(f"{column.name!r} is not a date column, so it cannot be grouped by month. {hint}")
    key = "month" if "month" not in {c.name for c in info.columns} | set(keys) else f"month of {column.name}"
    undated = int(data[column.name].isna().sum())
    if undated:
        notes.append(f"{undated:,} row(s) without a {column.name!r} date were left out of the monthly groups.")
    data = data[data[column.name].notna()]
    return data.assign(**{key: data[column.name].dt.to_period("M")}), key


def _chart(result: pd.DataFrame, query: Query, keys: list[str], functions: dict[str, str]) -> ChartSpec:
    """A chart of one numeric result column per group, from the computed result (never from model text)."""
    request = query.chart
    assert request is not None
    if not keys:
        raise AnalysisError(
            "A chart needs grouped results: add group_by (or group_by_month) with an aggregation, "
            "e.g. total sales by category."
        )
    if len(keys) > 1:
        raise AnalysisError("A chart can show one grouping at a time; group by a single column (or by month).")
    label = keys[0]
    lookup = {str(c).casefold(): str(c) for c in result.columns if str(c) != label}
    wanted = request.value_column or next((o for o in functions if o != "records"), "records")
    value = wanted if wanted in lookup.values() else lookup.get(wanted.strip().casefold())
    if value is None:
        raise AnalysisError(
            f"Cannot chart {wanted[:100]!r}: chartable result columns are {_list_names(lookup.values())}."
        )
    series = result[value]
    if not pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series):
        raise AnalysisError(f"Cannot chart {value!r}: a chart needs numbers.")
    points = result[[label, value]][series.notna()]
    notes = []
    if len(points) < len(result):
        notes.append(f"{len(result) - len(points)} group(s) with no {value} value are not in the chart.")
    if points.empty:
        raise AnalysisError(f"There are no {value} values to chart.")
    title = request.title or f"{value} by {label}"
    if request.type == "pie":
        labels, values, percentages = _pie(points, label, value, functions.get(value, ""), notes)
    else:
        if len(points) > query.limit:
            notes.append(f"The chart shows the first {query.limit} of {len(points)} groups.")
        points = points.head(query.limit)
        labels, values, percentages = [_label(v) for v in points[label]], [_number(v) for v in points[value]], None
    try:
        return ChartSpec(
            type=request.type,
            title=title,
            label_name=label,
            series_name=value,
            labels=labels,
            values=values,
            percentages=percentages,
            notes=notes,
        )
    except ValidationError:
        raise AnalysisError("This result cannot be shown as that chart; show the table instead.") from None


def _pie(
    points: pd.DataFrame, label: str, value: str, function: str, notes: list[str]
) -> tuple[list[str], list[float], list[float]]:
    """Slices for parts of a whole: only sums or counts, no negatives; small categories merged into 'Other'."""
    if function not in _ADDITIVE:
        raise AnalysisError(
            f"A pie chart shows parts of a whole, so it needs a sum or a count; {value!r} is a {function or 'value'} "
            "per group. Use a bar chart instead."
        )
    if bool((points[value] < 0).any()):
        raise AnalysisError(f"{value!r} has negative values, which a pie chart cannot show. Use a bar chart instead.")
    total = float(points[value].sum())
    if total <= 0:
        raise AnalysisError(f"The {value!r} values add up to zero, so there is nothing to divide into a pie chart.")
    ordered = points.sort_values(value, ascending=False, kind="stable")
    labels = [_label(v) for v in ordered[label]]
    values = [_number(v) for v in ordered[value]]
    if len(values) > MAX_PIE_SLICES:  # merging is valid because sums and counts add up
        rest = len(values) - (MAX_PIE_SLICES - 1)
        labels = labels[: MAX_PIE_SLICES - 1] + [f"Other ({rest} categories)"]
        values = values[: MAX_PIE_SLICES - 1] + [_number(sum(values[MAX_PIE_SLICES - 1 :]))]
        notes.append(f"The {rest} smallest categories are combined into 'Other'.")
    return labels, values, [round(v / total * 100, 2) for v in values]


def _result_type(series: pd.Series) -> ResultColumnType:
    """How a result column should be written to a spreadsheet (month periods stay text, e.g. "2024-01")."""
    if pdt.is_bool_dtype(series):
        return "boolean"
    if pdt.is_integer_dtype(series):
        return "integer"
    if pdt.is_float_dtype(series):
        return "decimal"
    if pdt.is_datetime64_any_dtype(series):
        return "date"
    values = series.dropna()
    if len(values) and all(isinstance(v, bool) for v in values):
        return "boolean"
    return "text"


def _formula_notes(info: DatasetInfo, used: list[str]) -> list[str]:
    """A note when used columns hold values that Excel formulas computed (imported, not recalculated here)."""
    if info.source is None or not info.source.formula_cells:
        return []
    columns = [name for name in dict.fromkeys(used) if name in info.source.formula_cells]
    if not columns:
        return []
    return [
        f"{_list_names(columns)} include values saved by Excel formulas in the workbook; they were imported as "
        "data, not recalculated."
    ]


def _query_columns(info: DatasetInfo, query: Query) -> list[str]:
    """Dataset columns a query reads (names already validated while running it)."""
    names = [f.column for f in query.filters] + query.group_by + [a.column for a in query.aggregations]
    names += [s.column for s in query.sort_by] + query.columns
    if query.group_by_month:
        names.append(query.group_by_month)
    if not (query.group_by or query.aggregations or query.group_by_month) and not query.columns:
        names += [c.name for c in info.columns]  # a plain row listing shows every column
    lookup = {c.name.casefold(): c.name for c in info.columns}
    return [lookup[n.strip().casefold()] for n in names if n.strip().casefold() in lookup]


def _number(value: Any) -> float:
    """A chart value without float noise (e.g. 6299.929999999999 -> 6299.93); 12 significant digits."""
    return float(f"{float(value):.12g}")


def _label(value: Any) -> str:
    text = _cell(value)
    return "(missing)" if text is None else text


def _rows(data: pd.DataFrame, info: DatasetInfo, query: Query, notes: list[str]) -> tuple[pd.DataFrame, str]:
    names = [_column(info, name).name for name in query.columns] or [c.name for c in info.columns]
    names = list(dict.fromkeys(names))
    if len(names) > MAX_RESULT_COLUMNS:
        notes.append(f"Only the first {MAX_RESULT_COLUMNS} of {len(names)} columns are included.")
        names = names[:MAX_RESULT_COLUMNS]
    if query.sort_by:  # on dataset columns, before narrowing to the requested columns
        data = _sort(
            data, [SortKey(column=_column(info, k.column).name, descending=k.descending) for k in query.sort_by]
        )
    return data[names], "Matching rows" if query.filters else "Rows"


def _sort(data: pd.DataFrame, keys: list[SortKey]) -> pd.DataFrame:
    by, ascending = [], []
    lookup = {str(c).casefold(): str(c) for c in data.columns}
    for key in keys:
        name = key.column if key.column in data.columns else lookup.get(key.column.strip().casefold())
        if name is None:
            raise AnalysisError(
                f"Cannot sort by {key.column[:100]!r}: available columns are {_list_names(data.columns)}."
            )
        by.append(name)
        ascending.append(not key.descending)
    return data.sort_values(by=by, ascending=ascending, na_position="last", kind="stable")


def _check_aggregation(column: DatasetColumn, function: str) -> None:
    if function in _NUMERIC_ONLY and column.dtype not in NUMERIC:
        raise AnalysisError(
            f"Cannot calculate {function} of {column.name!r}: it is a {column.dtype} column and {function} needs "
            "numbers. Use count or nunique for text."
        )
    if function in _ORDERED and column.dtype not in NUMERIC | {"date"}:
        raise AnalysisError(f"Cannot calculate {function} of {column.name!r}: it needs a numeric or date column.")


def _mask(data: pd.DataFrame, column: DatasetColumn, condition: Filter) -> pd.Series:
    series, operator = data[column.name], condition.operator
    if operator == "is_missing":
        return series.isna()
    if operator == "is_not_missing":
        return series.notna()
    if condition.value is None or not condition.value.strip():
        raise AnalysisError(f"The {operator} filter on {column.name!r} needs a value.")
    if operator == "contains":
        if column.dtype != "text":
            raise AnalysisError(f"'contains' works on text columns; {column.name!r} is a {column.dtype} column.")
        return series.str.contains(condition.value.strip(), case=False, regex=False, na=False).astype(bool)
    value = _convert(column, condition.value)
    if operator in ("equals", "not_equals"):
        if column.dtype == "text":
            equal = series.str.strip().str.casefold() == str(value).casefold()
        else:
            equal = series == value
        equal = equal.fillna(False).astype(bool)
        return equal if operator == "equals" else (~equal & series.notna())
    if column.dtype not in NUMERIC | {"date"}:
        raise AnalysisError(
            f"{operator} needs a numeric or date column; {column.name!r} is a {column.dtype} column. "
            "Use equals or contains for text."
        )
    compare: Callable[[Any], pd.Series] = getattr(series, _COMPARISONS[operator])
    return compare(value).fillna(False).astype(bool)


def _convert(column: DatasetColumn, raw: str) -> Any:
    """A filter value as the column's type; never evaluated."""
    text = raw.strip()
    if column.dtype in NUMERIC:
        try:
            number = float(text)
        except ValueError:
            raise AnalysisError(f"{text[:50]!r} is not a number, but {column.name!r} is numeric.") from None
        if not math.isfinite(number):
            raise AnalysisError(f"{text[:50]!r} is not a usable number.")
        return number
    if column.dtype == "date":
        try:
            return pd.Timestamp(text)
        except ValueError:
            raise AnalysisError(f"{text[:50]!r} is not a date; use YYYY-MM-DD for {column.name!r}.") from None
    if column.dtype == "boolean":
        if text.lower() in _TRUE | _FALSE:
            return text.lower() in _TRUE
        raise AnalysisError(f"{column.name!r} holds true/false values; use 'true' or 'false'.")
    return text


def _column(info: DatasetInfo, name: str) -> DatasetColumn:
    """A column from the dataset's own metadata (exact name, or a unique case-insensitive match)."""
    for column in info.columns:
        if column.name == name:
            return column
    matches = [c for c in info.columns if c.name.casefold() == name.strip().casefold()]
    if len(matches) == 1:
        return matches[0]
    raise AnalysisError(
        f"Unknown column {name[:100]!r}. Available columns: {_list_names(c.name for c in info.columns)}."
    )


# ---------------------------------------------------------------- formatting and model payload
def _cell(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat() if value == value.normalize() else value.isoformat()
    if hasattr(value, "item") and not isinstance(value, str):  # numpy scalar -> Python scalar
        value = value.item()
        if isinstance(value, bool):
            return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.10g}"
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 1] + "…"


def _list_names(names: Any, limit: int = 15) -> str:
    items = [str(n)[:60] for n in names]
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def to_model_payload(table: AnalysisTable) -> str:
    """A compact JSON view of the result for the model: bounded rows and characters."""
    payload: dict[str, Any] = {
        "dataset": table.dataset,
        "title": table.title,
        "columns": table.columns,
        "rows": table.rows[:MODEL_ROWS],
        "total_result_rows": table.total_rows,
        "rows_shown_to_user": len(table.rows),
        "notes": table.notes,
    }
    if table.chart is not None:  # summary only; the chart itself is drawn from the validated spec in the UI
        chart = table.chart
        payload["chart_shown_to_user"] = {
            "type": chart.type,
            "title": chart.title,
            "points": len(chart.values),
            "notes": chart.notes,
        }
        if chart.percentages is not None:
            payload["chart_shown_to_user"]["percent_of_total"] = dict(
                list(zip(chart.labels, chart.percentages, strict=True))[:MODEL_ROWS]
            )
    text = json.dumps(payload, ensure_ascii=False)
    while len(text) > MAX_MODEL_CHARS and payload["rows"]:
        payload["rows"] = payload["rows"][: len(payload["rows"]) // 2]
        text = json.dumps(payload, ensure_ascii=False)
    if len(payload["rows"]) < min(len(table.rows), MODEL_ROWS):
        payload["rows_included_here"] = len(payload["rows"])
        text = json.dumps(payload, ensure_ascii=False)
    return text[:MAX_MODEL_CHARS]
