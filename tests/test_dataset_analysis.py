"""Phase 2: controlled pandas analysis of a chat's CSV dataset, through the LangGraph tool loop.

Deterministic service tests (no LLM), tool/graph tests with the scripted FakeChatModel, and UI/API tests.
Temp folders only; no network, no Gemini.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import deps
from app.chat.engine import get_engine
from app.config import Settings
from app.datasets.analysis import (
    MAX_LIMIT,
    MAX_MODEL_CHARS,
    MODEL_ROWS,
    AnalysisError,
    Query,
    column_statistics,
    load_frame,
    resolve_dataset,
    run_query,
    summarize,
    to_model_payload,
)
from app.datasets.parsing import parse_csv
from app.datasets.store import DATA_FILE, META_FILE, DatasetNotFoundError, DatasetStore
from app.deps import Components, get_components
from app.main import create_app
from app.models import AnalysisTable, DatasetInfo, Message
from app.tools import get_tools
from app.tools.dataset_tools import NO_DATASETS_PROMPT, dataset_context, make_dataset_tools
from tests.fakes import TOOL_TRIGGER, FakeChatModel, StaticRetriever, make_chunk

SESSION_A, SESSION_B = "a" * 32, "b" * 32
# 8 rows: one duplicate row (Alpha), missing sales (Delta) and missing units (Beta).
SALES_CSV = b"""product,category,region,sales,units,date,active
Alpha,Tools,North,1200.5,10,2024-01-05,true
Beta,Tools,South,5400,,2024-02-10,false
Gamma,Garden,North,7000.25,30,2024-03-01,true
Delta,Garden,East,,5,2024-03-15,true
Alpha,Tools,North,1200.5,10,2024-01-05,true
Eps,Home,South,9100,12,2024-04-20,false
Zeta-9000,Home,West,300,1,2024-05-01,false
Eta,Tools,East,5000,8,2024-06-30,true
"""


@dataclass
class Loaded:
    store: DatasetStore
    info: DatasetInfo
    frame: Any


def save(store: DatasetStore, session_id: str, data: bytes = SALES_CSV, name: str = "sales.csv") -> DatasetInfo:
    return store.save(session_id, name, parse_csv(data), len(data))


@pytest.fixture()
def store(tmp_path: Path) -> DatasetStore:
    return DatasetStore(tmp_path / "datasets")


@pytest.fixture()
def sales(store: DatasetStore) -> Loaded:
    info = save(store, SESSION_A)
    return Loaded(store, info, load_frame(store, SESSION_A, info))


def query(loaded: Loaded, **spec: Any) -> AnalysisTable:
    return run_query(loaded.frame, loaded.info, Query.model_validate(spec))


def column(table: AnalysisTable, name: str) -> list[str | None]:
    index = table.columns.index(name)
    return [row[index] for row in table.rows]


# ---------------------------------------------------------------- summary, statistics, data quality
def test_summary_dimensions_types_missing_and_duplicates(sales: Loaded) -> None:
    table = summarize(sales.frame, sales.info)
    assert table.title == "Summary of sales.csv" and table.total_rows == 7
    by_column = {row[0]: row for row in table.rows}
    assert by_column["sales"][1:5] == ["decimal", "1", "12.5%", "6"]
    assert by_column["units"][1:4] == ["integer", "1", "12.5%"]
    assert by_column["sales"][5:] == ["300", "4171.607143", "9100"]  # min, mean of the 7 values, max
    assert by_column["date"][5] == "2024-01-05" and by_column["date"][7] == "2024-06-30"
    assert "8 rows × 7 columns." in table.notes and "1 duplicate row(s)." in table.notes


def test_column_statistics_values(sales: Loaded) -> None:
    table = column_statistics(sales.frame, sales.info, "sales")
    stats = dict(zip(column(table, "statistic"), column(table, "value"), strict=True))
    values = [1200.5, 5400, 7000.25, 1200.5, 9100, 300, 5000]  # the non-missing sales
    assert stats["records"] == "8" and stats["non-missing values"] == "7" and stats["missing values"] == "1"
    assert float(stats["sum"] or 0) == pytest.approx(sum(values)) == 29201.25
    assert float(stats["mean"] or 0) == pytest.approx(statistics.mean(values))
    assert stats["median"] == "5000" and stats["min"] == "300" and stats["max"] == "9100"
    assert float(stats["standard deviation (sample)"] or 0) == pytest.approx(statistics.stdev(values))
    assert any("missing value(s) were excluded" in n for n in table.notes)


@pytest.mark.parametrize("name", ["product", "date", "active"])
def test_statistics_need_a_numeric_column(sales: Loaded, name: str) -> None:
    with pytest.raises(AnalysisError, match="needs a numeric column|statistics need a numeric column"):
        column_statistics(sales.frame, sales.info, name)


def test_case_insensitive_column_names(sales: Loaded) -> None:
    table = column_statistics(sales.frame, sales.info, "SALES")
    assert table.title == "Statistics for sales"


# ---------------------------------------------------------------- grouping and aggregation
def test_group_by_category_with_several_aggregations(sales: Loaded) -> None:
    table = query(
        sales,
        group_by=["category"],
        aggregations=[
            {"column": c, "function": f} for c, f in [("sales", "sum"), ("sales", "mean"), ("sales", "median")]
        ]
        + [{"column": "units", "function": "count"}, {"column": "product", "function": "nunique"}],
    )
    rows = {row[0]: dict(zip(table.columns, row, strict=True)) for row in table.rows}
    assert table.columns == [
        "category",
        "records",
        "sum_sales",
        "mean_sales",
        "median_sales",
        "count_units",
        "nunique_product",
    ]
    assert rows["Tools"]["records"] == "4" and rows["Tools"]["sum_sales"] == "12801"
    assert float(rows["Tools"]["mean_sales"] or 0) == pytest.approx(3200.25)
    assert rows["Tools"]["median_sales"] == "3100.25" and rows["Tools"]["count_units"] == "3"
    assert rows["Tools"]["nunique_product"] == "3"
    assert rows["Garden"]["sum_sales"] == "7000.25" and rows["Garden"]["records"] == "2"
    assert any("excluded from sum_sales" in n for n in table.notes)
    assert any("'records' counts rows" in n for n in table.notes)


def test_min_max_on_numbers_and_dates_and_totals_without_grouping(sales: Loaded) -> None:
    table = query(
        sales,
        aggregations=[
            {"column": "sales", "function": "max"},
            {"column": "units", "function": "min"},
            {"column": "date", "function": "max"},
        ],
    )
    assert table.title == "Totals" and table.rows == [["8", "9100", "1", "2024-06-30"]]


def test_group_by_two_columns_and_count(sales: Loaded) -> None:
    table = query(sales, group_by=["region", "category"], sort_by=[{"column": "records", "descending": True}])
    assert table.columns == ["region", "category", "records"]
    assert table.rows[0] == ["North", "Tools", "2"] and table.total_rows == 7  # 7 region/category pairs


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"aggregations": [{"column": "product", "function": "sum"}]}, "needs numbers"),
        ({"aggregations": [{"column": "region", "function": "mean"}]}, "needs numbers"),
        ({"aggregations": [{"column": "product", "function": "max"}]}, "numeric or date column"),
        ({"group_by": ["nope"]}, "Unknown column 'nope'"),
        ({"group_by": ["category"], "sort_by": [{"column": "sum_sales"}]}, "Cannot sort by 'sum_sales'"),
    ],
)
def test_incompatible_or_unknown_columns_are_rejected(sales: Loaded, spec: dict[str, Any], message: str) -> None:
    with pytest.raises(AnalysisError, match=re.escape(message)):
        query(sales, **spec)


@pytest.mark.parametrize(
    "spec",
    [
        {"aggregations": [{"column": "sales", "function": "variance"}]},
        {"aggregations": [{"column": "sales", "function": "__import__('os')"}]},
        {"filters": [{"column": "sales", "operator": "drop", "value": "1"}]},
        {"filters": [{"column": "sales", "operator": "==", "value": "1"}]},
        {"group_by": ["a", "b", "c", "d"]},
        {"filters": [{"column": "sales", "operator": "is_missing"}] * 6},
        {"aggregations": [{"column": "sales", "function": "sum"}] * 6},
        {"sort_by": [{"column": "sales"}] * 4},
        {"columns": ["sales"] * 21},
        {"limit": MAX_LIMIT + 1},
        {"limit": 0},
        {"filters": [{"column": "sales", "operator": "equals", "value": "x" * 201}]},
    ],
)
def test_invalid_operations_and_limits_fail_validation(spec: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Query.model_validate(spec)


# ---------------------------------------------------------------- filters
@pytest.mark.parametrize(
    ("condition", "products"),
    [
        ({"column": "category", "operator": "equals", "value": "tools"}, ["Alpha", "Beta", "Alpha", "Eta"]),
        ({"column": "category", "operator": "not_equals", "value": "Tools"}, ["Gamma", "Delta", "Eps", "Zeta-9000"]),
        ({"column": "sales", "operator": "greater_than", "value": "5000"}, ["Beta", "Gamma", "Eps"]),
        ({"column": "sales", "operator": "greater_or_equal", "value": "5000"}, ["Beta", "Gamma", "Eps", "Eta"]),
        ({"column": "sales", "operator": "less_than", "value": "1200.5"}, ["Zeta-9000"]),
        ({"column": "sales", "operator": "less_or_equal", "value": "1200.5"}, ["Alpha", "Alpha", "Zeta-9000"]),
        ({"column": "product", "operator": "contains", "value": "ET"}, ["Beta", "Zeta-9000", "Eta"]),
        ({"column": "sales", "operator": "is_missing"}, ["Delta"]),
        (
            {"column": "units", "operator": "is_not_missing"},
            ["Alpha", "Gamma", "Delta", "Alpha", "Eps", "Zeta-9000", "Eta"],
        ),
        ({"column": "date", "operator": "greater_or_equal", "value": "2024-04-01"}, ["Eps", "Zeta-9000", "Eta"]),
        ({"column": "active", "operator": "equals", "value": "false"}, ["Beta", "Eps", "Zeta-9000"]),
        ({"column": "units", "operator": "equals", "value": "10"}, ["Alpha", "Alpha"]),
    ],
)
def test_each_filter_operator(sales: Loaded, condition: dict[str, Any], products: list[str]) -> None:
    table = query(sales, filters=[condition], columns=["product"])
    assert column(table, "product") == products
    assert f"{len(products)} of 8 rows matched the filters." in table.notes


def test_filters_combine_and_feed_aggregations(sales: Loaded) -> None:
    table = query(
        sales,
        filters=[
            {"column": "sales", "operator": "greater_than", "value": "1000"},
            {"column": "region", "operator": "equals", "value": "North"},
        ],
        aggregations=[{"column": "sales", "function": "sum"}],
    )
    assert table.rows == [["3", "9401.25"]]


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ({"column": "sales", "operator": "greater_than", "value": "lots"}, "is not a number"),
        ({"column": "sales", "operator": "equals", "value": "nan"}, "not a usable number"),
        ({"column": "product", "operator": "greater_than", "value": "A"}, "needs a numeric or date column"),
        ({"column": "sales", "operator": "contains", "value": "5"}, "works on text columns"),
        ({"column": "date", "operator": "less_than", "value": "next week"}, "is not a date"),
        ({"column": "active", "operator": "equals", "value": "maybe"}, "use 'true' or 'false'"),
        ({"column": "sales", "operator": "greater_than"}, "needs a value"),
    ],
)
def test_filter_values_must_fit_the_column(sales: Loaded, condition: dict[str, Any], message: str) -> None:
    with pytest.raises(AnalysisError, match=re.escape(message)):
        query(sales, filters=[condition])


def test_filter_values_are_never_code_or_regex(sales: Loaded) -> None:
    for value in ["__import__('os').system('echo hi')", "1; DROP TABLE x", "sales > 0 or True", "${7*7}"]:
        with pytest.raises(AnalysisError, match="is not a number"):
            query(sales, filters=[{"column": "sales", "operator": "greater_than", "value": value}])
    assert query(sales, filters=[{"column": "product", "operator": "contains", "value": ".*"}]).total_rows == 0
    for name in ["__import__('os')", "../../etc/passwd", "sales; import os"]:
        with pytest.raises(AnalysisError, match="Unknown column"):
            query(sales, group_by=[name])


# ---------------------------------------------------------------- sorting, top-N, empty and bounded results
def test_top_and_bottom_n(sales: Loaded) -> None:
    top = query(sales, columns=["product", "sales"], sort_by=[{"column": "sales", "descending": True}], limit=3)
    assert top.rows == [["Eps", "9100"], ["Gamma", "7000.25"], ["Beta", "5400"]] and top.total_rows == 8
    bottom = query(sales, columns=["product", "sales"], sort_by=[{"column": "sales"}], limit=2)
    assert bottom.rows == [["Zeta-9000", "300"], ["Alpha", "1200.5"]]
    last = query(sales, columns=["product", "sales"], sort_by=[{"column": "sales"}])
    assert last.rows[-1] == ["Delta", None], "missing values sort last"


def test_top_categories_by_revenue(sales: Loaded) -> None:
    table = query(
        sales,
        group_by=["category"],
        aggregations=[{"column": "sales", "function": "sum"}],
        sort_by=[{"column": "SUM_SALES", "descending": True}],
        limit=2,
    )
    assert column(table, "category") == ["Tools", "Home"] and table.total_rows == 3


def test_empty_results_are_handled(sales: Loaded) -> None:
    rows = query(sales, filters=[{"column": "sales", "operator": "greater_than", "value": "1e9"}])
    assert rows.rows == [] and rows.total_rows == 0 and "0 of 8 rows matched the filters." in rows.notes
    totals = query(
        sales,
        filters=[{"column": "category", "operator": "equals", "value": "Nothing"}],
        aggregations=[{"column": "sales", "function": "mean"}, {"column": "sales", "function": "sum"}],
    )
    assert totals.rows == [["0", None, None]]
    grouped = query(sales, filters=[{"column": "region", "operator": "equals", "value": "Mars"}], group_by=["region"])
    assert grouped.rows == [] and grouped.total_rows == 0


def test_result_rows_columns_and_model_payload_are_bounded(store: DatasetStore) -> None:
    wide_header = ",".join(f"c{i}" for i in range(25))
    data = (
        wide_header + "\n" + "\n".join(",".join(str(r * 100 + i) for i in range(25)) for r in range(150)) + "\n"
    ).encode()
    info = save(store, SESSION_A, data, "wide.csv")
    loaded = Loaded(store, info, load_frame(store, SESSION_A, info))
    table = query(loaded, limit=MAX_LIMIT)
    assert len(table.rows) == 100 and table.total_rows == 150 and len(table.columns) == 20
    assert any("first 20 of 25 columns" in n for n in table.notes)
    payload = json.loads(to_model_payload(table))
    assert len(payload["rows"]) <= MODEL_ROWS and payload["total_result_rows"] == 150
    assert len(to_model_payload(table)) <= MAX_MODEL_CHARS


def test_long_cells_are_cut(store: DatasetStore) -> None:
    info = save(store, SESSION_A, b"note\n" + b"x" * 500 + b"\n", "long.csv")
    table = run_query(load_frame(store, SESSION_A, info), info, Query())
    assert len(table.rows[0][0] or "") == 200 and (table.rows[0][0] or "").endswith("…")


# ---------------------------------------------------------------- dataset resolution and isolation
def test_resolution_rules(store: DatasetStore) -> None:
    with pytest.raises(AnalysisError, match="No CSV dataset has been uploaded"):
        resolve_dataset(store, SESSION_A)
    first = save(store, SESSION_A, name="first.csv")
    assert resolve_dataset(store, SESSION_A).id == first.id, "a single dataset is used"
    second = save(store, SESSION_A, name="second.csv")
    with pytest.raises(AnalysisError, match=r"several datasets \(first.csv, second.csv\) and none is selected"):
        resolve_dataset(store, SESSION_A)
    assert resolve_dataset(store, SESSION_A, "SECOND").id == second.id, "named by the user"
    with pytest.raises(AnalysisError, match="No dataset named 'third.csv'"):
        resolve_dataset(store, SESSION_A, "third.csv")
    store.select(SESSION_A, first.id)
    assert resolve_dataset(store, SESSION_A).id == first.id and store.selected(SESSION_A) == first


def test_a_session_cannot_resolve_select_or_load_another_sessions_dataset(store: DatasetStore) -> None:
    other = save(store, SESSION_B, name="secret.csv")
    with pytest.raises(AnalysisError, match="No CSV dataset"):
        resolve_dataset(store, SESSION_A, "secret.csv")
    save(store, SESSION_A, name="mine.csv")
    with pytest.raises(AnalysisError, match="No dataset named 'secret.csv'"):
        resolve_dataset(store, SESSION_A, "secret.csv")
    with pytest.raises(DatasetNotFoundError):
        store.select(SESSION_A, other.id)
    with pytest.raises(AnalysisError, match="could not be read"):
        load_frame(store, SESSION_A, other)  # metadata of B's dataset presented under session A


@pytest.mark.parametrize("bad", ["../" + "a" * 29, "A" * 32, "", "..\\..\\x"])
def test_invalid_session_ids_cannot_escape_the_store(store: DatasetStore, bad: str) -> None:
    with pytest.raises(AnalysisError, match="No CSV dataset"):
        resolve_dataset(store, bad)
    assert dataset_context(store, bad) == ""


def test_analysis_never_changes_the_stored_files(sales: Loaded) -> None:
    folder = sales.store.root / SESSION_A / sales.info.id

    def digest() -> dict[str, str]:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}

    before = digest()
    summarize(sales.frame, sales.info)
    column_statistics(sales.frame, sales.info, "sales")
    query(sales, filters=[{"column": "sales", "operator": "is_missing"}], sort_by=[{"column": "sales"}])
    query(sales, group_by=["category"], aggregations=[{"column": "sales", "function": "sum"}])
    load_frame(sales.store, SESSION_A, sales.info)
    assert digest() == before and sorted(before) == [DATA_FILE, META_FILE]


def test_unreadable_dataset_error_hides_paths(store: DatasetStore) -> None:
    info = save(store, SESSION_A)
    (store.root / SESSION_A / info.id / DATA_FILE).unlink()
    with pytest.raises(AnalysisError) as error:
        load_frame(store, SESSION_A, info)
    message = str(error.value)
    assert str(store.root) not in message and "data.csv" not in message and "\\" not in message


# ---------------------------------------------------------------- prompt context: metadata only
def test_dataset_context_lists_metadata_never_rows(store: DatasetStore) -> None:
    assert dataset_context(store, SESSION_A) == NO_DATASETS_PROMPT
    save(store, SESSION_A)
    text = dataset_context(store, SESSION_A)
    assert '"sales.csv" [used for analysis]: 8 rows' in text and "sales (decimal)" in text
    assert "Zeta-9000" not in text and "1200.5" not in text, "no cell values in the prompt"
    save(store, SESSION_A, name="other.csv")
    assert "No dataset is selected" in dataset_context(store, SESSION_A)


# ---------------------------------------------------------------- LangGraph integration (scripted fake LLM)
def engine_for(store: DatasetStore, llm: FakeChatModel, retriever: Any = None) -> Any:
    return get_engine(
        llm=llm,
        retriever=retriever or StaticRetriever([]),
        tools=[*get_tools(), *make_dataset_tools(store)],
        session_context=functools.partial(dataset_context, store),
    )


def analysis_llm(tool: str, **args: Any) -> FakeChatModel:
    return FakeChatModel(reply="ANSWER", tool_name=tool, tool_args=args)


def test_analysis_question_runs_the_tool_on_the_real_data(store: DatasetStore) -> None:
    save(store, SESSION_A)
    llm = analysis_llm("query_dataset", group_by=["category"], aggregations=[{"column": "sales", "function": "sum"}])
    result = engine_for(store, llm).invoke(f"{TOOL_TRIGGER} total sales by category", [], SESSION_A)
    assert [t.title for t in result.analysis] == ["Grouped by category"]
    assert dict((r[0], r[2]) for r in result.analysis[0].rows) == {
        "Garden": "7000.25",
        "Home": "9400",
        "Tools": "12801",
    }
    assert '"Tools", "4", "12801"' in result.answer, "the model receives the computed values"
    assert "Result computed with pandas" in result.answer


def test_summary_and_statistics_tools(store: DatasetStore) -> None:
    save(store, SESSION_A)
    summary = engine_for(store, analysis_llm("dataset_summary")).invoke(f"{TOOL_TRIGGER} summarize", [], SESSION_A)
    assert summary.analysis[0].title == "Summary of sales.csv"
    stats = engine_for(store, analysis_llm("column_statistics", column="units")).invoke(
        f"{TOOL_TRIGGER} average units", [], SESSION_A
    )
    assert dict((r[0], r[1]) for r in stats.analysis[0].rows)["mean"] == "10.85714286"


def test_ordinary_conversation_does_not_analyse(store: DatasetStore) -> None:
    save(store, SESSION_A)
    llm = analysis_llm("dataset_summary")
    result = engine_for(store, llm).invoke("Hello, how are you?", [], SESSION_A)
    assert result.analysis == [] and result.answer == "ANSWER"
    assert "sales.csv" in llm.prompt_text() and "Zeta-9000" not in llm.prompt_text()


def test_document_rag_questions_still_use_retrieval(store: DatasetStore) -> None:
    save(store, SESSION_A)
    retriever = StaticRetriever([make_chunk("Annual leave is 25 days.", source="leave_policy.md")])
    llm = analysis_llm("dataset_summary")
    result = engine_for(store, llm, retriever).invoke("How many leave days do I get?", [], SESSION_A)
    assert [s.source for s in result.sources] == ["leave_policy.md"] and result.analysis == []
    assert "Annual leave is 25 days." in llm.prompt_text()


def test_no_dataset_prompts_for_upload(store: DatasetStore) -> None:
    llm = analysis_llm("dataset_summary")
    result = engine_for(store, llm).invoke(f"{TOOL_TRIGGER} summarize my data", [], SESSION_A)
    assert result.analysis == [] and "upload one with the Datasets panel" in result.answer
    assert NO_DATASETS_PROMPT in llm.prompt_text()


def test_ambiguous_datasets_ask_for_clarification(store: DatasetStore) -> None:
    save(store, SESSION_A, name="2023.csv")
    save(store, SESSION_A, name="2024.csv")
    llm = analysis_llm("column_statistics", column="sales")
    result = engine_for(store, llm).invoke(f"{TOOL_TRIGGER} average sales", [], SESSION_A)
    assert result.analysis == [] and "none is selected. Ask the user which one" in result.answer
    assert "No dataset is selected" in llm.prompt_text()
    named = engine_for(store, analysis_llm("column_statistics", column="sales", dataset="2024.csv")).invoke(
        f"{TOOL_TRIGGER} average sales in 2024.csv", [], SESSION_A
    )
    assert named.analysis[0].dataset == "2024.csv"


def test_tools_stay_inside_the_current_session(store: DatasetStore) -> None:
    save(store, SESSION_A, name="private.csv")
    llm = analysis_llm("dataset_summary", dataset="private.csv")
    result = engine_for(store, llm).invoke(f"{TOOL_TRIGGER} summarize private.csv", [], SESSION_B)
    assert result.analysis == [] and "No CSV dataset has been uploaded" in result.answer
    system_prompt = str(llm.calls[0][0].content)
    assert NO_DATASETS_PROMPT in system_prompt and "private.csv" not in system_prompt, "B's context lists nothing of A"


def test_session_id_cannot_be_supplied_by_the_model(store: DatasetStore) -> None:
    save(store, SESSION_A)
    llm = analysis_llm("dataset_summary", session_id=SESSION_A, state={"session_id": SESSION_A})
    result = engine_for(store, llm).invoke(f"{TOOL_TRIGGER} summarize", [], SESSION_B)
    assert result.analysis == [], "the injected session comes from the graph state only"


def test_tool_failures_do_not_crash_the_graph(store: DatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
    save(store, SESSION_A)
    bad_args = analysis_llm("query_dataset", filters=[{"column": "sales", "operator": "drop_table"}])
    result = engine_for(store, bad_args).invoke(f"{TOOL_TRIGGER} weird", [], SESSION_A)
    assert result.answer.startswith("ANSWER | tool said: Error") and result.analysis == []

    import app.tools.dataset_tools as tools_module

    def explode(*args: Any, **kwargs: Any) -> AnalysisTable:
        raise RuntimeError(f"boom in {store.root}")

    monkeypatch.setattr(tools_module, "summarize", explode)
    result = engine_for(store, analysis_llm("dataset_summary")).invoke(f"{TOOL_TRIGGER} summarize", [], SESSION_A)
    assert "the analysis failed unexpectedly" in result.answer and str(store.root) not in result.answer


def test_analysis_tables_per_answer_are_bounded(store: DatasetStore) -> None:
    data = ("n\n" + "\n".join(str(i) for i in range(150)) + "\n").encode()
    save(store, SESSION_A, data, "numbers.csv")
    result = engine_for(store, analysis_llm("query_dataset", limit=100)).invoke(f"{TOOL_TRIGGER} list", [], SESSION_A)
    table = result.analysis[0]
    assert len(table.rows) == 100 and table.total_rows == 150


# ---------------------------------------------------------------- wiring
def test_build_components_adds_dataset_tools_only_with_a_store(
    store: DatasetStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(deps, "get_engine", lambda **kwargs: captured.append(kwargs))
    deps.build_components()
    deps.build_components(dataset_store=store)
    without, with_store = captured
    assert without["tools"] is None and "session_context" not in without
    names = [t.name for t in with_store["tools"]]
    assert names == ["calculator", "current_datetime", "dataset_summary", "column_statistics", "query_dataset"]
    assert with_store["session_context"](SESSION_A) == NO_DATASETS_PROMPT


# ---------------------------------------------------------------- UI and API
@pytest.fixture()
def ui(settings: Settings) -> Iterator[tuple[TestClient, DatasetStore]]:
    app = create_app(settings)
    store: DatasetStore = app.state.dataset_store
    llm = analysis_llm(
        "query_dataset",
        group_by=["category"],
        aggregations=[{"column": "sales", "function": "sum"}],
        sort_by=[{"column": "sum_sales", "descending": True}],
    )
    app.dependency_overrides[get_components] = lambda: Components(engine=engine_for(store, llm))
    with TestClient(app) as client:
        yield client, store


def upload(client: TestClient, session_id: str, name: str = "sales.csv", data: bytes = SALES_CSV) -> dict[str, Any]:
    response = client.post(f"/api/sessions/{session_id}/datasets", files={"file": (name, data, "text/csv")})
    assert response.status_code == 201
    result: dict[str, Any] = response.json()
    return result


def test_chat_turn_shows_and_saves_the_result_table(ui: tuple[TestClient, DatasetStore]) -> None:
    client, _ = ui
    session_id = client.post("/api/sessions", json={}).json()["id"]
    upload(client, session_id)
    html = client.post(
        f"/ui/sessions/{session_id}/messages", data={"content": f"{TOOL_TRIGGER} revenue by category"}
    ).text
    assert 'aria-label="Analysis results"' in html and "Grouped by category" in html
    assert "<td>Tools</td><td>4</td><td>12801</td>" in html.replace("\n", "").replace("  ", "")
    saved = client.get(f"/api/sessions/{session_id}").json()["messages"][1]
    assert saved["analysis"][0]["rows"][0] == ["Tools", "4", "12801"]
    assert "Grouped by category" in client.get(f"/sessions/{session_id}").text, "shown again after reload"


def test_result_tables_escape_values_and_report_omitted_rows(settings: Settings) -> None:
    from app.storage.json_store import SessionStore

    store = SessionStore(settings.sessions_dir)
    session = store.create()
    table = AnalysisTable(
        title="Rows",
        dataset="x.csv",
        columns=["<b>c</b>"],
        rows=[["<script>alert(1)</script>"], [None]],
        total_rows=250,
    )
    store.append_messages(session.id, Message(role="assistant", content="answer", analysis=[table]))
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=get_engine(llm=FakeChatModel(), tools=[]))
    with TestClient(app) as client:
        html = client.get(f"/sessions/{session.id}").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;c&lt;/b&gt;" in html and "Showing 2 of 250 result rows." in html
    assert '<td class="missing" title="Missing value">—</td>' in html


def test_dataset_selector_and_states(ui: tuple[TestClient, DatasetStore]) -> None:
    client, store = ui
    session_id = client.post("/api/sessions", json={}).json()["id"]
    assert "No datasets yet." in client.get(f"/sessions/{session_id}").text
    first = upload(client, session_id, "first.csv")
    page = client.get(f"/sessions/{session_id}").text
    assert "Analysis questions use <b>first.csv</b>" in page and "(the only dataset)" in page
    second = upload(client, session_id, "second.csv")
    page = client.get(f"/sessions/{session_id}").text
    assert "Several datasets: choose one" in page and page.count("Use for analysis</button>") == 2
    html = client.post(f"/ui/sessions/{session_id}/datasets/{second['id']}/select").text
    assert "second.csv will be used for analysis questions" in html and "used for analysis</span>" in html
    assert store.selected(session_id) is not None and store.selected(session_id).id == second["id"]  # type: ignore[union-attr]
    assert client.post(f"/api/sessions/{session_id}/datasets/{first['id']}/select").json()["id"] == first["id"]


def test_selecting_another_sessions_dataset_is_404(ui: tuple[TestClient, DatasetStore]) -> None:
    client, store = ui
    owner, other = (client.post("/api/sessions", json={}).json()["id"] for _ in range(2))
    info = upload(client, owner)
    assert client.post(f"/ui/sessions/{other}/datasets/{info['id']}/select").status_code == 404
    assert client.post(f"/api/sessions/{other}/datasets/{info['id']}/select").status_code == 404
    assert store.selected(other) is None


def test_old_messages_without_analysis_still_load(settings: Settings) -> None:
    session = {"id": "c" * 32, "title": "t", "messages": [{"role": "assistant", "content": "hi", "sources": []}]}
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    (settings.sessions_dir / f"{'c' * 32}.json").write_text(json.dumps(session), encoding="utf-8")
    from app.storage.json_store import SessionStore

    assert SessionStore(settings.sessions_dir).get("c" * 32).messages[0].analysis == []
