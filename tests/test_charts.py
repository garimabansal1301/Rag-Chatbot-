"""Phase 3: bar, line and pie charts built from real pandas results. Temp folders, fake LLMs, no network."""

from __future__ import annotations

import functools
import json
import math
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.chat.engine import get_engine
from app.config import Settings
from app.datasets.analysis import (
    MAX_PIE_SLICES,
    AnalysisError,
    Query,
    load_frame,
    run_query,
    to_model_payload,
)
from app.datasets.parsing import parse_csv
from app.datasets.store import DatasetStore
from app.deps import Components, get_components
from app.main import create_app
from app.models import MAX_CHART_POINTS, AnalysisTable, ChartSpec, DatasetInfo, Message
from app.storage.json_store import SessionStore
from app.tools import get_tools
from app.tools.dataset_tools import dataset_context, make_dataset_tools
from tests.fakes import TOOL_TRIGGER, FakeChatModel, StaticRetriever, make_chunk

SESSION_A, SESSION_B = "a" * 32, "b" * 32
# Rows deliberately out of date order and across a year boundary; one missing date, one missing sales value.
ORDERS = b"""order_id,order_date,product,category,region,sales,profit
1,2024-02-03,Laptop,Electronics,North,1000,200
2,2023-12-15,Chair,Furniture,South,300,-50
3,2024-01-20,Pen,Office,East,20,5
4,2024-02-10,Phone,Electronics,West,800,150
5,2023-12-01,Desk,Furniture,North,500,-80
6,2024-01-05,Laptop,Electronics,South,1200,250
7,,Paper,Office,East,40,8
8,2024-03-12,Chair,Furniture,West,,10
9,2024-03-30,Pen,Office,North,30,6
"""


def save(store: DatasetStore, session_id: str, data: bytes = ORDERS, name: str = "orders.csv") -> DatasetInfo:
    return store.save(session_id, name, parse_csv(data), len(data))


@pytest.fixture()
def store(tmp_path: Path) -> DatasetStore:
    return DatasetStore(tmp_path / "datasets")


@pytest.fixture()
def orders(store: DatasetStore) -> tuple[Any, DatasetInfo]:
    info = save(store, SESSION_A)
    return load_frame(store, SESSION_A, info), info


def chart_of(orders: tuple[Any, DatasetInfo], **spec: Any) -> AnalysisTable:
    frame, info = orders
    return run_query(frame, info, Query.model_validate(spec))


SUM_SALES = [{"column": "sales", "function": "sum"}]


# ---------------------------------------------------------------- ChartSpec validation
def valid_spec(**changes: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "type": "bar",
        "title": "Sales by category",
        "label_name": "category",
        "series_name": "sum_sales",
        "labels": ["A", "B"],
        "values": [1.0, 2.5],
    }
    spec.update(changes)
    return spec


def test_valid_chart_spec() -> None:
    chart = ChartSpec(**valid_spec())
    assert chart.type == "bar" and chart.values == [1.0, 2.5] and chart.percentages is None


@pytest.mark.parametrize(
    "changes",
    [
        {"type": "scatter"},
        {"type": "<script>"},
        {"labels": ["A"]},  # length mismatch
        {"values": [1.0, math.nan]},
        {"values": [1.0, math.inf]},
        {"values": [1.0, "x"]},
        {"labels": [], "values": []},
        {"labels": [str(i) for i in range(MAX_CHART_POINTS + 1)], "values": [1.0] * (MAX_CHART_POINTS + 1)},
        {"title": ""},
        {"html": "<b>x</b>"},  # unknown fields are refused
        {"options": {"onClick": "alert(1)"}},
        {"type": "pie", "values": [5.0, -1.0]},
        {"type": "pie", "values": [0.0, 0.0]},
        {"percentages": [50.0, 50.0]},  # percentages are for pie charts only
        {"type": "pie", "percentages": [50.0, 150.0]},
        {"type": "pie", "percentages": [50.0]},
    ],
)
def test_malformed_chart_specs_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ChartSpec(**valid_spec(**changes))


def test_long_texts_are_capped() -> None:
    chart = ChartSpec(**valid_spec(title="t" * 500, labels=["L" * 300, "B"]))
    assert len(chart.title) == 120 and len(chart.labels[0]) == 80


# ---------------------------------------------------------------- bar charts from grouped data
def test_bar_chart_of_totals_by_category(orders: tuple[Any, DatasetInfo]) -> None:
    table = chart_of(orders, group_by=["category"], aggregations=SUM_SALES, chart={"type": "bar"})
    chart = table.chart
    assert chart is not None and chart.type == "bar"
    assert chart.labels == ["Electronics", "Furniture", "Office"] and chart.values == [3000.0, 800.0, 90.0]
    assert (
        chart.label_name == "category" and chart.series_name == "sum_sales" and chart.title == "sum_sales by category"
    )
    assert [r[2] for r in table.rows] == ["3000", "800", "90"], "chart and table come from the same result"
    assert any("missing value(s) in 'sales' were excluded" in n for n in table.notes)


def test_top_n_bar_chart(orders: tuple[Any, DatasetInfo]) -> None:
    table = chart_of(
        orders,
        group_by=["product"],
        aggregations=SUM_SALES,
        sort_by=[{"column": "sum_sales", "descending": True}],
        limit=2,
        chart={"type": "bar", "title": "Top products"},
    )
    assert table.chart is not None and table.chart.title == "Top products"
    assert table.chart.labels == ["Laptop", "Phone"] and table.chart.values == [2200.0, 800.0]
    assert "The chart shows the first 2 of 6 groups." in table.chart.notes


def test_bar_chart_point_limit(store: DatasetStore) -> None:
    data = ("name,value\n" + "".join(f"item{i:03d},{i}\n" for i in range(150))).encode()
    info = save(store, SESSION_A, data, "many.csv")
    table = run_query(
        load_frame(store, SESSION_A, info),
        info,
        Query.model_validate(
            {
                "group_by": ["name"],
                "aggregations": [{"column": "value", "function": "sum"}],
                "limit": 100,
                "chart": {"type": "bar"},
            }
        ),
    )
    assert table.chart is not None and len(table.chart.values) == MAX_CHART_POINTS == 100
    assert "The chart shows the first 100 of 150 groups." in table.chart.notes


# ---------------------------------------------------------------- monthly line charts
def test_monthly_line_chart_is_chronological(orders: tuple[Any, DatasetInfo]) -> None:
    table = chart_of(orders, group_by_month="order_date", aggregations=SUM_SALES, chart={"type": "line"})
    chart = table.chart
    assert chart is not None and chart.type == "line" and chart.label_name == "month"
    assert chart.labels == ["2023-12", "2024-01", "2024-02", "2024-03"], "months in time order, across the year"
    assert chart.values == [800.0, 1220.0, 1800.0, 30.0]
    assert "1 row(s) without a 'order_date' date were left out of the monthly groups." in table.notes
    assert any("missing value(s) in 'sales' were excluded" in n for n in table.notes)
    assert table.columns == ["month", "records", "sum_sales"] and table.rows[0][:2] == ["2023-12", "2"]


def test_monthly_records_without_aggregation(orders: tuple[Any, DatasetInfo]) -> None:
    table = chart_of(orders, group_by_month="order_date", chart={"type": "line"})
    assert (
        table.chart is not None and table.chart.series_name == "records" and table.chart.values == [2.0, 2.0, 2.0, 2.0]
    )


def test_month_grouping_needs_a_date_column(orders: tuple[Any, DatasetInfo], store: DatasetStore) -> None:
    with pytest.raises(AnalysisError, match=re.escape("'product' is not a date column") + ".*Date columns: order_date"):
        chart_of(orders, group_by_month="product", aggregations=SUM_SALES, chart={"type": "line"})
    info = save(store, SESSION_B, b"category,sales\nA,1\nB,2\n", "nodates.csv")
    with pytest.raises(AnalysisError, match="has no date column"):
        run_query(load_frame(store, SESSION_B, info), info, Query.model_validate({"group_by_month": "category"}))


def test_month_column_name_does_not_clash(store: DatasetStore) -> None:
    info = save(store, SESSION_A, b"month,day,sales\nJan,2024-01-02,5\nFeb,2024-02-03,7\n", "clash.csv")
    table = run_query(
        load_frame(store, SESSION_A, info),
        info,
        Query.model_validate({"group_by_month": "day", "chart": {"type": "line"}}),
    )
    assert (
        table.columns[0] == "month of day" and table.chart is not None and table.chart.labels == ["2024-01", "2024-02"]
    )


# ---------------------------------------------------------------- pie charts and safeguards
def test_pie_chart_values_and_percentages(orders: tuple[Any, DatasetInfo]) -> None:
    table = chart_of(orders, group_by=["category"], aggregations=SUM_SALES, chart={"type": "pie"})
    chart = table.chart
    assert chart is not None and chart.type == "pie"
    assert chart.labels == ["Electronics", "Furniture", "Office"] and chart.values == [3000.0, 800.0, 90.0]
    assert chart.percentages == [round(v / 3890 * 100, 2) for v in (3000, 800, 90)]
    assert sum(chart.percentages) == pytest.approx(100, abs=0.05)
    payload = json.loads(to_model_payload(table))
    assert payload["chart_shown_to_user"]["percent_of_total"] == {
        "Electronics": 77.12,
        "Furniture": 20.57,
        "Office": 2.31,
    }


def test_pie_of_counts(orders: tuple[Any, DatasetInfo]) -> None:
    table = chart_of(orders, group_by=["region"], chart={"type": "pie"})
    assert table.chart is not None and table.chart.series_name == "records" and sum(table.chart.values) == 9


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"aggregations": [{"column": "profit", "function": "sum"}]}, "has negative values"),
        ({"aggregations": [{"column": "sales", "function": "mean"}]}, "needs a sum or a count"),
        ({"aggregations": [{"column": "sales", "function": "max"}]}, "needs a sum or a count"),
        ({"aggregations": [{"column": "product", "function": "nunique"}]}, "needs a sum or a count"),
    ],
)
def test_misleading_pie_charts_are_refused(orders: tuple[Any, DatasetInfo], spec: dict[str, Any], message: str) -> None:
    with pytest.raises(AnalysisError, match=message) as error:
        chart_of(orders, group_by=["category"], chart={"type": "pie"}, **spec)
    assert "bar chart" in str(error.value)


def test_pie_with_zero_total_is_refused(store: DatasetStore) -> None:
    info = save(store, SESSION_A, b"group,amount\nA,0\nB,0\n", "zero.csv")
    with pytest.raises(AnalysisError, match="add up to zero"):
        run_query(
            load_frame(store, SESSION_A, info),
            info,
            Query.model_validate(
                {
                    "group_by": ["group"],
                    "aggregations": [{"column": "amount", "function": "sum"}],
                    "chart": {"type": "pie"},
                }
            ),
        )


def test_many_pie_categories_are_combined_into_a_labelled_other(store: DatasetStore) -> None:
    data = ("name,amount\n" + "".join(f"c{i:02d},{i + 1}\n" for i in range(20))).encode()
    info = save(store, SESSION_A, data, "many.csv")
    table = run_query(
        load_frame(store, SESSION_A, info),
        info,
        Query.model_validate(
            {
                "group_by": ["name"],
                "aggregations": [{"column": "amount", "function": "sum"}],
                "limit": 5,
                "chart": {"type": "pie"},
            }
        ),
    )
    chart = table.chart
    assert chart is not None and len(chart.values) == MAX_PIE_SLICES
    assert chart.labels[0] == "c19" and chart.labels[-1] == "Other (9 categories)"
    assert sum(chart.values) == sum(range(1, 21)) == 210, "no category is silently dropped"
    assert (
        chart.values[-1] == sum(range(1, 10)) and "The 9 smallest categories are combined into 'Other'." in chart.notes
    )
    assert sum(chart.percentages or []) == pytest.approx(100, abs=0.05)


# ---------------------------------------------------------------- missing, empty and incompatible data
def test_groups_without_a_value_are_left_out_of_the_chart(orders: tuple[Any, DatasetInfo]) -> None:
    table = chart_of(
        orders,
        filters=[{"column": "product", "operator": "equals", "value": "Chair"}],
        group_by=["region"],
        aggregations=[{"column": "sales", "function": "mean"}],
        chart={"type": "bar"},
    )
    assert table.chart is not None and table.chart.labels == ["South"] and table.chart.values == [300.0]
    assert "1 group(s) with no mean_sales value are not in the chart." in table.chart.notes
    assert [r[0] for r in table.rows] == ["South", "West"], "the table still lists every group"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"chart": {"type": "bar"}}, "A chart needs grouped results"),
        ({"aggregations": SUM_SALES, "chart": {"type": "bar"}}, "A chart needs grouped results"),
        ({"group_by": ["category", "region"], "aggregations": SUM_SALES, "chart": {"type": "bar"}}, "one grouping"),
        (
            {"group_by": ["category"], "aggregations": SUM_SALES, "chart": {"type": "bar", "value_column": "x"}},
            "Cannot chart 'x'",
        ),
        ({"group_by": ["category"], "chart": {"type": "bar", "value_column": "category"}}, "Cannot chart 'category'"),
        (
            {
                "filters": [{"column": "sales", "operator": "greater_than", "value": "99999"}],
                "group_by": ["category"],
                "aggregations": SUM_SALES,
                "chart": {"type": "bar"},
            },
            "no sum_sales values to chart",
        ),
    ],
)
def test_charts_that_cannot_be_drawn_are_explained(
    orders: tuple[Any, DatasetInfo], spec: dict[str, Any], message: str
) -> None:
    with pytest.raises(AnalysisError, match=re.escape(message)):
        chart_of(orders, **spec)


@pytest.mark.parametrize(
    "spec",
    [
        {"chart": {"type": "scatter"}},
        {"chart": {"type": "bar", "value_column": "x" * 201}},
        {"chart": {"type": "bar", "title": "t" * 121}},
        {"group_by_month": "x" * 201},
    ],
)
def test_invalid_chart_requests_fail_validation(spec: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Query.model_validate(spec)


def test_value_column_is_case_insensitive_and_explicit(orders: tuple[Any, DatasetInfo]) -> None:
    table = chart_of(
        orders,
        group_by=["category"],
        aggregations=[{"column": "sales", "function": "sum"}, {"column": "profit", "function": "sum"}],
        chart={"type": "bar", "value_column": "SUM_PROFIT"},
    )
    # Electronics profit: 200 + 150 + 250 = 600; Furniture: -50 - 80 + 10 = -120; Office: 5 + 8 + 6 = 19
    assert (
        table.chart is not None
        and table.chart.series_name == "sum_profit"
        and table.chart.values == [600.0, -120.0, 19.0]
    )


# ---------------------------------------------------------------- LangGraph integration (scripted fake LLM)
def engine_for(store: DatasetStore, llm: FakeChatModel, retriever: Any = None) -> Any:
    return get_engine(
        llm=llm,
        retriever=retriever or StaticRetriever([]),
        tools=[*get_tools(), *make_dataset_tools(store)],
        session_context=functools.partial(dataset_context, store),
    )


def chart_llm(**args: Any) -> FakeChatModel:
    return FakeChatModel(reply="ANSWER", tool_name="query_dataset", tool_args=args)


CATEGORY_PIE = {"group_by": ["category"], "aggregations": SUM_SALES, "chart": {"type": "pie"}}


def test_chart_request_runs_through_the_tool_loop(store: DatasetStore) -> None:
    save(store, SESSION_A)
    result = engine_for(store, chart_llm(**CATEGORY_PIE)).invoke(
        f"{TOOL_TRIGGER} sales share by category", [], SESSION_A
    )
    chart = result.analysis[0].chart
    assert chart is not None and chart.type == "pie" and chart.values == [3000.0, 800.0, 90.0]
    assert '"chart_shown_to_user"' in result.answer and "never draw charts" in result.answer


def test_monthly_trend_through_the_tool_loop(store: DatasetStore) -> None:
    save(store, SESSION_A)
    llm = chart_llm(group_by_month="order_date", aggregations=SUM_SALES, chart={"type": "line"})
    result = engine_for(store, llm).invoke(f"{TOOL_TRIGGER} monthly sales trend", [], SESSION_A)
    assert result.analysis[0].chart is not None and result.analysis[0].chart.labels[0] == "2023-12"


def test_refused_chart_is_explained_not_crashing(store: DatasetStore) -> None:
    save(store, SESSION_A)
    llm = chart_llm(
        group_by=["category"], aggregations=[{"column": "sales", "function": "mean"}], chart={"type": "pie"}
    )
    result = engine_for(store, llm).invoke(f"{TOOL_TRIGGER} pie of average sales", [], SESSION_A)
    assert result.analysis == [] and "Use a bar chart instead" in result.answer


def test_ordinary_chat_and_rag_produce_no_chart(store: DatasetStore) -> None:
    save(store, SESSION_A)
    chat = engine_for(store, chart_llm(**CATEGORY_PIE)).invoke("Hello!", [], SESSION_A)
    assert chat.analysis == [] and chat.answer == "ANSWER"
    retriever = StaticRetriever([make_chunk("Annual leave is 25 days.", source="leave_policy.md")])
    rag = engine_for(store, chart_llm(**CATEGORY_PIE), retriever).invoke("How much leave?", [], SESSION_A)
    assert rag.analysis == [] and [s.source for s in rag.sources] == ["leave_policy.md"]


def test_ambiguous_dataset_asks_before_charting(store: DatasetStore) -> None:
    save(store, SESSION_A, name="2023.csv")
    save(store, SESSION_A, name="2024.csv")
    result = engine_for(store, chart_llm(**CATEGORY_PIE)).invoke(f"{TOOL_TRIGGER} chart sales", [], SESSION_A)
    assert result.analysis == [] and "Ask the user which one to analyse" in result.answer


def test_charts_respect_session_isolation(store: DatasetStore) -> None:
    save(store, SESSION_A, name="private.csv")
    llm = chart_llm(dataset="private.csv", **CATEGORY_PIE)
    result = engine_for(store, llm).invoke(f"{TOOL_TRIGGER} chart private.csv", [], SESSION_B)
    assert result.analysis == [] and "No CSV dataset has been uploaded" in result.answer


# ---------------------------------------------------------------- UI: rendering, saving, reloading, escaping
@pytest.fixture()
def ui(settings: Settings) -> Iterator[tuple[TestClient, DatasetStore, Settings]]:
    app = create_app(settings)
    store: DatasetStore = app.state.dataset_store
    app.dependency_overrides[get_components] = lambda: Components(engine=engine_for(store, chart_llm(**CATEGORY_PIE)))
    with TestClient(app) as client:
        yield client, store, settings


def chart_json(html: str) -> list[dict[str, Any]]:
    blocks = re.findall(r'<script type="application/json" class="chart-spec">(.*?)</script>', html, flags=re.S)
    return [json.loads(block) for block in blocks]


def test_chart_is_rendered_saved_and_shown_again_after_reload(ui: tuple[TestClient, DatasetStore, Settings]) -> None:
    client, store, _ = ui
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{session_id}/datasets", files={"file": ("orders.csv", ORDERS, "text/csv")})
    html = client.post(
        f"/ui/sessions/{session_id}/messages", data={"content": f"{TOOL_TRIGGER} share by category"}
    ).text
    assert '<figure class="analysis-chart chart-pie">' in html and '<canvas role="img"' in html
    assert chart_json(html)[0]["values"] == [3000.0, 800.0, 90.0]
    assert "Grouped by category" in html and "<td>Electronics</td>" in html, "the table is still there"
    assert '<p class="chart-fallback" hidden>' in html
    saved = client.get(f"/api/sessions/{session_id}").json()["messages"][1]
    assert saved["analysis"][0]["chart"]["type"] == "pie"
    reloaded = client.get(f"/sessions/{session_id}").text
    assert chart_json(reloaded) == chart_json(html)


def test_chart_text_cannot_break_out_of_the_json_block(settings: Settings) -> None:
    store = SessionStore(settings.sessions_dir)
    session = store.create()
    evil = "</script><script>alert(1)</script>"
    chart = ChartSpec(type="bar", title=evil, label_name="c", series_name="v", labels=[evil, "b"], values=[1, 2])
    table = AnalysisTable(title="T", dataset="x.csv", columns=["c", "v"], rows=[["a", "1"]], total_rows=1, chart=chart)
    store.append_messages(session.id, Message(role="assistant", content="answer", analysis=[table]))
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=get_engine(llm=FakeChatModel(), tools=[]))
    with TestClient(app) as client:
        html = client.get(f"/sessions/{session.id}").text
    assert "<script>alert(1)</script>" not in html and html.count("</script>") == html.count("<script")
    assert chart_json(html)[0]["labels"][0] == evil, "the JSON still carries the exact text, safely escaped"
    assert "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in html, "the caption is escaped text"


def test_messages_without_charts_render_as_before(settings: Settings) -> None:
    store = SessionStore(settings.sessions_dir)
    session = store.create()
    table = AnalysisTable(title="Rows", dataset="x.csv", columns=["a"], rows=[["1"]], total_rows=1)
    store.append_messages(
        session.id,
        Message(role="assistant", content="plain answer"),
        Message(role="assistant", content="table answer", analysis=[table]),
    )
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: Components(engine=get_engine(llm=FakeChatModel(), tools=[]))
    with TestClient(app) as client:
        html = client.get(f"/sessions/{session.id}").text
    assert "plain answer" in html and "table answer" in html and 'aria-label="Analysis results"' in html
    assert "analysis-chart" not in html and "chart-spec" not in html
    assert SessionStore(settings.sessions_dir).get(session.id).messages[1].analysis[0].chart is None


def test_page_loads_pinned_chart_library_and_chart_script(client: TestClient) -> None:
    session_id = client.post("/api/sessions", json={}).json()["id"]
    page = client.get(f"/sessions/{session_id}").text
    assert '<script src="https://unpkg.com/chart.js@4.4.1/dist/chart.umd.js" defer></script>' in page
    assert re.search(r'<script src="[^"]*/static/charts\.js" defer></script>', page)
    script = client.get("/static/charts.js")
    assert script.status_code == 200 and "eval(" not in script.text and "innerHTML" not in script.text


# ---------------------------------------------------------------- static/charts.js, run in Node with a fake Chart.js
CHARTS_JS = Path(__file__).resolve().parent.parent / "static" / "charts.js"
HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const scenario = JSON.parse(process.argv[1]);  // with `node -e`, the first extra argument is argv[1]
class El {
  constructor(tag, cls, parent, text) {
    this.tag = tag; this.classes = new Set((cls || "").split(" ").filter(Boolean)); this.parent = parent || null;
    this.children = []; this.dataset = {}; this.hidden = false; this.textContent = text || "";
    if (parent) parent.children.push(this);
  }
  matches(sel) {
    const [tag, ...cls] = sel.split(".");
    return (!tag || this.tag === tag) && cls.every((c) => this.classes.has(c));
  }
  descendants() { return this.children.flatMap((c) => [c, ...c.descendants()]); }
  querySelectorAll(selector) {
    const parts = selector.split(" ");
    const last = parts.pop();
    return this.descendants().filter((el) => el.matches(last) && parts.every((p) => {
      for (let n = el.parent; n; n = n.parent) if (n.matches(p)) return true; return false; }));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
const root = new El("div", "messages");
const figures = scenario.specs.map((text) => {
  const figure = new El("figure", "analysis-chart", root);
  const box = new El("div", "chart-canvas", figure);
  new El("canvas", "", box);
  new El("script", "chart-spec", figure, text);
  const fallback = new El("p", "chart-fallback", figure); fallback.hidden = true;
  return figure;
});
const created = [], registry = new Map(), listeners = {}, warnings = [];
class FakeChart {
  constructor(canvas, config) {
    this.canvas = canvas; this.config = config; this.destroyed = false; registry.set(canvas, this); created.push(this);
  }
  destroy() { this.destroyed = true; registry.delete(this.canvas); }
  static getChart(canvas) { return registry.get(canvas); }
}
globalThis.window = globalThis;
if (scenario.chartLoaded) globalThis.Chart = FakeChart;
globalThis.document = { documentElement: {}, querySelectorAll: (s) => root.querySelectorAll(s),
  addEventListener: (type, fn) => { (listeners[type] = listeners[type] || []).push(fn); } };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
console.warn = (...args) => warnings.push(args.join(" "));
vm.runInThisContext(fs.readFileSync(scenario.file, "utf8"));
const fire = (type, target) => (listeners[type] || []).forEach((fn) => fn({ target }));
fire("DOMContentLoaded");
for (let i = 0; i < (scenario.extraSwaps || 0); i++) fire("htmx:afterSwap", root);
const tooltip = created.length && created[0].config.options.plugins.tooltip
  ? created[0].config.options.plugins.tooltip.callbacks.label({ dataIndex: 0 }) : null;
if (scenario.cleanup) fire("htmx:beforeCleanupElement", root);
console.log(JSON.stringify({
  created: created.map((c) => ({ type: c.config.type, labels: c.config.data.labels,
                                 data: c.config.data.datasets[0].data, destroyed: c.destroyed })),
  states: figures.map((f) => f.dataset.chartState || ""),
  fallbackShown: figures.map((f) => !f.querySelector(".chart-fallback").hidden),
  canvasHidden: figures.map((f) => f.querySelector(".chart-canvas").hidden),
  tooltip, warnings, listeners: Object.keys(listeners).sort(),
}));
"""
BAR = json.dumps(
    {"type": "bar", "title": "t", "label_name": "c", "series_name": "v", "labels": ["A", "B"], "values": [1, 2]}
)
PIE = json.dumps(
    {
        "type": "pie",
        "title": "t",
        "label_name": "c",
        "series_name": "v",
        "labels": ["<b>A</b>", "B"],
        "values": [3, 1],
        "percentages": [75, 25],
    }
)


def run_charts_js(
    specs: list[str], chart_loaded: bool = True, extra_swaps: int = 0, cleanup: bool = False
) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; the browser script cannot be exercised")
    scenario = {
        "file": str(CHARTS_JS),
        "specs": specs,
        "chartLoaded": chart_loaded,
        "extraSwaps": extra_swaps,
        "cleanup": cleanup,
    }
    done = subprocess.run(
        [node, "-e", HARNESS, json.dumps(scenario)], capture_output=True, text=True, timeout=60, check=True
    )
    result: dict[str, Any] = json.loads(done.stdout)
    return result


def test_js_draws_each_chart_once_even_after_more_htmx_swaps() -> None:
    result = run_charts_js([BAR, PIE], extra_swaps=3)
    assert [c["type"] for c in result["created"]] == ["bar", "pie"], "no duplicate chart instances"
    assert result["states"] == ["rendered", "rendered"] and result["fallbackShown"] == [False, False]
    assert result["created"][1]["labels"] == ["<b>A</b>", "B"], "labels are passed as plain text, not HTML"
    assert result["listeners"] == ["DOMContentLoaded", "htmx:afterSwap", "htmx:beforeCleanupElement"]


def test_js_pie_tooltip_uses_the_server_percentages() -> None:
    assert run_charts_js([PIE])["tooltip"] == "<b>A</b>: 3 (75%)"


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps({"type": "scatter", "labels": ["A"], "values": [1]}),
        json.dumps({"type": "bar", "labels": ["A", "B"], "values": [1]}),
        json.dumps({"type": "bar", "labels": ["A"], "values": ["1"]}),
        json.dumps({"type": "bar", "labels": [], "values": []}),
        json.dumps({"type": "bar", "labels": [str(i) for i in range(101)], "values": list(range(101))}),
        '{"type": "bar", "labels": ["A"], "values": [NaN]}',
    ],
)
def test_js_invalid_spec_shows_the_fallback_and_keeps_going(bad: str) -> None:
    result = run_charts_js([bad, BAR])
    assert result["fallbackShown"] == [True, False] and result["canvasHidden"] == [True, False]
    assert [c["type"] for c in result["created"]] == ["bar"], "a broken chart does not stop the others"
    assert result["warnings"] == ["A chart could not be displayed; its table is still shown."]


def test_js_without_chart_library_shows_the_fallback() -> None:
    result = run_charts_js([BAR], chart_loaded=False)
    assert result["created"] == [] and result["fallbackShown"] == [True] and result["states"] == ["failed"]


def test_js_destroys_charts_when_htmx_removes_them() -> None:
    result = run_charts_js([BAR, PIE], cleanup=True)
    assert all(c["destroyed"] for c in result["created"])
