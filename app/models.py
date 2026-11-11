"""Shared data types. These are part of the CONTRACT — do not rename fields.

The trainer's evaluator builds these objects directly, so every module you
write must accept and return exactly these shapes.
"""

from __future__ import annotations

import ipaddress
import math
import re
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- untrusted external data
# Web and image results come from external providers, so these types are strict: URLs must be public
# http(s) addresses, and text is cleaned and length-capped. Building a model from bad provider data
# raises ValidationError; the provider adapters (later phases) drop such items instead of failing a turn.

MAX_URL_LENGTH = 2048
_HOSTNAME = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,62}$")
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _public_hostname(host: str) -> str:
    """Lower-case DNS name with a dot, not an IP address and not a local-network name."""
    host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    else:
        raise ValueError("IP-address hosts are not allowed")
    if not _HOSTNAME.match(host) or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise ValueError(f"not a public host name: {host!r}")
    return host


def _check_url(value: str, schemes: tuple[str, ...]) -> str:
    if len(value) > MAX_URL_LENGTH or _CONTROL_CHARS.search(value) or " " in value:
        raise ValueError("URL is too long or contains spaces or control characters")
    parts = urlsplit(value)
    if parts.scheme.lower() not in schemes:
        raise ValueError(f"URL scheme must be {' or '.join(schemes)}")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URL must not contain credentials")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("URL has an invalid port") from None
    if port is not None:
        raise ValueError("URL must use the default port")
    _public_hostname(parts.hostname or "")
    return value


def _clean_text(max_length: int) -> BeforeValidator:
    """Collapse whitespace, drop control characters and cut to `max_length` (provider text can be anything)."""

    def clean(value: object) -> object:
        if not isinstance(value, str):
            return value  # let pydantic report the type error
        text = " ".join(_CONTROL_CHARS.sub(" ", value).split())
        return text if len(text) <= max_length else text[: max_length - 1].rstrip() + "…"

    return BeforeValidator(clean)


HttpsUrl = Annotated[str, AfterValidator(lambda v: _check_url(v, ("https",)))]
"""A public https URL; required for anything the browser loads, such as image thumbnails."""
WebUrl = Annotated[str, AfterValidator(lambda v: _check_url(v, ("https", "http")))]
"""A public http(s) URL; acceptable for links the user chooses to open."""
Domain = Annotated[str, AfterValidator(_public_hostname)]


class WebSource(BaseModel):
    """A web page that a grounded (Google Search) answer is based on, from the grounding metadata."""

    title: Annotated[str, _clean_text(300)]
    url: HttpsUrl  # Gemini returns Google redirect links; shown as-is, nothing placed in between
    domain: Domain


class WebGrounding(BaseModel):
    """A Google Search grounded answer, kept SEPARATE from the document-based answer.

    Privacy boundary: the grounded request is built only from the user's own question (and at most
    minimal earlier user turns), never from retrieved document excerpts, so internal text is not
    sent to Google Search. Compliance: Google's terms forbid modifying grounded results or mixing
    other content into them, so `answer` is displayed unmodified, in its own block, together with
    `search_suggestions_html`. See docs/web-and-image-search.md.
    """

    answer: str = Field(min_length=1, max_length=20_000)
    sources: list[WebSource] = Field(default_factory=list, max_length=20)
    search_queries: list[Annotated[str, _clean_text(200)]] = Field(default_factory=list, max_length=10)
    # Google's Search Suggestions HTML must be shown with the answer and must not be modified, so it is
    # not sanitised; it may only ever be rendered inside a sandboxed iframe (no scripts, no same-origin).
    search_suggestions_html: str | None = Field(default=None, max_length=50_000)


class ImageResult(BaseModel):
    """One image card. The thumbnail must be https (the browser loads it); everything else is a link or text."""

    title: Annotated[str, _clean_text(200)]
    thumbnail_url: HttpsUrl
    source_page_url: WebUrl
    creator: Annotated[str, _clean_text(200)] = ""
    creator_url: WebUrl | None = None
    license_name: Annotated[str, _clean_text(60)]
    license_url: WebUrl | None = None
    attribution: Annotated[str, _clean_text(1000)] = ""
    provider: Literal["openverse"] = "openverse"


ResultColumnType = Literal["integer", "decimal", "percent", "date", "boolean", "text", "mixed"]
MAX_CHART_POINTS = 100  # bars, line points or pie slices in one chart
MAX_CHART_LABEL_LENGTH = 80


class ChartSpec(BaseModel):
    """A chart of one numeric series, built only from a computed AnalysisTable (never from model text).

    Rendered in the browser from this validated JSON; labels are drawn as text, never as HTML.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["bar", "line", "pie"]
    title: Annotated[str, _clean_text(120)] = Field(min_length=1)
    label_name: Annotated[str, _clean_text(100)]  # what the labels are, e.g. "category" or "month of order_date"
    series_name: Annotated[str, _clean_text(100)]  # what the values are, e.g. "sum_sales"
    labels: list[Annotated[str, _clean_text(MAX_CHART_LABEL_LENGTH)]] = Field(min_length=1, max_length=MAX_CHART_POINTS)
    values: list[float] = Field(min_length=1, max_length=MAX_CHART_POINTS)
    percentages: list[float] | None = None  # pie only: share of the total, adding up to ~100
    notes: list[Annotated[str, _clean_text(300)]] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def _check(self) -> ChartSpec:
        if len(self.labels) != len(self.values):
            raise ValueError("labels and values must have the same length")
        if not all(math.isfinite(v) for v in self.values):
            raise ValueError("chart values must be finite numbers")
        if self.percentages is not None:
            if self.type != "pie" or len(self.percentages) != len(self.values):
                raise ValueError("percentages are only for pie charts, one per value")
            if not all(math.isfinite(p) and 0 <= p <= 100 for p in self.percentages):
                raise ValueError("percentages must be between 0 and 100")
        if self.type == "pie" and (any(v < 0 for v in self.values) or sum(self.values) <= 0):
            raise ValueError("a pie chart needs non-negative values with a positive total")
        return self


class AnalysisTable(BaseModel):
    """A bounded result table computed with pandas from one of the session's CSV datasets."""

    title: str = Field(max_length=300)
    dataset: str  # file name of the analysed dataset
    columns: list[str] = Field(max_length=30)
    rows: list[list[str | None]] = Field(default_factory=list, max_length=100)  # formatted values; None = missing
    total_rows: int = 0  # rows in the full result, before the display limit
    notes: list[str] = Field(default_factory=list, max_length=10)
    chart: ChartSpec | None = None  # optional chart of this same result (absent in older messages)
    # Type of each column, so exports can write real numbers and dates (absent in older messages).
    column_types: list[ResultColumnType] | None = None

    @model_validator(mode="after")
    def _check_types(self) -> AnalysisTable:
        if self.column_types is not None and len(self.column_types) != len(self.columns):
            raise ValueError("column_types needs one entry per column")
        return self


# --------------------------------------------------------------------------- RAG types


class Document(BaseModel):
    """A whole source file after loading. `id` must be stable for the same file."""

    id: str
    text: str
    source: str  # file name, e.g. "leave_policy.md"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A piece of a Document. `position` is 0, 1, 2 ... in document order."""

    id: str  # must be unique and stable, e.g. f"{doc_id}:{position}"
    doc_id: str
    text: str
    source: str
    position: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A search hit. Higher `score` = more relevant."""

    chunk: Chunk
    score: float


class Source(BaseModel):
    """What the UI shows under an answer so users can see where it came from."""

    source: str
    chunk_id: str
    score: float | None = None
    snippet: str = ""


# --------------------------------------------------------------------------- human review (HITL)

ReviewKind = Literal["unsupported_answer", "sensitive_request", "consequential_action"]
ReviewStatus = Literal["pending", "processing", "approved", "edited", "rejected", "expired", "failed"]
ReviewChoice = Literal["approve", "reject", "edit"]
REVIEW_KINDS: dict[str, str] = {
    "unsupported_answer": "Answer not supported by the documents",
    "sensitive_request": "Sensitive customer request",
    "consequential_action": "Action with external consequences",
}
MAX_REVIEW_TEXT = 8000


class ProposedAction(BaseModel):
    """A tool call that needs approval before it runs (shown to the reviewer, never executed before)."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ReviewRequest(BaseModel):
    """What the engine returns when the graph paused for a human (the interrupt payload plus its thread)."""

    thread_id: str  # server-side checkpoint thread; never sent to the browser
    kind: ReviewKind
    reasons: list[str] = Field(min_length=1)
    question: str
    proposed_answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    actions: list[ProposedAction] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    """A reviewer's decision, validated by the server before the graph is resumed with it."""

    model_config = ConfigDict(extra="forbid")

    choice: ReviewChoice
    response: str | None = Field(default=None, max_length=MAX_REVIEW_TEXT)  # the edited answer, for "edit"
    note: str | None = Field(default=None, max_length=1000)  # optional reviewer note (kept in the review only)


class ReviewSummary(BaseModel):
    """A review as shown to the session owner (no internal thread or checkpoint details)."""

    id: str
    session_id: str
    kind: ReviewKind
    status: ReviewStatus = "pending"
    reasons: list[str]
    question: str
    proposed_answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    actions: list[ProposedAction] = Field(default_factory=list)
    final_answer: str | None = None
    note: str | None = None
    created_at: str = Field(default_factory=utc_now)
    decided_at: str | None = None

    @property
    def kind_label(self) -> str:
        return REVIEW_KINDS[self.kind]


class Review(ReviewSummary):
    """The stored review record: the summary plus the graph thread it resumes."""

    thread_id: str


class MessageReview(BaseModel):
    """Marks an assistant message that waits for (or went through) human review."""

    id: str
    kind: ReviewKind
    status: ReviewStatus = "pending"


class EngineResult(BaseModel):
    """What `ChatEngine.invoke` returns."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    web: WebGrounding | None = None  # optional, separate grounded web answer
    images: list[ImageResult] = Field(default_factory=list)
    analysis: list[AnalysisTable] = Field(default_factory=list)  # dataset analysis results for this answer
    review: ReviewRequest | None = None  # set when the graph paused for human review (answer is then empty)


# --------------------------------------------------------------------------- chat sessions


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = Field(default_factory=utc_now)
    sources: list[Source] = Field(default_factory=list)
    web: WebGrounding | None = None  # absent in sessions saved before web search existed
    images: list[ImageResult] = Field(default_factory=list)
    analysis: list[AnalysisTable] = Field(default_factory=list)  # absent in sessions saved before analysis existed
    review: MessageReview | None = None  # set when this answer needed human review (absent otherwise)


class Session(BaseModel):
    id: str
    title: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    messages: list[Message] = Field(default_factory=list)


class SessionSummary(BaseModel):
    id: str
    title: str
    updated_at: str
    message_count: int


# --------------------------------------------------------------------------- datasets (CSV uploads)


DatasetDtype = Literal["integer", "decimal", "boolean", "date", "text", "empty"]


class DatasetColumn(BaseModel):
    name: str  # cleaned and de-duplicated header name
    dtype: DatasetDtype
    missing: int = 0  # number of missing values in the column


class DatasetSource(BaseModel):
    """Where an imported dataset came from, when it was not a CSV upload (metadata only, never a path)."""

    format: Literal["xlsx"] = "xlsx"
    workbook: str  # cleaned file name of the uploaded workbook
    sheet: str  # worksheet that was imported
    formula_cells: dict[str, int] = Field(default_factory=dict)  # per column: cells holding Excel-saved results
    preview_formula_cells: list[tuple[int, int]] = Field(default_factory=list)  # (preview row, column) positions
    uncached_formula_cells: int = 0  # formulas without a saved value (imported as missing)
    error_cells: int = 0  # Excel error values such as #DIV/0! (imported as missing)


class WorkbookSheet(BaseModel):
    index: int
    name: str
    state: Literal["visible", "hidden", "veryHidden"] = "visible"
    rows: int | None = None  # as declared in the file (approximate; checked again on import)
    columns: int | None = None


class PendingWorkbook(BaseModel):
    """An uploaded .xlsx kept briefly in its session until the user picks a worksheet to import."""

    id: str
    session_id: str
    filename: str
    size_bytes: int
    sheets: list[WorkbookSheet]
    created_at: str = Field(default_factory=utc_now)


class DatasetInfo(BaseModel):
    """A CSV (or an imported worksheet) of one chat session: metadata and a short preview, never the whole file."""

    id: str  # internal id (also the folder name); never derived from the file name
    session_id: str  # owning session; a dataset is only reachable through this id
    filename: str  # cleaned display name of the uploaded file
    size_bytes: int
    encoding: str  # "UTF-8" or "UTF-16"
    delimiter: str
    rows: int
    columns: list[DatasetColumn]
    preview: list[list[str | None]] = Field(default_factory=list)  # first rows as written; None = missing
    warnings: list[str] = Field(default_factory=list)
    formula_like_cells: int = 0  # cells starting with = + - @ (neutralised if ever exported)
    created_at: str = Field(default_factory=utc_now)
    source: DatasetSource | None = None  # set for worksheets imported from .xlsx; None for CSV uploads
