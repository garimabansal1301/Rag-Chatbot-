"""Application settings, read from environment variables (and `.env`).

Every path the app writes to comes from here, so tests and the trainer's
evaluator can point the app at temporary folders without touching your data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Values already set in the real environment win over `.env`.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


_TRUE, _FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
IMAGE_SEARCH_PROVIDERS = ("none", "openverse")

# Same name as app.vectordb.embeddings.DEFAULT_EMBEDDING_MODEL (kept here to avoid an import cycle; a test checks it).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Minimum cosine score for a retrieved chunk to be used, per embedding model, measured on this project's documents
# (relevant questions scored >= 0.59, unrelated ones <= 0.58). Other models get no threshold unless one is set.
CALIBRATED_MIN_SCORES = {DEFAULT_EMBEDDING_MODEL: 0.60}
# Same folder as app.evaluation.report.REPORTS_DIR (the CLI's default), kept here to avoid an import cycle.
EVAL_REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"


def _check_min_score(value: float) -> float:
    if not 0.0 <= value <= 1.0:  # also rejects NaN
        raise ValueError(f"RETRIEVAL_MIN_SCORE must be a number from 0 to 1 (0 disables filtering), not {value!r}")
    return value


def _env_min_score(embedding_model: str) -> float:
    """RETRIEVAL_MIN_SCORE if set, else the calibrated value for the embedding model, else 0 (no filtering)."""
    raw = (os.getenv("RETRIEVAL_MIN_SCORE") or "").strip()
    if not raw:
        return CALIBRATED_MIN_SCORES.get(embedding_model.strip() or DEFAULT_EMBEDDING_MODEL, 0.0)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"RETRIEVAL_MIN_SCORE must be a number from 0 to 1 (0 disables filtering), not {raw!r}"
        ) from None
    return _check_min_score(value)


def _env_bool(name: str, default: bool) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    if not value:
        return default
    if value in _TRUE or value in _FALSE:
        return value in _TRUE
    raise ValueError(f"{name} must be true or false, not {value!r}")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    sessions_dir: Path
    documents_dir: Path
    vector_dir: Path
    chunking_strategy: str = "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    llm_provider: str = "anthropic"
    llm_model: str = ""
    embedding_model: str = ""
    max_upload_bytes: int = 5 * 1024 * 1024
    allowed_upload_suffixes: tuple[str, ...] = field(default=(".md", ".txt", ".pdf"))
    web_search_enabled: bool = False  # Gemini Google Search grounding (not wired up yet)
    image_search_provider: str = "none"  # one of IMAGE_SEARCH_PROVIDERS
    retrieval_min_score: float = CALIBRATED_MIN_SCORES[DEFAULT_EMBEDDING_MODEL]  # 0 = keep every retrieved chunk
    datasets_dir: Path | None = None  # CSV datasets per chat session; None -> data_dir / "datasets"
    max_dataset_bytes: int = 10 * 1024 * 1024
    human_review_enabled: bool = True  # hold unsupported or sensitive answers for review (app/chat/review.py)
    reviews_dir: Path | None = None  # review records per chat session; None -> data_dir / "reviews"
    checkpoint_db: Path | None = None  # SQLite file for paused graph turns; None -> data_dir / "checkpoints" / ...
    eval_reports_dir: Path = EVAL_REPORTS_DIR  # evaluation reports read by the Evals dashboard (and written by runs)
    mcp_enabled: bool = False  # serve calculator/current_datetime from the local MCP server (app/mcp/)

    def __post_init__(self) -> None:
        _check_min_score(self.retrieval_min_score)
        if self.image_search_provider not in IMAGE_SEARCH_PROVIDERS:
            raise ValueError(
                f"IMAGE_SEARCH_PROVIDER must be one of {', '.join(IMAGE_SEARCH_PROVIDERS)}, "
                f"not {self.image_search_provider!r}"
            )
        if self.datasets_dir is None:
            object.__setattr__(self, "datasets_dir", self.data_dir / "datasets")
        if self.reviews_dir is None:
            object.__setattr__(self, "reviews_dir", self.data_dir / "reviews")
        if self.checkpoint_db is None:
            object.__setattr__(self, "checkpoint_db", self.data_dir / "checkpoints" / "graph.sqlite")

    @property
    def dataset_root(self) -> Path:
        """Where datasets are stored (always set after __post_init__)."""
        assert self.datasets_dir is not None
        return self.datasets_dir

    @property
    def review_root(self) -> Path:
        assert self.reviews_dir is not None
        return self.reviews_dir

    @property
    def checkpoint_path(self) -> Path:
        """The SQLite checkpoint file (created when the chat engine is first built, not at start-up)."""
        assert self.checkpoint_db is not None
        return self.checkpoint_db

    def ensure_dirs(self) -> None:
        for path in (self.sessions_dir, self.documents_dir, self.vector_dir, self.dataset_root):
            path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Build a fresh Settings object from the current environment."""
    data_dir = _env_path("DATA_DIR", PROJECT_ROOT / "data")
    embedding_model = os.getenv("EMBEDDING_MODEL", "")
    return Settings(
        data_dir=data_dir,
        sessions_dir=_env_path("SESSIONS_DIR", data_dir / "sessions"),
        documents_dir=_env_path("DOCUMENTS_DIR", data_dir / "documents"),
        vector_dir=_env_path("VECTOR_DIR", data_dir / "vectorstore"),
        chunking_strategy=os.getenv("CHUNKING_STRATEGY", "fixed"),
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 100),
        top_k=_env_int("TOP_K", 4),
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("LLM_MODEL", ""),
        embedding_model=embedding_model,
        max_upload_bytes=_env_int("MAX_UPLOAD_MB", 5) * 1024 * 1024,
        web_search_enabled=_env_bool("WEB_SEARCH_ENABLED", False),
        image_search_provider=(os.getenv("IMAGE_SEARCH_PROVIDER") or "none").strip().lower(),
        retrieval_min_score=_env_min_score(embedding_model),
        datasets_dir=_env_path("DATASETS_DIR", data_dir / "datasets"),
        max_dataset_bytes=_env_int("MAX_DATASET_MB", 10) * 1024 * 1024,
        human_review_enabled=_env_bool("HUMAN_REVIEW", True),
        reviews_dir=_env_path("REVIEWS_DIR", data_dir / "reviews"),
        checkpoint_db=_env_path("CHECKPOINT_DB", data_dir / "checkpoints" / "graph.sqlite"),
        eval_reports_dir=_env_path("EVAL_REPORTS_DIR", EVAL_REPORTS_DIR),
        mcp_enabled=_env_bool("MCP_ENABLED", False),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for the running app. Call `get_settings.cache_clear()` in tests."""
    return load_settings()
