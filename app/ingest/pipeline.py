"""MILESTONE 6 — Ingestion: file -> Document -> Chunks -> vectors -> store.

Used in two places:
  - `uv run python -m app.ingest` indexes everything in data/documents/.
  - The upload endpoints (app/api/documents.py, app/web.py) index one uploaded file.

Rules the evaluator checks (with its own chunker/embedder/store):
  - `load_document` supports .md and .txt (and here also .pdf); anything else raises ValueError.
  - `Document.id` is stable for the same file name (re-loading gives the same id).
  - `ingest_file` returns the number of chunks stored, and the store grows by that much.
  - Ingesting the same file twice does NOT create duplicates
    (hint: `store.delete_document(doc.id)` before adding).
  - `ingest_directory` ingests every supported file and returns the total chunk count.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import threading
from pathlib import Path
from typing import Any

from app.models import Chunk, Document
from app.vectordb.chunking import Chunker
from app.vectordb.embeddings import Embedder
from app.vectordb.store import VectorStore

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = (".md", ".txt")
SUPPORTED_SUFFIXES = (*TEXT_SUFFIXES, ".pdf")
MAX_PDF_PAGES = 1000

# Serialises "delete old copy -> add new copy" so two ingestions in this process never
# interleave. It does not protect against another process writing the same store.
_WRITE_LOCK = threading.Lock()


def document_id(file_name: str) -> str:
    """Stable id from the file name only, so re-ingesting or re-uploading a file replaces its chunks."""
    return hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:16]


def load_document(path: Path) -> Document:
    """IN: path to a .md, .txt or .pdf file.  OUT: Document(id, text, source=file name, metadata).

    Raises ValueError for an unsupported suffix or unreadable content (see `document_from_bytes`);
    OSError (e.g. FileNotFoundError) if the file cannot be read.
    """
    _check_suffix(path.name)
    return document_from_bytes(path.name, path.read_bytes())


def document_from_bytes(file_name: str, data: bytes) -> Document:
    """Build the Document for `file_name` from its raw bytes, without touching the filesystem.

    Raises ValueError for an unsupported suffix, invalid UTF-8 or binary text, and for a PDF that
    cannot be read, is password-protected, or has no extractable text (e.g. a scanned PDF).
    """
    suffix = _check_suffix(file_name)
    metadata: dict[str, Any] = {"suffix": suffix, "content_sha256": hashlib.sha256(data).hexdigest()}
    if suffix == ".pdf":
        text, metadata["pages"] = _pdf_text(file_name, data)
    else:
        text = _plain_text(file_name, data)
    return Document(id=document_id(file_name), text=text, source=file_name, metadata=metadata)


def _check_suffix(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"{file_name}: unsupported file type {suffix or '(none)'}; allowed: {SUPPORTED_SUFFIXES}")
    return suffix


def _plain_text(file_name: str, data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig")  # strips a leading BOM
    except UnicodeDecodeError as exc:
        raise ValueError(f"{file_name}: not valid UTF-8 text ({exc.reason} at byte {exc.start})") from None
    if "\x00" in text:
        raise ValueError(f"{file_name}: contains NUL bytes; looks like a binary file")
    return text


class _PdfError(ValueError):
    """A PDF problem already described for the user (vs. a raw pypdf error that still needs wrapping)."""


def _pdf_text(file_name: str, data: bytes) -> tuple[str, int]:
    """Text of every page, pages separated by a blank line. Returns (text, page count)."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):  # an empty user password still allows reading
            raise _PdfError(f"{file_name}: the PDF is password-protected; upload an unprotected copy")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise _PdfError(f"{file_name}: the PDF has more than {MAX_PDF_PAGES} pages")
        pages = [page.extract_text() or "" for page in reader.pages]
    except _PdfError:
        raise
    except Exception as exc:  # pypdf raises many exception types for damaged or unusual files
        raise ValueError(f"{file_name}: not a readable PDF ({type(exc).__name__}: {exc})") from None
    text = "\n\n".join(page.replace("\x00", "").strip() for page in pages if page.strip())
    if not text:
        raise ValueError(
            f"{file_name}: the PDF has no extractable text (it may be a scanned image); "
            "upload a text-based PDF or run OCR on it first"
        )
    return text, len(pages)


def _embed(chunks: list[Chunk], embedder: Embedder, source: str) -> list[list[float]]:
    """Embed all chunks in one batched call and check the result before anything is stored."""
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError(f"{source}: embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
    if any(len(v) != embedder.dimension or not all(math.isfinite(x) for x in v) for v in vectors):
        raise ValueError(f"{source}: embedder returned vectors of the wrong size or with invalid numbers")
    return vectors


def ingest_file(path: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """IN: one file plus the components to use.  OUT: number of chunks stored.

    Loading, chunking and embedding all happen before the store is touched, so a failure
    there leaves the previous copy of the document intact. The old copy is then deleted
    and the new one added; an empty file just removes the old copy.
    """
    return ingest_document(load_document(path), chunker=chunker, embedder=embedder, store=store)


def ingest_document(document: Document, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """Chunk, embed and store an already loaded Document, replacing any older copy. Same rules as `ingest_file`."""
    chunks = chunker.chunk(document)
    vectors = _embed(chunks, embedder, document.source) if chunks else []
    with _WRITE_LOCK:
        store.delete_document(document.id)
        if not chunks:
            logger.warning("%s has no text; nothing indexed", document.source)
            return 0
        try:
            store.add(chunks, vectors)
        except Exception as exc:
            raise RuntimeError(
                f"{document.source}: old chunks were removed but the new ones could not be saved; re-ingest this file"
            ) from exc
    return len(chunks)


def ingest_directory(directory: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """IN: a folder.  OUT: total chunks stored across all supported files (sorted by name).

    Only the top level of the folder is read. A file that cannot be loaded (ValueError/OSError)
    is logged and skipped so the rest are still indexed; other errors stop the run.
    """
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    total = 0
    for path in files:
        try:
            total += ingest_file(path, chunker=chunker, embedder=embedder, store=store)
        except (ValueError, OSError) as exc:
            logger.warning("Skipped %s: %s", path.name, exc)
    return total
