"""Chunker guarantees the M3 contract tests do not check: no gaps or reordering, bounded overlap."""

from __future__ import annotations

import re

import pytest

from app.models import Chunk, Document
from app.vectordb.chunking import get_chunker
from tests.conftest import SAMPLE_DOCS

_TOKEN = re.compile(r"\S+")
# Repeated phrases make set-based word checks blind to dropped passages.
REPEATED = Document(
    id="rep",
    source="rep.md",
    text="\n\n".join(f"Section {i}. The same sentence repeats here. Leave is 24 days." for i in range(30)),
)


def _docs() -> list[Document]:
    docs = [  # text files only: data/documents may also hold uploaded PDFs
        Document(id=p.stem, text=p.read_text(encoding="utf-8"), source=p.name)
        for p in sorted(SAMPLE_DOCS.glob("*"))
        if p.suffix in {".md", ".txt"}
    ]
    return [*docs, REPEATED]


def _overlaps(doc_tokens: list[str], chunks: list[Chunk]) -> list[int]:
    """Walk the chunks over the document's token sequence; return each chunk's overlap in characters.

    Fails if a chunk is not a contiguous run of document tokens starting at or before the
    first token not yet covered (a gap), or if the chunks do not reach the end of the document.
    """
    covered, overlaps = 0, []
    for chunk in chunks:
        tokens = [m.group() for m in _TOKEN.finditer(chunk.text)]
        starts = [
            s
            for s in range(covered, max(covered - len(tokens), 0) - 1, -1)
            if doc_tokens[s : s + len(tokens)] == tokens
        ]
        assert starts, f"chunk {chunk.position} does not continue the document without a gap"
        shared = covered - starts[0]  # tokens repeated from the previous chunk (fewest that fit)
        token_spans = [m.span() for m in _TOKEN.finditer(chunk.text)]
        overlaps.append(token_spans[shared - 1][1] if shared else 0)
        covered = max(covered, starts[0] + len(tokens))
    assert covered == len(doc_tokens), "chunks stop before the end of the document"
    return overlaps


@pytest.mark.parametrize("doc", _docs(), ids=lambda d: d.source)
@pytest.mark.parametrize(("size", "overlap"), [(120, 0), (300, 50), (800, 100)])
def test_chunks_cover_document_in_order_with_bounded_overlap(doc: Document, size: int, overlap: int) -> None:
    chunks = get_chunker("fixed", chunk_size=size, chunk_overlap=overlap).chunk(doc)
    overlaps = _overlaps(_TOKEN.findall(doc.text), chunks)
    assert all(o <= overlap for o in overlaps), f"overlap above {overlap}: {max(overlaps)}"
    if overlap and len(chunks) > 1:
        assert any(overlaps), "consecutive chunks should share text when chunk_overlap > 0"


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_sizes_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        get_chunker("fixed", chunk_size=size, chunk_overlap=overlap)


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_chunker("no-such-strategy", chunk_size=100, chunk_overlap=10)
