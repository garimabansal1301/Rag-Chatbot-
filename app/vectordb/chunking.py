"""MILESTONE 3 — Chunking.

Split a Document into Chunks small enough to embed and retrieve.

You choose the METHOD (fixed-size with overlap, recursive by paragraph/sentence,
markdown-heading aware, semantic ...). The method is picked by name from
`CHUNKING_STRATEGY` in `.env`, so you can add several and switch between them.

Whatever method you use, every chunker must obey these rules (the evaluator
checks them on documents you have never seen):

  1. An empty / whitespace-only document gives `[]`.
  2. No chunk is empty or whitespace-only.
  3. `len(chunk.text) <= chunk_size` (characters) for every chunk.
  4. No text is lost: every word of the document appears in at least one chunk.
  5. `chunk.doc_id == document.id` and `chunk.source == document.source`.
  6. `position` is 0, 1, 2 ... in document order; `id` is unique (e.g. f"{doc_id}:{position}").
  7. Same input -> same output (no randomness).
  8. A smaller `chunk_size` gives more chunks for the same long document.
"""

from __future__ import annotations

import bisect
import re
from typing import Protocol

from app.config import get_settings
from app.models import Chunk, Document

Span = tuple[int, int]  # [start, end) character offsets into the document text

# Coarse to fine: paragraph break, line break, sentence end, any whitespace.
# Each match stays attached to the piece before it, so pieces cover the text exactly.
_SEPARATORS = (
    re.compile(r"\n[ \t]*\n\s*"),
    re.compile(r"\n\s*"),
    re.compile(r"(?<=[.!?])\s+"),
    re.compile(r"\s+"),
)
_WORD = re.compile(r"\S+")


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        """IN: one Document.  OUT: its Chunks, in order."""
        ...


class FixedSizeChunker:
    """Chunks of at most `chunk_size` characters, split at the coarsest natural boundary that fits.

    Pieces are found by splitting on paragraphs, then lines, then sentences, then words
    (a single word longer than `chunk_size` is hard-sliced). Pieces are packed greedily,
    and each new chunk starts up to `chunk_overlap` characters back, at a word start.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text.strip():
            return []
        pieces = self._split(text, (0, len(text)), level=0)
        texts = [text[start:end].strip() for start, end in self._merge(text, pieces)]
        return [
            Chunk(
                id=f"{document.id}:{position}",
                doc_id=document.id,
                text=chunk_text,
                source=document.source,
                position=position,
                metadata=dict(document.metadata),
            )
            for position, chunk_text in enumerate(t for t in texts if t)
        ]

    def _split(self, text: str, span: Span, level: int) -> list[Span]:
        """Split `span` into contiguous pieces, each at most chunk_size long."""
        start, end = span
        if end - start <= self.chunk_size:
            return [span]
        if level == len(_SEPARATORS):
            return [(i, min(i + self.chunk_size, end)) for i in range(start, end, self.chunk_size)]

        cuts = [m.end() for m in _SEPARATORS[level].finditer(text, start, end) if m.end() < end]
        bounds = [start, *cuts, end]
        pieces: list[Span] = []
        for piece in zip(bounds, bounds[1:], strict=False):
            pieces.extend(self._split(text, piece, level + 1))
        return pieces

    def _merge(self, text: str, pieces: list[Span]) -> list[Span]:
        """Pack contiguous pieces into chunk spans of at most chunk_size, with overlap."""
        word_starts = [m.start() for m in _WORD.finditer(text)]
        chunks: list[Span] = []
        start, end = pieces[0]
        for _, piece_end in pieces[1:]:
            if piece_end - start <= self.chunk_size:
                end = piece_end
                continue
            chunks.append((start, end))
            start, end = self._overlap_start(word_starts, end, piece_end), piece_end
        chunks.append((start, end))
        return chunks

    def _overlap_start(self, word_starts: list[int], prev_end: int, next_end: int) -> int:
        """Earliest word start within the overlap window that keeps the next chunk within chunk_size."""
        earliest = max(prev_end - self.chunk_overlap, next_end - self.chunk_size)
        i = bisect.bisect_left(word_starts, earliest)
        if i < len(word_starts) and word_starts[i] < prev_end:
            return word_starts[i]
        return prev_end


_STRATEGIES: dict[str, type[FixedSizeChunker]] = {"fixed": FixedSizeChunker}


def get_chunker(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Chunker:
    """Return the chunker for `strategy` (default: settings.chunking_strategy).

    IN:  strategy name, max chunk size in characters, overlap in characters.
         Any argument left as None falls back to `app.config.get_settings()`.
    OUT: an object with a `.chunk(document) -> list[Chunk]` method.
         Raises ValueError for an unknown strategy or invalid sizes.
    """
    settings = get_settings()
    name = (strategy if strategy is not None else settings.chunking_strategy).strip().lower()
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {name!r}; available: {', '.join(sorted(_STRATEGIES))}")
    return _STRATEGIES[name](
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )
 