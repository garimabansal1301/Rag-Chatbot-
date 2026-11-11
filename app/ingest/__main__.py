"""`uv run python -m app.ingest` — index every file in DOCUMENTS_DIR.

The vector store supports one writing process at a time. Stop the web app first (or
restart it afterwards): it keeps the index in memory, and its next upload would
overwrite what this command wrote.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.ingest.pipeline import ingest_directory
from app.vectordb.chunking import get_chunker
from app.vectordb.embeddings import get_embedder
from app.vectordb.store import get_vector_store


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    settings = get_settings()
    total = ingest_directory(
        settings.documents_dir,
        chunker=get_chunker(),
        embedder=get_embedder(),
        store=get_vector_store(),
    )
    print(f"Indexed {total} chunks from {settings.documents_dir}")
    print("If the web app is running, restart it so it loads this index (one writing process at a time).")


if __name__ == "__main__":
    main()
