# Trainee Tasks

Do the milestones **in order**. There are no time limits. Each file you need to edit contains `TODO(milestone N)`
comments that say what goes in and what comes out. After each milestone run:

```bash
uv run pytest -m contract tests/contract/test_mN_*.py
uv run python self_check.py
```

| # | Milestone | Files | Done when |
|---|---|---|---|
| M1 | LLM + LangGraph engine | `app/chat/llm.py`, `app/chat/engine.py` | The chat replies with a real model answer and remembers earlier messages in the session |
| M2 | LangSmith observability | `app/observability.py`, `.env` | A chat turn appears in your LangSmith project, tagged with the `session_id`|
| M3 | Chunking | `app/vectordb/chunking.py` | Every sample document splits into chunks that obey the rules in the file |
| M4 | Embeddings | `app/vectordb/embeddings.py` | Related sentences give closer vectors than unrelated ones |
| M5 | Vector store | `app/vectordb/store.py` | Chunks survive a restart and `search` returns the closest first |
| M6 | Ingestion | `app/ingest/pipeline.py` | `uv run python -m app.ingest` fills the store and uploads in the browser get indexed |
| M7 | Retriever + RAG | `app/vectordb/retriever.py`, `app/chat/engine.py` | Answers use your documents and the UI shows sources under each answer | 
| M8 | Tools | `app/tools/`, `app/chat/engine.py` | The assistant can call `calculator` and `current_datetime` and then answer |
| M9 (stretch) | Streaming | `app/api/chat.py`, templates | The answer appears token by token |

## Details

**M1 — LLM + engine.** `get_llm()` returns a LangChain chat model chosen by `LLM_PROVIDER`. `get_engine(llm, retriever, tools)`
builds a compiled LangGraph graph (`START → generate → END`). In: new message, history, session id. Out: `EngineResult(answer, sources)`.
Use the objects you are given; don't create your own inside the graph.

**M2 — Observability.** `configure_tracing()` returns `True` only when tracing is on *and* an API key exists, else `False`
(never crash). Pass `config={"metadata": {"session_id": ...}}` when invoking the graph.

**M3 — Chunking.** Pick any method (fixed size with overlap, by paragraph, by heading, semantic…). Choose by name through
`CHUNKING_STRATEGY`. The rules (size limit, nothing lost, unique ids, deterministic) are listed in the file.

**M4 — Embeddings.** A class with `dimension`, `embed_documents` (batched, order kept) and `embed_query`. Any local or hosted model.

**M5 — Vector store.** `add` (upsert by chunk id), `search`, `delete_document`, `count`, persisted under `persist_dir`.

**M6 — Ingestion.** `load_document` (.md/.txt, stable id), `ingest_file` (no duplicates when run twice), `ingest_directory`.

**M7 — Retriever and RAG.** `retrieve(query, k=None)` embeds, searches and returns ranked chunks; decorate it so it shows up in
LangSmith as a retriever run. Add a `retrieve` node before `generate` so context reaches the prompt on every turn, and return `sources`.

**M8 — Tools.** `calculator` must never use `eval` on raw input and must return a string starting with `Error` for bad input.
Add the tool loop to the graph (`ToolNode` + `tools_condition`).

## Rules of thumb

- Read keys and model names from `.env`; never hard-code them.
- Keep functions small and typed; `ruff` and `mypy` are part of the evaluation.
- Test each module on its own with the fakes in `tests/fakes.py` before connecting it to the others.
