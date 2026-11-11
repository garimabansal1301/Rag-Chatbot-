# RAG Chatbot — Training Project

A small chat app with a sessions sidebar (create, rename, delete) and a chat pane. The web pages, the JSON
session storage and the upload endpoint already work. **You build the AI parts** — the LLM, the LangGraph engine,
LangSmith tracing, chunking, embeddings, a vector store, ingestion, retrieval and tools — one milestone at a time.

Read [TRAINEE_TASKS.md](TRAINEE_TASKS.md) for the milestones (M1–M8, in order).

## 1. Install the tools (macOS)

```bash
brew install uv
```

`uv` installs Python 3.12 and every dependency for you. No separate `pip` or `venv` steps.

## 2. Set up the project

```bash
cd rag-chatbot
uv sync                  # creates .venv and installs dependencies
cp .env.example .env     # then edit .env: set SELF_CHECK_KEY to your name, add API keys later
```

## 3. Run the app

```bash
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Create a session and send a message. Until you finish Milestone 1 the reply is a
placeholder, and the sidebar lists the modules that are not built yet.

## 4. Test your work

| What | Command |
|---|---|
| App plumbing tests (always green) | `uv run pytest` |
| Milestone contract tests (pass = milestone done) | `uv run pytest -m contract` |
| One milestone only | `uv run pytest -m contract tests/contract/test_m3_chunking.py` |
| Pass/fail summary per milestone | `uv run python self_check.py` |
| Lint and types | `uv run ruff check` and `uv run mypy app` |

`self_check.py` prints PASS or FAIL per milestone and writes `self_check_report.json`. It needs `SELF_CHECK_KEY` in `.env`.
Passing it is a good sign, not a grade — your trainer runs a separate, stricter evaluation.

The fakes in `tests/fakes.py` (a fake chat model, a hash embedder, an in-memory store) let you test one module at a
time without any API key.

## 5. Index documents

```bash
uv run python -m app.ingest
```

Indexes everything in `data/documents/`. You can also upload `.md` / `.txt` files in the browser.

## 6. MCP (Model Context Protocol)

MCP is an open protocol through which an AI application (the *host*, here SupportPilot) discovers and calls tools
served by a separate program (an MCP *server*). The app ships a small local server that exposes the
`calculator` and `current_datetime` tools.

**Ordinary LangChain tool vs MCP tool.** An ordinary tool (`app/tools/calculator.py`) is a Python function inside the
app process; the app knows it at import time. An MCP tool lives in another process: the app starts the server,
asks it which tools it has (`tools/list`), receives their names, descriptions and JSON schemas, and sends each call
over the protocol (`tools/call`). The same server could be used by any MCP host, and the app could use other
servers without new Python code. Both kinds end up as LangChain tools in the same LangGraph `ToolNode`.

```
LangGraph ToolNode ─► MCP tool wrapper (app/mcp/client.py) ─► MCP session ─stdio─► app/mcp/server.py (FastMCP)
                                                                                    └─ app/tools/calculator.py logic
```

- `app/mcp/server.py`: FastMCP server, **stdio only** (no network port), exactly two read-only tools, input
  validated by its schema (`expression`: 1–200 characters), arithmetic parsed with `ast` (never `eval`/`exec`).
- `app/mcp/client.py`: `MCPToolset` opens **one** session at app start-up (`MultiServerMCPClient.session` +
  `load_mcp_tools`) and closes it at shutdown, which ends the server process. The tools are wrapped so the
  synchronous graph can call them, each call with a 30 s timeout. The server is started as
  `python -m app.mcp.server` from this folder with a minimal environment (no API keys); no other command can run.
- `app/deps.py`: the MCP tools *replace* the local tools of the same name (never both). If the server cannot
  start, the app logs a warning, shows it in the sidebar's status list, and keeps using the local tools.

**Enable it:** set `MCP_ENABLED=true` in `.env` (default `false`) and restart the app. `GET /health` reports
`"mcp": "connected"` (or `"disabled"` / `"unavailable"`).

**Run the server on its own** (it waits for an MCP client on stdin; stop it with Ctrl+C):

```bash
uv run python -m app.mcp.server
```

**Test:** `uv run pytest tests/test_mcp.py` starts the real server as a subprocess and uses a fake chat model
(no API key, no network).

## Adding provider packages

```bash
uv add langchain-anthropic      # or langchain-openai
uv add chromadb                 # or faiss-cpu, lancedb ... any vector store you pick
```

Set `LLM_PROVIDER`, `LLM_MODEL` and the API key in `.env`. Never commit `.env`.

## Layout

```
app/
  main.py, config.py, deps.py, services.py, web.py   working plumbing — leave alone
  api/            JSON endpoints (sessions, chat, documents)
  storage/        one JSON file per session
  chat/llm.py            M1  get_llm
  chat/engine.py         M1, M7, M8  LangGraph engine
  observability.py       M2  LangSmith
  vectordb/chunking.py   M3
  vectordb/embeddings.py M4
  vectordb/store.py      M5
  ingest/                M6
  vectordb/retriever.py  M7
  tools/                 M8
  mcp/                   MCP server (stdio) and client, off unless MCP_ENABLED=true
data/documents/  sample documents      data/sessions/  chat history (JSON)
tests/           plumbing tests + tests/contract (one file per milestone)
templates/, static/   the server-rendered UI
```

Models in `app/models.py` are the contract between your modules and the trainer's evaluator — don't rename fields.
