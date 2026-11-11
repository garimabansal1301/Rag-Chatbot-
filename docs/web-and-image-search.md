# Web search and image cards: design rules

Status: **Phase 2** — image search via Openverse is implemented (`app/search/images.py`,
`app/tools/image_search.py`); Google Search grounding is not implemented yet. Both are off by default:

```
WEB_SEARCH_ENABLED=false       # true / false
IMAGE_SEARCH_PROVIDER=none     # none | openverse
```

## 1. Privacy boundary (internal documents vs. the web)

Internal RAG answering is unchanged: retrieved excerpts go to the configured LLM exactly as today.
That already means excerpts leave this machine — they are sent to the LLM provider (on Gemini's
free tier Google may use them). Nothing here makes the system private or confidential.

For web grounding and image search the rules are stricter:

| Data | Normal RAG call | Grounded Gemini request | Image search query |
|---|---|---|---|
| User's current question | yes | yes (verbatim) | no — a short visual subject only |
| Earlier **user** turns | yes | only if needed to resolve a follow-up, at most the previous one | no |
| Earlier **assistant** answers | yes | **no** (they may quote internal documents) | no |
| Retrieved document excerpts / snippets | yes | **never** | **never** |
| Tool results (calculator, date) | yes | no | no |

How the code will enforce it (Phase 3):

- The grounded request is built by a function that receives only the user's text — it never gets
  the graph state, so excerpts cannot reach it by accident. Tests assert that no retrieved chunk
  text appears in the request.
- The main model only *signals* that web results are wanted (`web_search` tool call); its
  arguments are ignored, because that model has seen the excerpts and could copy them into a query.
- Image queries (implemented): the model's query is cut to 6 words; if any meaningful word in it
  appears in the retrieved excerpts but in none of the user's own messages, it is replaced by
  keywords from the user's latest question, and if none remain the search is skipped
  (`safe_image_query`). At most one search per turn is enforced in code, by tool-call id.

## 2. Combining documents and the web (Google's grounding terms)

Google's Gemini API terms forbid modifying grounded results or interspersing other content with
them, and require showing the **Search Suggestions** with the grounded result.

So a turn that needs both is two separate operations:

1. the normal RAG answer → `Message.content` + `Message.sources` (document citations, unchanged);
2. a separate grounded answer from the question alone → `Message.web` (`WebGrounding`: answer,
   web sources, search queries, Search Suggestions HTML), shown **unmodified in its own block**.

The two are displayed next to each other, never merged into one text. A purely web question gets a
short document-side answer (e.g. that the documents do not cover it) plus the web block.

## 3. Rendering rules (Phase 3 UI)

- `WebGrounding.answer` is rendered as escaped text (Jinja autoescape), unmodified.
- `search_suggestions_html` is Google's HTML and must be shown as provided, so it is **not**
  sanitised. It is only ever rendered as
  `<iframe sandbox="allow-popups allow-popups-to-escape-sandbox" srcdoc="{{ html }}" referrerpolicy="no-referrer">`
  — no `allow-scripts`, no `allow-same-origin`, so it cannot run code in, or read from, our page.
  It is never inserted with `|safe` into the page itself.
- Links (web sources, image source pages, creators, licenses) open in a new tab with
  `rel="noopener noreferrer nofollow"` and nothing between the link and the destination.
- Image cards show title, creator (linked when available), license (linked), "View source" and the
  Openverse notice ("Images via Openverse; not endorsed by Openverse"). Broken thumbnails remove
  their card; an empty strip is hidden.

## 4. Untrusted provider data

- `WebSource`, `WebGrounding` and `ImageResult` validate at construction: public `http(s)` hosts
  only (no IP addresses, `localhost`, `.local`/`.internal` names, credentials or non-default
  ports), `https` for anything the browser loads (thumbnails, web source links), text cleaned of
  control characters and length-capped.
- Provider adapters will build these models item by item and **drop** invalid items, so one bad
  result never fails a chat turn. Thumbnails will additionally be limited to Openverse's own host.
- The server never fetches image or page URLs itself (only the provider's search API), so result
  URLs cannot be used to make the server request internal addresses.

## 5. Open points for later phases

- Web grounding is Gemini-only; with another `LLM_PROVIDER`, `WEB_SEARCH_ENABLED=true` will be
  ignored with a warning (decided in Phase 3, not enforced at settings load).
- Openverse attribution strings and license metadata are provided by Openverse, which does not
  verify licenses.
- Rate limits: Openverse anonymous limits are unpublished; Gemini grounding has 5,000 free search
  queries/month on Gemini 3+ models, then billed per query.
