"""image_search tool: optional image cards from an image provider (Openverse).

Not part of `get_tools()`; app/deps.py adds it when IMAGE_SEARCH_PROVIDER is set. The rules
(docs/web-and-image-search.md):
  - at most one search per chat turn, enforced here, not only in the prompt;
  - the query must come from the user's own words: a model-written query that uses wording found
    only in the retrieved document excerpts is replaced by keywords from the user's question, and
    if none are left, image search is skipped;
  - images go to the UI as cards (graph state `images`), never into the answer text;
  - any provider failure becomes a short tool error and the answer continues without images.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.models import ImageResult
from app.search.images import MAX_IMAGES_PER_TURN, ImageSearchError, ImageSearchProvider

logger = logging.getLogger(__name__)

TOOL_NAME = "image_search"
MAX_QUERY_WORDS = 6
NO_IMAGE_LINKS = "Do not include image links."
_WORD = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")
# Words that carry no subject and never count as "copied" (they also never make a useful query).
_STOPWORDS = frozenset(
    """a an and are as at be but by can could did do does for from had has have how i if in into is it its
    me my of on or our please should show so some than that the their them then there these this those to
    us was we what when where which who why will with would you your picture pictures image images photo
    photos look looks like see example examples""".split()
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def safe_image_query(requested: str, user_texts: list[str], excerpts: str) -> str | None:
    """A short query built only from the user's own words, or None if none can be formed.

    IN:  requested   the query the model asked for (it has seen the documents, so it is not trusted)
         user_texts  everything the user typed in this session, newest last
         excerpts    the retrieved document text for this turn
    """
    user_words = {w for text in user_texts for w in _words(text)}
    document_only = {w for w in _words(excerpts) if w not in user_words}
    asked = [w for w in _words(requested) if w not in _STOPWORDS][:MAX_QUERY_WORDS]
    if asked and not any(w in document_only for w in asked):
        return " ".join(asked)
    fallback = [w for w in _words(user_texts[-1] if user_texts else "") if w not in _STOPWORDS][:MAX_QUERY_WORDS]
    return " ".join(fallback) or None


def _turn_messages(messages: list[Any]) -> tuple[list[str], list[AIMessage]]:
    """All user texts in the session, and this turn's AI messages (after the newest user message)."""
    user_texts = [str(m.text) for m in messages if isinstance(m, HumanMessage)]
    turn: list[AIMessage] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage):
            turn.insert(0, message)
    return user_texts, turn


def _first_call_id(turn: list[AIMessage]) -> str | None:
    for message in turn:
        for call in message.tool_calls:
            if call["name"] == TOOL_NAME:
                return call["id"]
    return None


def _summary(images: list[ImageResult]) -> str:
    if not images:
        return "No suitable openly licensed images were found. Answer without images. Do not include image links."
    titles = "; ".join(f'"{i.title[:80]}" ({i.license_name})' for i in images)
    return (
        f"{len(images)} openly licensed image(s) will be shown to the user as cards under your answer, "
        f"with credits and links. Do not add image links or describe details you cannot see. Titles: {titles}"
    )


def make_image_search_tool(provider: ImageSearchProvider) -> BaseTool:
    """Build the `image_search` tool around a provider (a fake one in tests)."""

    @tool(TOOL_NAME)
    def image_search(
        query: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[Any]:
        """Find up to three openly licensed images (photos, diagrams, illustrations) to show next to your answer.

        Use it ONLY when seeing something materially helps the user: what a device, part, connector, place,
        plant, animal or symbol looks like. Do NOT use it for policies, procedures, numbers, dates or other
        text answers, and never more than once per question.
        query: 2-6 plain words naming the visual subject, taken from the user's own question. Never put text
        from the documents in the query.
        """
        user_texts, turn = _turn_messages(state.get("messages", []))

        def reply(text: str, images: list[ImageResult] | None = None) -> Command[Any]:
            message = ToolMessage(content=text, tool_call_id=tool_call_id, name=TOOL_NAME)
            update: dict[str, Any] = {"messages": [message]}
            if images:
                update["images"] = images
            return Command(update=update)

        # Outcome logs are fixed messages plus a count: never the query, URLs or document text.
        if _first_call_id(turn) != tool_call_id:
            logger.info("Image search skipped: already used for this question")
            return reply(
                f"Error: image search was already used for this question. Answer without more images. {NO_IMAGE_LINKS}"
            )
        safe_query = safe_image_query(query, user_texts, str(state.get("context", "")))
        if safe_query is None:
            logger.info("Image search skipped: no safe query words")
            return reply(f"Image search skipped: no safe search words. Answer without images. {NO_IMAGE_LINKS}")
        try:
            images = provider.search(safe_query, limit=MAX_IMAGES_PER_TURN)[:MAX_IMAGES_PER_TURN]
        except ImageSearchError as exc:
            logger.warning("Image search failed: %s", exc)  # the message never contains the query
            return reply(f"Error: image search is unavailable ({exc}). Answer without images. {NO_IMAGE_LINKS}")
        if images:
            logger.info("Image search returned %d image(s)", len(images))
        else:
            logger.info("Image search found no valid images")
        return reply(_summary(images), images)

    return image_search
