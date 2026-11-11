"""Openverse image search: https://api.openverse.org/v1/images/ (anonymous access).

Openverse aggregates openly licensed works; it does not verify each work's license. Its terms ask
apps to attribute CC-licensed works and to say they were made using Openverse without being
endorsed by it (the chat UI shows both).

Everything Openverse returns is untrusted: each result is validated through `ImageResult`, results
that fail are dropped, and thumbnails must be Openverse's own https thumbnail URLs. The server only
calls the search endpoint; it never fetches image or page URLs itself.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import TypeAdapter, ValidationError

from app.models import ImageResult, WebUrl

logger = logging.getLogger(__name__)

OPENVERSE_IMAGES_URL = "https://api.openverse.org/v1/images/"
THUMBNAIL_HOST = "api.openverse.org"
MAX_IMAGES_PER_TURN = 3
PAGE_SIZE = 10  # a few spare results, since some are dropped by validation; anonymous limit is 20
MAX_QUERY_LENGTH = 200  # Openverse's documented limit for `q`
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
TIMEOUT = httpx.Timeout(6.0, connect=3.0, read=12.0)
USER_AGENT = "SupportIQ-RAG-Chatbot/0.1 (development)"

_LICENSE_CODE = re.compile(r"^[a-z0-9+-]{1,20}$")
_LICENSE_VERSION = re.compile(r"^[0-9.]{1,8}$")
_OPENVERSE_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_LICENSE_NAMES = {
    "cc0": "CC0",
    "pdm": "Public Domain Mark",
    "sampling+": "CC Sampling+",
    "nc-sampling+": "CC NC-Sampling+",
}
_web_url: TypeAdapter[str] = TypeAdapter(WebUrl)


class ImageSearchError(Exception):
    """The search could not be completed. The message is safe to log and show: no query, no URL."""


class ImageSearchProvider(Protocol):
    def search(self, query: str, limit: int = MAX_IMAGES_PER_TURN) -> list[ImageResult]: ...


class OpenverseImageSearch:
    """Anonymous Openverse client: one request per search, short timeout, no retries."""

    def __init__(self, client: httpx.Client | None = None, url: str = OPENVERSE_IMAGES_URL) -> None:
        self._client = client or httpx.Client(
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=False,
        )
        self._url = url

    def search(self, query: str, limit: int = MAX_IMAGES_PER_TURN) -> list[ImageResult]:
        """IN: a short plain-text query.  OUT: at most `limit` (<= 3) valid, non-mature images.

        Raises ImageSearchError for timeouts, network/HTTP errors, rate limiting and malformed responses.
        """
        query = " ".join(query.split())[:MAX_QUERY_LENGTH]
        if not query:
            return []
        params: dict[str, str | int] = {"q": query, "page_size": PAGE_SIZE, "mature": "false"}
        start = time.monotonic()
        try:
            response = self._client.get(self._url, params=params)
        except httpx.TimeoutException as exc:  # which limit expired (connect/read/write/pool) and when; no query or URL
            elapsed = time.monotonic() - start
            raise ImageSearchError(
                f"Openverse did not answer in time ({type(exc).__name__} after {elapsed:.1f}s)"
            ) from None
        except httpx.HTTPError as exc:  # str(exc) can contain the URL with the query, so only the type is kept
            raise ImageSearchError(f"could not reach Openverse ({type(exc).__name__})") from None
        if response.status_code == 429:
            raise ImageSearchError("Openverse rate limit reached (HTTP 429); try again later")
        if response.status_code != 200:
            raise ImageSearchError(f"Openverse returned HTTP {response.status_code}")
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise ImageSearchError("Openverse response too large")
        try:
            payload = response.json()
        except ValueError:
            raise ImageSearchError("Openverse returned invalid JSON") from None
        return parse_results(payload, limit=limit)


def parse_results(payload: Any, limit: int = MAX_IMAGES_PER_TURN) -> list[ImageResult]:
    """Valid images from an Openverse search response, in order, without duplicates; never more than 3."""
    limit = min(limit, MAX_IMAGES_PER_TURN)
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ImageSearchError("unexpected Openverse response format")
    images: list[ImageResult] = []
    seen: set[str] = set()
    for item in results:
        image = to_image(item)
        if image is None or image.thumbnail_url in seen:
            continue
        seen.add(image.thumbnail_url)
        images.append(image)
        if len(images) >= limit:
            break
    logger.debug("Openverse: kept %d image(s) from %d result(s)", len(images), len(results))
    return images


def to_image(item: Any) -> ImageResult | None:
    """One Openverse result as an ImageResult, or None if it is unsafe, mature or incomplete."""
    if not isinstance(item, dict) or item.get("mature") or item.get("unstable__sensitivity"):
        return None
    thumbnail = _text(item, "thumbnail")
    if not _is_openverse_thumbnail(thumbnail):
        return None
    license_name = _license_name(_text(item, "license").lower(), _text(item, "license_version"))
    if not license_name:
        return None  # a card without a license would be misleading attribution
    title = _text(item, "title") or "Untitled image"
    creator = _text(item, "creator")
    source_page = _optional_url(_text(item, "foreign_landing_url")) or _openverse_page(_text(item, "id"))
    if source_page is None:
        return None
    attribution = (
        _text(item, "attribution")
        or f'"{title}"{f" by {creator}" if creator else ""} is licensed under {license_name}.'
    )
    try:
        return ImageResult(
            title=title,
            thumbnail_url=thumbnail,
            source_page_url=source_page,
            creator=creator,
            creator_url=_optional_url(_text(item, "creator_url")),
            license_name=license_name,
            license_url=_optional_url(_text(item, "license_url")),
            attribution=attribution,
        )
    except ValidationError:
        return None


def _text(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    return value.strip() if isinstance(value, str) else ""


def _optional_url(value: str) -> str | None:
    if not value:
        return None
    try:
        return _web_url.validate_python(value)
    except ValidationError:
        return None


def _is_openverse_thumbnail(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "https" and parts.hostname == THUMBNAIL_HOST and parts.path.startswith("/v1/images/")


def _openverse_page(identifier: str) -> str | None:
    """Openverse's own page for the image, used when the original page URL is missing or unsafe."""
    return f"https://openverse.org/image/{identifier}" if _OPENVERSE_ID.match(identifier) else None


def _license_name(code: str, version: str) -> str:
    """e.g. ("by-nc", "2.0") -> "CC BY-NC 2.0"; "" when the code is missing or odd."""
    if not _LICENSE_CODE.match(code):
        return ""
    name = _LICENSE_NAMES.get(code, f"CC {code.upper()}")
    return f"{name} {version}" if _LICENSE_VERSION.match(version) else name
