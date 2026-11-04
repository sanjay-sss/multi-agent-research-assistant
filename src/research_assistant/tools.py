"""External tools used by the agents."""

from __future__ import annotations

import logging
from collections.abc import Callable

from research_assistant.schemas import Source

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int], list[Source]]
"""Signature of a web search function: (query, max_results) -> sources."""


def duckduckgo_search(query: str, max_results: int = 5) -> list[Source]:
    """Search the web with DuckDuckGo (no API key required).

    Network errors and rate limits are logged and return an empty list, so one
    failed search never crashes the whole research run.
    """
    from ddgs import DDGS

    try:
        results = DDGS().text(query, max_results=max_results) or []
    except Exception as exc:  # ddgs raises several exception types for network/rate-limit issues
        logger.warning("Search failed for %r: %s", query, exc)
        return []

    sources: list[Source] = []
    seen: set[str] = set()
    for item in results:
        url = (item.get("href") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append(
            Source(
                title=(item.get("title") or url).strip(),
                url=url,
                snippet=(item.get("body") or "").strip(),
            )
        )
    return sources
