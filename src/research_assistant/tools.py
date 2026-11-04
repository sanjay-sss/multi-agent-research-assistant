"""External tools used by the agents."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable

from research_assistant.schemas import Source

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int], list[Source]]
"""Signature of a web search function: (query, max_results) -> sources."""

# Researchers run in parallel; DuckDuckGo throttles bursts of simultaneous
# queries, so only a couple of searches are allowed in flight at once.
_SEARCH_SLOTS = threading.Semaphore(2)


def duckduckgo_search(
    query: str,
    max_results: int = 5,
    *,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> list[Source]:
    """Search the web with DuckDuckGo (no API key required).

    Rate limits and network errors are retried with exponential backoff. If
    every attempt fails, the error is logged and an empty list is returned, so
    one failed search never crashes the whole research run.
    """
    from ddgs import DDGS

    results: list[dict] = []
    for attempt in range(1, attempts + 1):
        try:
            with _SEARCH_SLOTS:
                results = DDGS().text(query, max_results=max_results) or []
            break
        except Exception as exc:  # ddgs raises several types for rate limits / network issues
            if "no results" in str(exc).lower():
                return []  # a genuine empty result; retrying the same query won't help
            if attempt == attempts:
                logger.warning("Search failed for %r after %d attempts: %s", query, attempts, exc)
                return []
            delay = backoff_seconds * 2 ** (attempt - 1) + random.uniform(0, backoff_seconds)
            logger.info("Search attempt %d for %r failed (%s); retrying", attempt, query, exc)
            time.sleep(delay)

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
