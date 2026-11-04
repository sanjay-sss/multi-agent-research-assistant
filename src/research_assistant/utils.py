"""Small shared helpers."""

from __future__ import annotations

import re

_STOPWORDS = {
    "a", "about", "affect", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "its", "of", "on", "or", "over", "such",
    "that", "the", "their", "these", "they", "this", "to", "what", "when", "where", "which",
    "who", "why", "will", "with", "within",
}  # fmt: skip


def keyword_query(text: str, max_words: int = 8) -> str:
    """Shorten a natural-language question into a keyword search query.

    >>> keyword_query("What are the main challenges facing solid-state batteries in EVs?")
    'main challenges facing solid-state batteries EVs'
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-.']*", text)
    kept = [w.rstrip(".") for w in words if w.lower() not in _STOPWORDS]
    return " ".join(kept[:max_words])


def dedupe(items: list[str]) -> list[str]:
    """Strip whitespace and drop empty or duplicate strings (case-insensitive), keeping order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
