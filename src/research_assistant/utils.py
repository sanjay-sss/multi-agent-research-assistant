"""Small shared helpers."""

from __future__ import annotations


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
