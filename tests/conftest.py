"""Test doubles: a scripted fake LLM and a deterministic fake search.

These let the full multi-agent graph run offline, with no API key, so the
routing logic, loops and guardrails can be tested exactly.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict

import pytest
from langchain_core.messages import AIMessage

from research_assistant.schemas import Source


class FakeLLM:
    """Returns scripted responses.

    ``structured`` maps a Pydantic schema to a list of responses returned in
    order by ``with_structured_output(schema).invoke``; the last response
    repeats once the list is exhausted. A response that is an Exception is
    raised instead. ``texts`` works the same way for plain ``invoke`` calls.
    """

    def __init__(self, structured: dict | None = None, texts: list[str] | None = None):
        self.structured = {schema: list(items) for schema, items in (structured or {}).items()}
        self.texts = list(texts or ["# Report\n\nA claim [1]."])
        self.calls: dict[str, int] = defaultdict(int)
        self.prompts: dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()  # researchers call the LLM from parallel threads

    @staticmethod
    def _next(queue: list):
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    def with_structured_output(self, schema):
        llm = self

        class _Runner:
            def invoke(self, messages):
                with llm._lock:
                    llm.calls[schema.__name__] += 1
                    llm.prompts[schema.__name__].append(messages)
                    return llm._next(llm.structured[schema])

        return _Runner()

    def invoke(self, messages):
        with self._lock:
            self.calls["text"] += 1
            self.prompts["text"].append(messages)
            return AIMessage(content=self._next(self.texts))


def fake_search(query: str, max_results: int) -> list[Source]:
    """Two predictable results per query, with URLs unique to the query."""
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return [
        Source(title=f"{query} result {i}", url=f"https://example.com/{slug}/{i}", snippet="...")
        for i in range(1, 3)
    ]


@pytest.fixture
def search():
    return fake_search
