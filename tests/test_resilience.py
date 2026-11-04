"""Tests for retrying transient failures (model overload, search rate limits)."""

from __future__ import annotations

import pytest
from conftest import FakeLLM, fake_search
from langchain_core.exceptions import ModelAPIError, ModelNotFoundError, ModelRateLimitError
from langgraph.types import RetryPolicy

from research_assistant.config import Settings
from research_assistant.graph import RECURSION_LIMIT, build_graph, is_transient_model_error
from research_assistant.schemas import Critique, KeyPoint, ResearchNotes, ResearchPlan
from research_assistant.tools import duckduckgo_search
from research_assistant.utils import keyword_query

FAST_RETRY = RetryPolicy(
    initial_interval=0.01, max_attempts=3, jitter=False, retry_on=is_transient_model_error
)
PLAN = ResearchPlan(questions=["Q1"], rationale="r")
NOTES = ResearchNotes(key_points=[KeyPoint(point="Fact", source_ids=[1])])
APPROVED = Critique(score=9, approved=True)


def _run(llm: FakeLLM) -> dict:
    graph = build_graph(llm=llm, settings=Settings(), search=fake_search, retry_policy=FAST_RETRY)
    return graph.invoke({"topic": "t"}, config={"recursion_limit": RECURSION_LIMIT})


def test_transient_error_classification():
    assert is_transient_model_error(ModelAPIError("503 overloaded"))
    assert is_transient_model_error(ModelRateLimitError("429"))
    assert not is_transient_model_error(ModelNotFoundError("404 no such model"))
    assert not is_transient_model_error(ValueError("bug"))
    daily = ModelRateLimitError("429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert not is_transient_model_error(daily)


def test_agent_step_retries_after_model_overload():
    llm = FakeLLM(
        structured={
            ResearchPlan: [PLAN],
            ResearchNotes: [NOTES],
            Critique: [ModelAPIError("503 high demand"), APPROVED],
        }
    )

    state = _run(llm)

    assert llm.calls["Critique"] == 2  # failed once, retried, succeeded
    assert state["critique"].approved
    assert state["report"]


def test_permanent_model_error_is_not_retried():
    llm = FakeLLM(structured={ResearchPlan: [ModelNotFoundError("404"), PLAN]})

    with pytest.raises(ModelNotFoundError):
        _run(llm)
    assert llm.calls["ResearchPlan"] == 1


class _FlakyDDGS:
    """Stands in for ddgs.DDGS: fails a set number of times, then returns results."""

    failures_left = 0
    calls = 0

    def text(self, query, max_results=5):
        type(self).calls += 1
        if type(self).failures_left > 0:
            type(self).failures_left -= 1
            raise RuntimeError("202 Ratelimit")
        return [
            {"title": "A", "href": "https://a.example", "body": "a"},
            {"title": "A dup", "href": "https://a.example", "body": "dup"},
            {"title": "B", "href": "https://b.example", "body": "b"},
        ]


@pytest.fixture
def flaky_ddgs(monkeypatch):
    _FlakyDDGS.failures_left = 0
    _FlakyDDGS.calls = 0
    monkeypatch.setattr("ddgs.DDGS", _FlakyDDGS)
    return _FlakyDDGS


def test_search_retries_rate_limits(flaky_ddgs):
    flaky_ddgs.failures_left = 2

    sources = duckduckgo_search("q", 5, attempts=3, backoff_seconds=0)

    assert flaky_ddgs.calls == 3
    assert [s.url for s in sources] == ["https://a.example", "https://b.example"]  # deduped


def test_keyword_query():
    assert (
        keyword_query("What are the main challenges facing solid-state batteries in EVs?")
        == "main challenges facing solid-state batteries EVs"
    )


def test_researcher_retries_with_keywords_when_question_finds_nothing():
    queries: list[str] = []

    def picky_search(query: str, n: int):
        queries.append(query)
        return [] if query.endswith("?") else fake_search(query, n)

    llm = FakeLLM(
        structured={
            ResearchPlan: [ResearchPlan(questions=["How do heat pumps work?"], rationale="r")],
            ResearchNotes: [NOTES],
            Critique: [APPROVED],
        }
    )
    graph = build_graph(llm=llm, settings=Settings(), search=picky_search, retry_policy=FAST_RETRY)
    state = graph.invoke({"topic": "t"}, config={"recursion_limit": RECURSION_LIMIT})

    assert queries == ["How do heat pumps work?", "heat pumps work"]
    assert state["findings"][0].sources  # found via the keyword query


def test_search_does_not_retry_genuine_empty_results(flaky_ddgs, monkeypatch):
    def no_results(self, query, max_results=5):
        type(self).calls += 1
        raise RuntimeError("No results found.")

    monkeypatch.setattr(flaky_ddgs, "text", no_results)

    assert duckduckgo_search("q", 5, attempts=3, backoff_seconds=0) == []
    assert flaky_ddgs.calls == 1


def test_search_gives_up_gracefully(flaky_ddgs):
    flaky_ddgs.failures_left = 10

    assert duckduckgo_search("q", 5, attempts=2, backoff_seconds=0) == []
    assert flaky_ddgs.calls == 2
