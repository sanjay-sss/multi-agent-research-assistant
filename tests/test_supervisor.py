"""Unit tests for the supervisor's routing guardrails."""

from __future__ import annotations

from research_assistant.agents.supervisor import available_actions, pending_questions
from research_assistant.config import Settings
from research_assistant.schemas import Critique, Finding

SETTINGS = Settings(max_revisions=2, max_research_rounds=2)


def test_plans_first():
    assert available_actions({"topic": "t"}, SETTINGS)[0] == ["planner"]


def test_researches_pending_questions():
    state = {"topic": "t", "questions": ["A", "B"], "findings": [Finding(question="a")]}
    assert pending_questions(state) == ["B"]  # matching is case-insensitive
    assert available_actions(state, SETTINGS)[0] == ["researcher"]


def test_writes_after_research():
    state = {"topic": "t", "questions": ["A"], "findings": [Finding(question="A")]}
    assert available_actions(state, SETTINGS)[0] == ["writer"]


def test_reviews_each_new_draft():
    base = {"topic": "t", "questions": ["A"], "findings": [Finding(question="A")], "draft": "d"}
    assert available_actions({**base, "revision": 1}, SETTINGS)[0] == ["critic"]

    stale_review = Critique(score=5, approved=False)
    state = {**base, "revision": 2, "critique": stale_review, "critique_for_revision": 1}
    assert available_actions(state, SETTINGS)[0] == ["critic"]


def _reviewed(critique: Critique, revision: int = 1, rounds: int = 1) -> dict:
    return {
        "topic": "t",
        "questions": ["A"],
        "findings": [Finding(question="A")],
        "draft": "d",
        "revision": revision,
        "critique": critique,
        "critique_for_revision": revision,
        "research_rounds": rounds,
    }


def test_finishes_when_approved():
    state = _reviewed(Critique(score=9, approved=True))
    assert available_actions(state, SETTINGS)[0] == ["finalize"]


def test_finishes_at_revision_limit():
    state = _reviewed(Critique(score=4, approved=False), revision=2)
    assert available_actions(state, SETTINGS)[0] == ["finalize"]


def test_rejection_without_follow_ups_means_rewrite():
    state = _reviewed(Critique(score=5, approved=False, issues=["x"]))
    assert available_actions(state, SETTINGS)[0] == ["writer"]


def test_rejection_with_follow_ups_offers_a_choice():
    state = _reviewed(Critique(score=5, approved=False, follow_up_questions=["New Q"]))
    assert available_actions(state, SETTINGS)[0] == ["researcher", "writer"]


def test_already_asked_follow_ups_are_ignored():
    state = _reviewed(Critique(score=5, approved=False, follow_up_questions=["a"]))
    assert available_actions(state, SETTINGS)[0] == ["writer"]
