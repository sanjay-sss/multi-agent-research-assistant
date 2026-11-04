"""Graph state definitions."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from research_assistant.schemas import Critique, Finding


class ResearchState(TypedDict, total=False):
    """Shared state that flows through the whole graph.

    Fields annotated with ``operator.add`` are *accumulated*: each node returns
    only the new items and LangGraph appends them. This is what lets several
    researcher agents run in parallel and all write their findings safely.
    """

    topic: str

    # Planning / research
    plan_rationale: str
    questions: Annotated[list[str], operator.add]
    findings: Annotated[list[Finding], operator.add]
    research_rounds: int

    # Writing / review
    draft: str
    revision: int  # number of drafts written so far
    critique: Critique
    critique_for_revision: int  # which draft the current critique refers to

    # Control flow
    next: str
    log: Annotated[list[str], operator.add]

    # Output
    report: str


class ResearchTask(TypedDict):
    """Private input sent to one parallel researcher via ``Send``."""

    topic: str
    question: str
