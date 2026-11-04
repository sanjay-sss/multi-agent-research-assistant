"""Pydantic models shared by the agents.

The LLM-facing models (``ResearchPlan``, ``ResearchNotes``, ``Critique``,
``SupervisorDecision``) are used with structured output, so their field
descriptions double as instructions to the model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Data gathered by the team
# --------------------------------------------------------------------------- #


class Source(BaseModel):
    """A single web search result."""

    title: str
    url: str
    snippet: str = ""


class KeyPoint(BaseModel):
    """One fact extracted from the search results, with its supporting sources."""

    point: str = Field(description="A specific, self-contained fact that answers the question.")
    source_ids: list[int] = Field(
        description="Numbers of the search results (as shown in brackets) that support this point."
    )


class Finding(BaseModel):
    """Everything the researcher learned about one question."""

    question: str
    key_points: list[KeyPoint] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Structured outputs returned by the LLM agents
# --------------------------------------------------------------------------- #


class ResearchPlan(BaseModel):
    """Planner output."""

    questions: list[str] = Field(
        description="Focused, non-overlapping research questions; each works as a web search query."
    )
    rationale: str = Field(
        description="One or two sentences on why these questions cover the topic."
    )


class ResearchNotes(BaseModel):
    """Researcher output for one question."""

    key_points: list[KeyPoint] = Field(
        description="Key facts answering the question, each citing the search results it came from."
    )


class Critique(BaseModel):
    """Critic output."""

    score: int = Field(description="Overall quality score from 1 (poor) to 10 (excellent).")
    approved: bool = Field(description="True if the report is ready to publish as-is.")
    strengths: list[str] = Field(default_factory=list, description="What the draft does well.")
    issues: list[str] = Field(
        default_factory=list, description="Concrete, actionable problems the writer must fix."
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Up to 3 new web research questions, only if key information is missing.",
    )


class SupervisorDecision(BaseModel):
    """Supervisor output when it has a real choice to make."""

    next: Literal["researcher", "writer"] = Field(
        description="'researcher' to gather more evidence, 'writer' to revise the draft."
    )
    reason: str = Field(description="One sentence explaining the decision.")
