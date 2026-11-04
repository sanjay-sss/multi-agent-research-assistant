"""Supervisor agent: decides which agent works next.

The supervisor combines deterministic guardrails with LLM judgment:

1. ``available_actions`` works out which steps are valid from the current state
   (e.g. you can't write before researching, and the run must stop after
   ``max_revisions`` drafts). This guarantees the workflow always terminates.
2. When more than one step is valid (the critic rejected a draft and suggested
   new research), the LLM decides whether to research more or revise.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from research_assistant import prompts
from research_assistant.config import Settings
from research_assistant.schemas import SupervisorDecision
from research_assistant.state import ResearchState

logger = logging.getLogger(__name__)

NODES = ["planner", "researcher", "writer", "critic", "finalize"]


def pending_questions(state: ResearchState) -> list[str]:
    """Questions that have not been researched yet."""
    done = {f.question.lower() for f in state.get("findings", [])}
    return [q for q in state.get("questions", []) if q.lower() not in done]


def new_follow_up_questions(state: ResearchState) -> list[str]:
    """Critic follow-up questions that haven't been asked before."""
    critique = state.get("critique")
    if critique is None:
        return []
    asked = {q.lower() for q in state.get("questions", [])}
    return [q for q in critique.follow_up_questions if q.lower() not in asked]


def available_actions(state: ResearchState, settings: Settings) -> tuple[list[str], str]:
    """Return the valid next steps and a short explanation."""
    if not state.get("questions"):
        return ["planner"], "No research plan yet."

    pending = pending_questions(state)
    if pending:
        return ["researcher"], f"{len(pending)} question(s) need research."

    if not state.get("draft"):
        return ["writer"], "Research is complete; write the first draft."

    revision = state.get("revision", 0)
    critique = state.get("critique")
    if critique is None or state.get("critique_for_revision", 0) < revision:
        return ["critic"], f"Draft v{revision} needs review."

    if critique.approved:
        return ["finalize"], f"Critic approved draft v{revision} ({critique.score}/10)."

    if revision >= settings.max_revisions:
        return ["finalize"], (
            f"Revision limit reached ({settings.max_revisions}); publishing draft v{revision} "
            f"({critique.score}/10)."
        )

    options = ["writer"]
    if (
        new_follow_up_questions(state)
        and state.get("research_rounds", 0) < settings.max_research_rounds
    ):
        options.insert(0, "researcher")
    return options, f"Critic rejected draft v{revision} ({critique.score}/10); revise it."


def make_supervisor(llm: BaseChatModel, settings: Settings):
    def decide(state: ResearchState, options: list[str]) -> tuple[str, str]:
        critique = state["critique"]
        summary = (
            f"Topic: {state['topic']}\n"
            f"Draft version: {state.get('revision', 0)} of at most {settings.max_revisions}\n"
            f"Critic score: {critique.score}/10\n"
            "Issues:\n" + "\n".join(f"- {i}" for i in critique.issues) + "\n"
            "Follow-up research questions:\n"
            + "\n".join(f"- {q}" for q in new_follow_up_questions(state))
        )
        try:
            decision: SupervisorDecision = llm.with_structured_output(SupervisorDecision).invoke(
                [SystemMessage(prompts.SUPERVISOR), HumanMessage(summary)]
            )
        except Exception as exc:  # a failed routing call shouldn't kill the run
            logger.warning("Supervisor decision failed, defaulting to writer: %s", exc)
            return "writer", "Routing call failed; revising with existing evidence."
        if decision.next not in options:
            return "writer", "Invalid choice from model; revising with existing evidence."
        return decision.next, decision.reason

    def supervisor(state: ResearchState) -> dict:
        options, reason = available_actions(state, settings)
        choice = options[0]
        if len(options) > 1:
            choice, reason = decide(state, options)

        update: dict = {"next": choice, "log": [f"{choice}: {reason}"]}
        if choice == "researcher":
            update["research_rounds"] = state.get("research_rounds", 0) + 1
            if not pending_questions(state):
                # Follow-up round requested by the critic.
                update["questions"] = new_follow_up_questions(state)[: settings.max_questions]
        return update

    return supervisor


def route_after_supervisor(state: ResearchState):
    """Conditional edge: go to the chosen node, fanning out researchers in parallel."""
    if state["next"] == "researcher":
        return [
            Send("researcher", {"topic": state["topic"], "question": q})
            for q in pending_questions(state)
        ]
    return state["next"]
