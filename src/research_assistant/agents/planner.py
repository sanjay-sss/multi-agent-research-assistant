"""Planner agent: breaks the topic into research questions."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_assistant import prompts
from research_assistant.config import Settings
from research_assistant.schemas import ResearchPlan
from research_assistant.state import ResearchState
from research_assistant.utils import dedupe


def make_planner(llm: BaseChatModel, settings: Settings):
    def planner(state: ResearchState) -> dict:
        plan: ResearchPlan = llm.with_structured_output(ResearchPlan).invoke(
            [
                SystemMessage(prompts.PLANNER.format(max_questions=settings.max_questions)),
                HumanMessage(f"Topic: {state['topic']}"),
            ]
        )
        questions = dedupe(plan.questions)[: settings.max_questions] or [state["topic"]]
        return {"questions": questions, "plan_rationale": plan.rationale}

    return planner
