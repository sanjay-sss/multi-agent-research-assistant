"""Critic agent: reviews the draft against the evidence."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_assistant import prompts
from research_assistant.config import Settings
from research_assistant.evidence import build_evidence
from research_assistant.schemas import Critique
from research_assistant.state import ResearchState
from research_assistant.utils import dedupe


def make_critic(llm: BaseChatModel, settings: Settings):
    def critic(state: ResearchState) -> dict:
        evidence = build_evidence(state.get("findings", []))
        review: Critique = llm.with_structured_output(Critique).invoke(
            [
                SystemMessage(prompts.CRITIC.format(approval_score=settings.approval_score)),
                HumanMessage(
                    f"Topic: {state['topic']}\n\n"
                    f"## Evidence\n\n{evidence.markdown or '(no evidence)'}\n\n"
                    f"## Draft report\n\n{state['draft']}"
                ),
            ]
        )

        # Enforce the rules in code instead of trusting the model to follow them.
        score = max(1, min(10, review.score))
        review = review.model_copy(
            update={
                "score": score,
                "approved": review.approved and score >= settings.approval_score,
                "follow_up_questions": dedupe(review.follow_up_questions)[:3],
            }
        )
        return {"critique": review, "critique_for_revision": state.get("revision", 0)}

    return critic
