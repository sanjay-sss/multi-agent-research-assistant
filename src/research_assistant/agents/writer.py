"""Writer agent: drafts (and revises) the report from the evidence."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_assistant import prompts
from research_assistant.evidence import build_evidence
from research_assistant.llm import message_text
from research_assistant.state import ResearchState


def make_writer(llm: BaseChatModel):
    def writer(state: ResearchState) -> dict:
        evidence = build_evidence(state.get("findings", []))
        evidence_text = evidence.markdown if not evidence.is_empty else "(no evidence was found)"

        request = f"Topic: {state['topic']}\n\n## Evidence\n\n{evidence_text}"

        critique = state.get("critique")
        if state.get("draft") and critique is not None:
            issues = (
                "\n".join(f"- {issue}" for issue in critique.issues) or "- (no specific issues)"
            )
            request += "\n\n" + prompts.WRITER_REVISION.format(
                draft=state["draft"], score=critique.score, issues=issues
            )

        response = llm.invoke([SystemMessage(prompts.WRITER), HumanMessage(request)])
        return {"draft": message_text(response), "revision": state.get("revision", 0) + 1}

    return writer
