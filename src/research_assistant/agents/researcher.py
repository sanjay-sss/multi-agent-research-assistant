"""Researcher agent: searches the web for one question and extracts key facts.

The supervisor fans out one researcher per open question using LangGraph's
``Send`` API, so all questions are researched in parallel.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_assistant import prompts
from research_assistant.config import Settings
from research_assistant.schemas import Finding, KeyPoint, ResearchNotes
from research_assistant.state import ResearchTask
from research_assistant.tools import SearchFn


def make_researcher(llm: BaseChatModel, settings: Settings, search: SearchFn):
    def researcher(task: ResearchTask) -> dict:
        question = task["question"]
        sources = search(question, settings.results_per_question)

        # Always record a finding, even an empty one, so the supervisor marks
        # the question as researched and never loops on it.
        if not sources:
            return {"findings": [Finding(question=question)]}

        results = "\n\n".join(
            f"[{i}] {s.title}\nURL: {s.url}\n{s.snippet}" for i, s in enumerate(sources, start=1)
        )
        notes: ResearchNotes = llm.with_structured_output(ResearchNotes).invoke(
            [
                SystemMessage(prompts.RESEARCHER),
                HumanMessage(
                    f"Overall topic: {task['topic']}\n"
                    f"Research question: {question}\n\n"
                    f"Search results:\n\n{results}"
                ),
            ]
        )

        # Keep only citations that point at real search results.
        key_points = []
        for kp in notes.key_points:
            ids = sorted({i for i in kp.source_ids if 1 <= i <= len(sources)})
            if ids and kp.point.strip():
                key_points.append(KeyPoint(point=kp.point.strip(), source_ids=ids))

        return {"findings": [Finding(question=question, key_points=key_points, sources=sources)]}

    return researcher
