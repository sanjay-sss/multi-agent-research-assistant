"""Assemble the agents into a LangGraph state machine.

Flow (the supervisor runs between every step)::

    START -> supervisor -> planner -> supervisor -> researcher x N (parallel)
          -> supervisor -> writer -> supervisor -> critic -> supervisor
          -> (writer | researcher | finalize) ... -> finalize -> END
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from research_assistant.agents import (
    make_critic,
    make_planner,
    make_researcher,
    make_supervisor,
    make_writer,
    route_after_supervisor,
)
from research_assistant.agents.supervisor import NODES
from research_assistant.config import Settings
from research_assistant.evidence import build_evidence, render_sources
from research_assistant.llm import create_llm
from research_assistant.state import ResearchState
from research_assistant.tools import SearchFn, duckduckgo_search

# Number of graph steps allowed before LangGraph aborts. Generous on purpose:
# the supervisor's guardrails are what actually bound the run.
RECURSION_LIMIT = 100


def finalize(state: ResearchState) -> dict:
    """Append a Sources section built from real search results to the final draft."""
    evidence = build_evidence(state.get("findings", []))
    draft = state.get("draft", "").rstrip()
    sources = render_sources(draft, evidence.sources)
    report = f"{draft}\n\n{sources}" if sources else draft
    return {"report": report + "\n"}


def build_graph(
    llm: BaseChatModel | None = None,
    settings: Settings | None = None,
    search: SearchFn | None = None,
    checkpointer=None,
):
    """Build and compile the research graph.

    Args:
        llm: Chat model shared by all agents. Defaults to Gemini (needs GOOGLE_API_KEY).
        settings: Workflow settings. Defaults to values from the environment.
        search: Web search function. Defaults to DuckDuckGo.
        checkpointer: Optional LangGraph checkpointer for persistence / resuming.
    """
    settings = settings or Settings.from_env()
    llm = llm if llm is not None else create_llm(settings)
    search = search or duckduckgo_search

    builder = StateGraph(ResearchState)
    builder.add_node("supervisor", make_supervisor(llm, settings))
    builder.add_node("planner", make_planner(llm, settings))
    builder.add_node("researcher", make_researcher(llm, settings, search))
    builder.add_node("writer", make_writer(llm))
    builder.add_node("critic", make_critic(llm, settings))
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_after_supervisor, NODES)
    for worker in ("planner", "researcher", "writer", "critic"):
        builder.add_edge(worker, "supervisor")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def mermaid_diagram() -> str:
    """Mermaid source for the graph (no API key needed)."""

    class _DiagramOnly:  # never called; drawing the graph doesn't invoke the LLM
        pass

    graph = build_graph(llm=_DiagramOnly(), settings=Settings(), search=lambda q, n: [])  # type: ignore[arg-type]
    return graph.get_graph().draw_mermaid()
