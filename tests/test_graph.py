"""End-to-end tests of the multi-agent graph with a fake LLM."""

from __future__ import annotations

from conftest import FakeLLM, fake_search

from research_assistant.config import Settings
from research_assistant.graph import RECURSION_LIMIT, build_graph
from research_assistant.schemas import (
    Critique,
    KeyPoint,
    ResearchNotes,
    ResearchPlan,
    SupervisorDecision,
)

PLAN = ResearchPlan(questions=["Q1", "Q2", "q1 "], rationale="Covers the basics.")
NOTES = ResearchNotes(
    key_points=[
        KeyPoint(point="Fact A", source_ids=[1]),
        KeyPoint(point="Unsupported", source_ids=[99]),  # invalid citation: must be dropped
    ]
)
APPROVED = Critique(score=9, approved=True, strengths=["Clear"])
REJECTED = Critique(score=5, approved=False, issues=["Too vague"])
REJECTED_NEEDS_DATA = Critique(
    score=5, approved=False, issues=["Missing costs"], follow_up_questions=["Q3"]
)


def run(llm: FakeLLM, settings: Settings | None = None, search=fake_search) -> dict:
    graph = build_graph(llm=llm, settings=settings or Settings(), search=search)
    return graph.invoke({"topic": "Test topic"}, config={"recursion_limit": RECURSION_LIMIT})


def test_happy_path_produces_cited_report():
    llm = FakeLLM(
        structured={ResearchPlan: [PLAN], ResearchNotes: [NOTES], Critique: [APPROVED]},
        texts=["# Title\n\nFirst claim [1]. Second claim [3]. Bogus [42]."],
    )

    state = run(llm)

    # Planner output was de-duplicated, and each question researched exactly once.
    assert state["questions"] == ["Q1", "Q2"]
    assert sorted(f.question for f in state["findings"]) == ["Q1", "Q2"]
    assert llm.calls["ResearchNotes"] == 2
    # Invalid citations from the researcher were filtered out.
    assert all(len(f.key_points) == 1 for f in state["findings"])

    # One draft, one review, approved.
    assert state["revision"] == 1
    assert state["critique"].approved

    # Sources list only the cited, real sources, never the out-of-range [42].
    report = state["report"]
    sources_section = report.split("## Sources", 1)[1]
    assert sources_section.count("- [") == 2
    assert "[1]" in sources_section and "[3]" in sources_section
    assert "[42]" not in sources_section
    assert "https://example.com/" in sources_section


def test_stops_at_revision_limit_when_critic_never_approves():
    llm = FakeLLM(structured={ResearchPlan: [PLAN], ResearchNotes: [NOTES], Critique: [REJECTED]})

    state = run(llm, Settings(max_revisions=3))

    assert state["revision"] == 3
    assert llm.calls["text"] == 3  # writer
    assert llm.calls["Critique"] == 3
    assert llm.calls["SupervisorDecision"] == 0  # no follow-ups, so no choice to make
    assert state["report"]


def test_writer_revision_receives_critic_feedback():
    llm = FakeLLM(
        structured={
            ResearchPlan: [PLAN],
            ResearchNotes: [NOTES],
            Critique: [REJECTED, APPROVED],
        },
        texts=["# Draft one [1]", "# Draft two [1]"],
    )

    state = run(llm)

    assert state["revision"] == 2
    revision_prompt = llm.prompts["text"][1][-1].content
    assert "# Draft one [1]" in revision_prompt
    assert "Too vague" in revision_prompt
    assert state["report"].startswith("# Draft two")


def test_supervisor_can_send_team_back_to_research():
    llm = FakeLLM(
        structured={
            ResearchPlan: [PLAN],
            ResearchNotes: [NOTES],
            Critique: [REJECTED_NEEDS_DATA, APPROVED],
            SupervisorDecision: [SupervisorDecision(next="researcher", reason="Need cost data.")],
        }
    )

    state = run(llm)

    assert state["questions"] == ["Q1", "Q2", "Q3"]
    assert sorted(f.question for f in state["findings"]) == ["Q1", "Q2", "Q3"]
    assert state["research_rounds"] == 2
    assert state["revision"] == 2
    assert any(entry.startswith("researcher: Need cost data") for entry in state["log"])


def test_research_rounds_are_capped():
    llm = FakeLLM(
        structured={
            ResearchPlan: [PLAN],
            ResearchNotes: [NOTES],
            Critique: [REJECTED_NEEDS_DATA, APPROVED],
            SupervisorDecision: [SupervisorDecision(next="researcher", reason="More data.")],
        }
    )

    state = run(llm, Settings(max_research_rounds=1))

    # Only one research round allowed, so the supervisor never gets to choose research.
    assert llm.calls["SupervisorDecision"] == 0
    assert state["research_rounds"] == 1
    assert "Q3" not in state["questions"]


def test_supervisor_failure_falls_back_to_writer():
    llm = FakeLLM(
        structured={
            ResearchPlan: [PLAN],
            ResearchNotes: [NOTES],
            Critique: [REJECTED_NEEDS_DATA, APPROVED],
            SupervisorDecision: [RuntimeError("model unavailable")],
        }
    )

    state = run(llm)

    assert state["research_rounds"] == 1
    assert state["revision"] == 2
    assert state["critique"].approved


def test_empty_search_results_do_not_loop_forever():
    llm = FakeLLM(
        structured={ResearchPlan: [PLAN], ResearchNotes: [NOTES], Critique: [APPROVED]},
        texts=["# Report\n\nNo evidence was available."],
    )

    state = run(llm, search=lambda query, n: [])

    assert len(state["findings"]) == 2
    assert all(not f.sources for f in state["findings"])
    assert llm.calls["ResearchNotes"] == 0  # nothing to summarise
    assert "## Sources" not in state["report"]


def test_critic_cannot_approve_below_threshold():
    sloppy_approval = Critique(score=6, approved=True)
    llm = FakeLLM(
        structured={
            ResearchPlan: [PLAN],
            ResearchNotes: [NOTES],
            Critique: [sloppy_approval, APPROVED],
        }
    )

    state = run(llm, Settings(approval_score=8))

    assert state["revision"] == 2  # the 6/10 "approval" was overridden
