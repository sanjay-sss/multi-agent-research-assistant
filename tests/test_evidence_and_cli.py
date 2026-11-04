"""Tests for evidence numbering, source rendering and the CLI."""

from __future__ import annotations

from research_assistant.cli import main, slugify
from research_assistant.evidence import build_evidence, cited_numbers, render_sources
from research_assistant.schemas import Finding, KeyPoint, Source


def _src(n: int) -> Source:
    return Source(title=f"Title {n}", url=f"https://example.com/{n}")


def test_evidence_uses_global_numbering_and_dedupes_urls():
    findings = [
        Finding(
            question="Q1",
            sources=[_src(1), _src(2)],
            key_points=[KeyPoint(point="p1", source_ids=[2])],
        ),
        Finding(
            question="Q2",
            sources=[_src(2), _src(3)],  # _src(2) is shared with Q1
            key_points=[KeyPoint(point="p2", source_ids=[1, 2])],
        ),
    ]

    pack = build_evidence(findings)

    assert [s.url for s in pack.sources] == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]
    assert "- p1 [2]" in pack.markdown
    assert "- p2 [2][3]" in pack.markdown  # local ids 1,2 in Q2 -> global 2,3


def test_key_points_without_valid_sources_are_omitted():
    pack = build_evidence(
        [Finding(question="Q", sources=[_src(1)], key_points=[KeyPoint(point="x", source_ids=[5])])]
    )
    assert pack.is_empty


def test_render_sources_lists_only_valid_cited_sources():
    text = "A [2]. B [1][2]. C [9]."
    assert cited_numbers(text) == [2, 1, 9]

    section = render_sources(text, [_src(1), _src(2), _src(3)])

    assert section.startswith("## Sources")
    assert "- [1] [Title 1](https://example.com/1)" in section
    assert "- [2] [Title 2](https://example.com/2)" in section
    assert "Title 3" not in section and "[9]" not in section


def test_render_sources_empty_when_nothing_cited():
    assert render_sources("No citations here.", [_src(1)]) == ""


def test_slugify():
    assert slugify("Impact of AI on Healthcare?!") == "impact-of-ai-on-healthcare"
    assert slugify("!!!") == "report"


def test_cli_show_graph(capsys):
    assert main(["--show-graph"]) == 0
    out = capsys.readouterr().out
    for node in ("supervisor", "planner", "researcher", "writer", "critic", "finalize"):
        assert node in out


def test_cli_reports_missing_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # make sure no local .env is picked up
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert main(["some topic"]) == 1
    assert "GOOGLE_API_KEY" in capsys.readouterr().err
