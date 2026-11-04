"""Turn the researchers' findings into a single, globally numbered evidence pack.

Each researcher numbers its own search results 1..N. Before writing, all
findings are merged, duplicate URLs are collapsed, and every key point's
citations are remapped to one global numbering. The final report's Sources
section is generated from this index, so cited URLs always come from real
search results and are never invented by the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from research_assistant.schemas import Finding, Source

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class EvidencePack:
    sources: list[Source]  # global list; source number n is sources[n - 1]
    markdown: str  # evidence formatted for the writer and critic

    @property
    def is_empty(self) -> bool:
        return not self.markdown.strip()


def build_evidence(findings: list[Finding]) -> EvidencePack:
    sources: list[Source] = []
    number_by_url: dict[str, int] = {}
    sections: list[str] = []

    for finding in findings:
        local_to_global: dict[int, int] = {}
        for local_id, source in enumerate(finding.sources, start=1):
            if source.url not in number_by_url:
                sources.append(source)
                number_by_url[source.url] = len(sources)
            local_to_global[local_id] = number_by_url[source.url]

        lines: list[str] = []
        for kp in finding.key_points:
            refs = sorted({local_to_global[i] for i in kp.source_ids if i in local_to_global})
            if not refs:
                continue
            lines.append(f"- {kp.point} " + "".join(f"[{n}]" for n in refs))

        if lines:
            sections.append(f"### {finding.question}\n" + "\n".join(lines))

    source_lines = [f"[{n}] {s.title} ({s.url})" for n, s in enumerate(sources, start=1)]
    markdown = "\n\n".join(sections)
    if markdown:
        markdown += "\n\n### Source list\n" + "\n".join(source_lines)
    return EvidencePack(sources=sources, markdown=markdown)


def cited_numbers(text: str) -> list[int]:
    """Citation numbers used in ``text``, in order of first appearance."""
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def render_sources(text: str, sources: list[Source]) -> str:
    """Markdown 'Sources' section listing only the sources cited in ``text``."""
    valid = sorted(n for n in cited_numbers(text) if 1 <= n <= len(sources))
    if not valid:
        return ""
    # Each line keeps the same [n] number used in the report body.
    lines = [f"- [{n}] [{sources[n - 1].title}]({sources[n - 1].url})" for n in valid]
    return "## Sources\n\n" + "\n".join(lines)
