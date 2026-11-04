"""Command-line interface: ``research-assistant "your topic"``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from research_assistant.config import Settings
from research_assistant.graph import RECURSION_LIMIT, build_graph, mermaid_diagram
from research_assistant.llm import MissingAPIKeyError


def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "report"


def print_update(node: str, data: dict) -> None:
    """Print a one-line progress message for each agent step."""
    if node == "supervisor":
        print(f"[supervisor] -> {data['log'][0]}")
    elif node == "planner":
        print(f"[planner]    {len(data['questions'])} research questions:")
        for i, question in enumerate(data["questions"], start=1):
            print(f"               {i}. {question}")
    elif node == "researcher":
        for finding in data["findings"]:
            sources, facts = len(finding.sources), len(finding.key_points)
            print(f"[researcher] {finding.question}")
            print(f"               {sources} sources, {facts} key facts")
    elif node == "writer":
        words = len(data["draft"].split())
        print(f"[writer]     draft v{data['revision']} written ({words} words)")
    elif node == "critic":
        critique = data["critique"]
        verdict = "approved" if critique.approved else "needs revision"
        print(f"[critic]     score {critique.score}/10, {verdict}")
        for issue in critique.issues[:3]:
            print(f"               - {issue}")
    elif node == "finalize":
        print("[finalize]   report assembled with sources")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-assistant",
        description="Multi-agent research assistant: turns a topic into a cited Markdown report.",
    )
    parser.add_argument("topic", nargs="?", help="What to research (prompted for if omitted).")
    parser.add_argument("-o", "--output-dir", default="outputs", help="Where to save the report.")
    parser.add_argument("--model", help="Gemini model name (overrides GEMINI_MODEL).")
    parser.add_argument("--max-questions", type=int, help="Max research questions to plan.")
    parser.add_argument("--max-revisions", type=int, help="Max drafts the writer may produce.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print the output path.")
    parser.add_argument(
        "--show-graph", action="store_true", help="Print the agent graph as Mermaid and exit."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # avoid crashes on non-UTF-8 consoles
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_graph:
        print(mermaid_diagram())
        return 0

    topic = (args.topic or input("Research topic: ")).strip()
    if not topic:
        parser.error("a research topic is required")

    overrides = {
        "model": args.model,
        "max_questions": args.max_questions,
        "max_revisions": args.max_revisions,
    }
    settings = dataclasses.replace(
        Settings.from_env(), **{k: v for k, v in overrides.items() if v is not None}
    )

    try:
        graph = build_graph(settings=settings)
    except MissingAPIKeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Researching: {topic}\nModel: {settings.model}\n")
    report = ""
    try:
        for update in graph.stream(
            {"topic": topic},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="updates",
        ):
            for node, data in update.items():
                if not data:
                    continue
                if node == "finalize":
                    report = data["report"]
                if not args.quiet:
                    print_update(node, data)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(topic)}-{datetime.now():%Y%m%d-%H%M%S}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
