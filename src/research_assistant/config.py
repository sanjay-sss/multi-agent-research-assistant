"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Knobs that control the research workflow."""

    model: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_questions: int = 4
    results_per_question: int = 5
    max_revisions: int = 2
    max_research_rounds: int = 2
    approval_score: int = 8

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.getenv("GEMINI_MODEL", cls.model),
            max_questions=_env_int("MAX_RESEARCH_QUESTIONS", cls.max_questions),
            results_per_question=_env_int("SEARCH_RESULTS_PER_QUESTION", cls.results_per_question),
            max_revisions=_env_int("MAX_REVISIONS", cls.max_revisions),
            max_research_rounds=_env_int("MAX_RESEARCH_ROUNDS", cls.max_research_rounds),
            approval_score=_env_int("APPROVAL_SCORE", cls.approval_score),
        )
