"""LLM construction and small helpers for working with chat model responses."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from research_assistant.config import Settings


class MissingAPIKeyError(RuntimeError):
    """Raised when GOOGLE_API_KEY is not configured."""


def create_llm(settings: Settings) -> BaseChatModel:
    """Create the Gemini chat model used by every agent."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://aistudio.google.com/apikey)."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.model,
        temperature=settings.temperature,
        max_retries=3,
    )


def message_text(message: BaseMessage) -> str:
    """Return the plain text of a chat message.

    Some providers return ``content`` as a list of typed parts rather than a
    string; this flattens either shape to text.
    """
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts).strip()
