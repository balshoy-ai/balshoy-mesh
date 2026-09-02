"""LLM backend abstraction shared by the planner, agents and finalizer."""

from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI


class LLM(Protocol):
    """A single chat-completion backend. Contract: system + user -> assistant text."""

    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the assistant reply for the given system and user prompts."""


class OpenAILLM:
    """Completes via an OpenAI-compatible endpoint (used with Ollama by default)."""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", "ollama"),
        )
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:3b")

    def complete(self, system: str, user: str) -> str:
        """Send the prompt to the model and return the assistant message text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


class FakeLLM:
    """Deterministic stand-in so the in-memory loop runs with no API key."""

    name = "fake"

    def complete(self, _system: str, user: str) -> str:
        """Return the user prompt verbatim as a stubbed completion."""
        return user


def is_llm_configured() -> bool:
    """Return True when a real LLM provider is explicitly requested via env vars."""
    used_env = any(os.getenv(key) for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL"))
    return used_env and os.getenv("LLM_PROVIDER", "openai") != "fake"


def build_llm() -> LLM:
    """Build the LLM backend selected by the environment (real or fake)."""
    if is_llm_configured():
        return OpenAILLM()
    return FakeLLM()
