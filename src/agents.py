"""Specialized agents. Contract: run(agent_type, instruction, context) -> result."""

from __future__ import annotations

from typing import Protocol

from llm import LLM, FakeLLM, build_llm

ROLE_PROMPTS: dict[str, str] = {
    "code_review": (
        "You are a senior code reviewer. Inspect the provided code or request, find bugs, "
        "security issues and design flaws, and report them precisely. Use the dependency "
        "context if present. Reply in the same language as the instruction. Be concrete."
    ),
    "doc_gen": (
        "You are a technical writer. From the given material write clear, complete "
        "documentation. Preserve every technical fact from the dependency context. Reply "
        "in the same language as the instruction."
    ),
    "translate": (
        "You are a professional translator. Translate the given text faithfully, keeping "
        "meaning, tone and terminology. Use the dependency context for terms already "
        "introduced. Reply only with the translation."
    ),
}


class Agent(Protocol):
    """A model behind a task type. Contract: (instruction, context) -> result."""

    def run(self, instruction: str, context: str) -> str:
        """Run one task and return its result text."""


class FakeAgent:
    """Deterministic stand-in: proves the DAG engine with no API key."""

    def __init__(self, agent_type: str) -> None:
        self.agent_type = agent_type

    def run(self, instruction: str, context: str) -> str:
        """Return a canned result echoing the instruction and dependency context."""
        return f"[{self.agent_type}] done: {instruction} (ctx: {context or 'none'})"


class LLMAgent:
    """A real agent: role prompt + LLM completion."""

    def __init__(self, agent_type: str, llm: LLM) -> None:
        if agent_type not in ROLE_PROMPTS:
            msg = f"no agent registered for '{agent_type}'"
            raise KeyError(msg)
        self.agent_type = agent_type
        self.llm = llm

    def run(self, instruction: str, context: str) -> str:
        """Run the agent's role prompt against the LLM with the step context."""
        role = ROLE_PROMPTS[self.agent_type]
        user = instruction
        if context:
            user += "\n\nDependency context:\n" + context
        return self.llm.complete(role, user)


def _build_registry() -> dict[str, Agent]:
    llm = build_llm()
    if isinstance(llm, FakeLLM):
        return {agent_type: FakeAgent(agent_type) for agent_type in ROLE_PROMPTS}
    return {agent_type: LLMAgent(agent_type, llm) for agent_type in ROLE_PROMPTS}


_REGISTRY: dict[str, Agent] = {}


def get_registry() -> dict[str, Agent]:
    """Return the agent registry, selecting fake or real agents from the environment."""
    if not _REGISTRY:
        _REGISTRY.update(_build_registry())
    return _REGISTRY


def run(agent: str, instruction: str, context: str) -> str:
    """Dispatch one task to its agent. Keep this signature stable across phases."""
    fn = get_registry().get(agent)
    if fn is None:
        msg = f"no agent registered for '{agent}'"
        raise KeyError(msg)
    return fn.run(instruction, context)
