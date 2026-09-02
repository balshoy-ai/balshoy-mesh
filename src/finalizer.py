"""Finalizer: merge the per-step results into one coherent answer (N -> 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from dag import topological_order
from llm import LLM, build_llm

if TYPE_CHECKING:
    from planner import Plan

MERGE_PROMPT = (
    "You are the finalizer of a multi-agent system. Several specialized agents produced "
    "independent parts in response to a single user request. Merge these parts into ONE "
    "coherent final answer:\n"
    "- preserve all substantive information from every part; do not drop facts;\n"
    "- do not duplicate content; if parts overlap, keep one authoritative statement;\n"
    "- keep a natural order (causes before effects, context before conclusion);\n"
    "- do not invent facts absent from the parts;\n"
    "- reply only with the merged answer, in the same language as the user request."
)


class Finalizer(Protocol):
    """Assembles the per-step results into a single final answer."""

    def finalize(self, plan: Plan, results: dict[str, str]) -> str:
        """Return the final merged answer for the given plan and results."""


class FakeFinalizer:
    """Deterministic fallback: concatenate the parts in dependency order."""

    def finalize(self, plan: Plan, results: dict[str, str]) -> str:
        """Return the ordered parts joined in topological order, loss-free."""
        order = [step.id for step in topological_order(plan)]
        if len(order) == 1:
            return results[order[0]]
        return "\n\n".join(results[step_id] for step_id in order)


class LLMFinalizer:
    """Merge the parts via an LLM to produce a coherent, de-duplicated answer."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def finalize(self, plan: Plan, results: dict[str, str]) -> str:
        """Merge all step results into one answer through the LLM."""
        order = topological_order(plan)
        if len(order) == 1:
            return results[order[0].id]

        parts = [
            f"## Step {step.id} ({step.agent})\n"
            f"Instruction: {step.instruction}\n"
            f"Result:\n{results[step.id]}"
            for step in order
        ]
        return self.llm.complete(MERGE_PROMPT, "\n\n".join(parts))


def build_finalizer() -> Finalizer:
    """Build the finalizer selected by the environment (real or fake)."""
    llm = build_llm()
    if llm.name == "fake":
        return FakeFinalizer()
    return LLMFinalizer(llm)
