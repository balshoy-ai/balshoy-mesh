from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from agents import run as run_agent
from dag import topological_order

if TYPE_CHECKING:
    from planner import Plan

AgentRun = Callable[[str, str, str], str]


def execute(plan: Plan, agent_run: AgentRun = run_agent) -> dict[str, str]:
    """Validate + topologically execute the plan, threading dependency results.

    Returns a mapping of step id -> result text. The Finalizer (see finalizer.py)
    is responsible for merging these parts into the final answer.
    """
    order = topological_order(plan)
    results: dict[str, str] = {}
    for step in order:
        context = " | ".join(results[d] for d in step.depends_on)
        results[step.id] = agent_run(step.agent, step.instruction, context)
    return results
