"""DAG validation and topological ordering shared by the executor and finalizer."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planner import Plan, Step


class DAGError(Exception):
    """Raised when a plan is cyclic or references unknown steps."""


def topological_order(plan: Plan) -> list[Step]:  # noqa: C901
    """Validate the plan and return its steps in a dependency-respecting order.

    Raises:
        DAGError: if a step depends on an unknown id or the graph has a cycle.
    """
    ids = {s.id for s in plan.steps}

    for s in plan.steps:
        for dep in s.depends_on:
            if dep not in ids:
                msg = f"step {s.id} depends on unknown {dep}"
                raise DAGError(msg)

    indegree = {s.id: len(s.depends_on) for s in plan.steps}
    by_id = {s.id: s for s in plan.steps}
    ready = deque(s.id for s in plan.steps if indegree[s.id] == 0)
    order: list[str] = []

    while ready:
        node = ready.popleft()
        order.append(node)
        for s in plan.steps:
            if node in s.depends_on:
                indegree[s.id] -= 1
                if indegree[s.id] == 0:
                    ready.append(s.id)

    if len(order) != len(plan.steps):
        msg = "cycle detected in plan"
        raise DAGError(msg)

    return [by_id[i] for i in order]
