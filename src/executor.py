from planner import Plan, Step
from agents import run


class DAGError(Exception):
    pass


def _topo(plan: Plan) -> List[Step]:
    ids = {s.id for s in plan.steps}
    for s in plan.steps:
        for d in s.depends_on:
            if d not in ids:
                raise DAGError(f"step {s.id} depends on unknown {d}")

    indeg = {s.id: len(s.depends_on) for s in plan.steps}
    by_id = {s.id: s for s in plan.steps}
    ready = [s.id for s in plan.steps if indeg[s.id] == 0]
    order: List[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for s in plan.steps:
            if n in s.depends_on:
                indeg[s.id] -= 1
                if indeg[s.id] == 0:
                    ready.append(s.id)

    if len(order) != len(plan.steps):
        raise DAGError("cycle detected in plan")
    return [by_id[i] for i in order]


def execute(plan: Plan) -> dict:
    """Validate + topologically execute the plan, threading dependency results."""
    order = _topo(plan)
    results: dict = {}
    for s in order:
        ctx = " | ".join(results[d] for d in s.depends_on)
        results[s.id] = run(s.agent, s.instruction, ctx)
    return results
