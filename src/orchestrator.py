from __future__ import annotations

from executor import execute
from finalizer import build_finalizer
from llm import is_llm_configured
from planner import FakePlanner, OpenAIPlanner, Planner
from responses import ChatCompletionResponse, build_response


def build_planner() -> Planner:
    """Select a fake (offline) or OpenAI planner from the environment."""
    if is_llm_configured():
        return OpenAIPlanner()
    return FakePlanner()


def run(prompt: str) -> ChatCompletionResponse:
    """Planner -> Plan(DAG) -> Executor -> Finalizer. Returns an OpenAI response."""
    planner = build_planner()
    plan = planner.plan_safe(prompt) if hasattr(planner, "plan_safe") else planner.plan(prompt)
    print("PLAN:", plan.model_dump_json(indent=2))  # noqa: T201

    results = execute(plan)
    finalizer = build_finalizer()
    answer = finalizer.finalize(plan, results)
    return build_response(answer)
