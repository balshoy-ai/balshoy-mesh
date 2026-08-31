import os
from planner import FakePlanner, OpenAIPlanner
from executor import execute


def build_planner():
    used_env = any(os.getenv(k) for k in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL"))
    if used_env and os.getenv("LLM_PROVIDER", "openai") != "fake":
        return OpenAIPlanner()
    return FakePlanner()


def run(prompt: str) -> dict:
    planner = build_planner()
    plan = planner.plan_safe(prompt) if hasattr(planner, "plan_safe") else planner.plan(prompt)
    print("PLAN:", plan.model_dump_json(indent=2))
    return execute(plan)
