import os
from planner import FakePlanner, OpenAIPlanner
from executor import execute


def build_planner():
    if os.getenv("LLM_PROVIDER") == "openai":
        return OpenAIPlanner()
    return FakePlanner()


def run(prompt: str) -> dict:
    planner = build_planner()
    plan = planner.plan(prompt)
    print("PLAN:", plan.model_dump_json(indent=2))
    return execute(plan)
