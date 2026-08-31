import os
import json
from typing import List, Protocol
from pydantic import BaseModel


class Step(BaseModel):
    id: str
    agent: str
    depends_on: List[str] = []
    instruction: str


class Plan(BaseModel):
    steps: List[Step]


SYSTEM = (
    "You are a task planner for a multi-agent system. "
    "Decompose the user request into ordered steps. Each step names an "
    "agent type (e.g. code_review, doc_gen, translate) and what it should do. "
    "Use depends_on to express DATA dependencies between steps. "
    "Respond ONLY with JSON of shape {\"steps\": [{\"id\", \"agent\", "
    "\"depends_on\", \"instruction\"}]}."
)


class Planner(Protocol):
    def plan(self, prompt: str) -> Plan: ...


class FakePlanner:
    """Canned plan so the full loop runs with no API key. Proves the DAG engine."""

    def plan(self, prompt: str) -> Plan:
        return Plan(steps=[
            Step(id="s1", agent="code_review", instruction="найди баги в коде"),
            Step(id="s2", agent="doc_gen", depends_on=["s1"],
                 instruction="напиши документацию по результатам ревью"),
        ])


class OpenAIPlanner:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def plan(self, prompt: str) -> Plan:
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        raw = json.loads(resp.choices[0].message.content)
        return Plan.model_validate(raw)
