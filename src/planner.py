import json
import os
from collections import deque
from typing import List, Protocol
from pydantic import BaseModel, ValidationError


class Step(BaseModel):
    id: str
    agent: str
    depends_on: List[str] = []
    instruction: str


class Plan(BaseModel):
    steps: List[Step]


class PlanError(Exception):
    """Raised when the planner cannot produce a structurally valid Plan."""


SYSTEM = (
    "You are a task planner for a multi-agent system. "
    "Decompose the user request into a DAG of steps. Each step must name one "
    "allowed agent type: code_review, doc_gen, translate. "
    "Use depends_on to express only DATA dependencies between steps: a step "
    "may depend on another step only if it actually needs that step's output. "
    "Avoid redundant or duplicate steps and avoid empty plans of a single "
    "trivial step when the request is simple. "
    "Respond ONLY with a valid JSON object of shape "
    '{"steps": [{"id": "s1", "agent": "code_review", '
    '"depends_on": ["s2"], "instruction": "..."}]}. No prose, no code fences.'
)


def _deps_of(s: Step) -> List[str]:
    ids = {x.id for x in s.depends_on}
    return list(ids)


def _topo(steps: List[Step]) -> List[str]:
    ids = {s.id for s in steps}
    indeg = {s.id: 0 for s in steps}
    consumers = {sid: [] for sid in ids}
    for s in steps:
        deps = [d for d in set(s.depends_on) if d in ids and d != s.id]
        indeg[s.id] = len(deps)
        for d in deps:
            consumers[d].append(s.id)
    ready = deque(sid for sid in ids if indeg[sid] == 0)
    order = []
    while ready:
        n = ready.popleft()
        order.append(n)
        for c in consumers[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != len(ids):
        raise PlanError("cycle detected: " + ", ".join(sorted(set(ids) - set(order))))
    return order


def _cycle_nodes(steps: List[Step]) -> set:
    ids = {s.id for s in steps}
    indeg = {s.id: 0 for s in steps}
    consumers = {sid: [] for sid in ids}
    for s in steps:
        deps = [d for d in set(s.depends_on) if d in ids and d != s.id]
        indeg[s.id] = len(deps)
        for d in deps:
            consumers[d].append(s.id)
    ready = deque(sid for sid in ids if indeg[sid] == 0)
    order = []
    while ready:
        n = ready.popleft()
        order.append(n)
        for c in consumers[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    return set(ids) - set(order)


def _repair(plan: Plan) -> Plan:
    seen = set()
    steps = []
    for s in plan.steps:
        if s.id in seen:
            continue
        seen.add(s.id)
        steps.append(s.model_copy(deep=True))

    ids = {s.id for s in steps}

    for s in steps:
        keep = []
        for d in s.depends_on:
            if d in ids and d != s.id and d not in keep:
                keep.append(d)
        s.depends_on = keep

    while True:
        cyc = _cycle_nodes(steps)
        if not cyc:
            break
        for s in steps:
            s.depends_on = [d for d in s.depends_on if d not in cyc]

    return Plan(steps=steps)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    return json.loads(text)


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
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 retries: int = 3):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "ollama"),
        )
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:3b")
        self.retries = retries

    def plan(self, prompt: str, _feedback: str | None = None) -> Plan:
        user = prompt
        if _feedback:
            user += ("\n\nYour previous output was invalid:\n"
                     f"{_feedback}\nFix it and reply with valid JSON only.")
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        return self._parse(resp.choices[0].message.content)

    def _parse(self, content: str) -> Plan:
        try:
            raw = _extract_json(content)
            plan = Plan.model_validate(raw)
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            raise PlanError(f"invalid output: {e}") from e
        return _repair(plan)

    def plan_safe(self, prompt: str) -> Plan:
        last_err = None
        for _ in range(self.retries):
            try:
                return self.plan(prompt, _feedback=last_err)
            except Exception as e:
                last_err = str(e)
        raise PlanError(f"planner failed after {self.retries} attempts: {last_err}")
