"""Phase 2 tests: real DAG execution, agents, finalizer assembly and the OpenAI response."""

from __future__ import annotations

import pytest

from agents import LLMAgent
from dag import DAGError, topological_order
from executor import execute
from finalizer import FakeFinalizer, LLMFinalizer
from orchestrator import run
from planner import Plan, Step
from responses import build_response


class StubLLM:
    """Recording LLM so LLM-backed code can be tested deterministically."""

    name = "stub"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        """Record the call and return a deterministic stub response."""
        self.calls.append((system, user))
        return f"stub:{len(self.calls)}"


def _plan_two_steps() -> Plan:
    return Plan(
        steps=[
            Step(id="s1", agent="code_review", instruction="find bugs"),
            Step(id="s2", agent="doc_gen", depends_on=["s1"], instruction="write docs"),
        ]
    )


def test_execute_threads_dependency_context() -> None:
    """Dependency results are threaded into dependent steps as context."""
    plan = _plan_two_steps()
    results = execute(plan)
    assert results["s1"] == "[code_review] done: find bugs (ctx: none)"
    assert results["s2"] == (
        "[doc_gen] done: write docs (ctx: [code_review] done: find bugs (ctx: none))"
    )


@pytest.mark.parametrize(
    ("steps", "raises"),
    [
        pytest.param(
            [Step(id="a", agent="code_review", instruction="i", depends_on=["ghost"])],
            True,
            id="unknown-dep",
        ),
        pytest.param(
            [
                Step(id="a", agent="code_review", instruction="ia", depends_on=["b"]),
                Step(id="b", agent="doc_gen", instruction="ib", depends_on=["a"]),
            ],
            True,
            id="cycle",
        ),
        pytest.param(
            [
                Step(id="a", agent="code_review", instruction="ia"),
                Step(id="b", agent="doc_gen", instruction="ib", depends_on=["a"]),
            ],
            False,
            id="linear",
        ),
    ],
)
def test_topological_order(steps: list[Step], raises: bool) -> None:
    """Cycles and dangling deps are rejected; valid DAGs are ordered."""
    plan = Plan(steps=steps)
    if raises:
        with pytest.raises(DAGError):
            topological_order(plan)
    else:
        assert [s.id for s in topological_order(plan)] == ["a", "b"]


def test_fake_finalizer_merges_in_topological_order() -> None:
    """Fake finalizer concatenates parts in dependency order without loss."""
    out = FakeFinalizer().finalize(_plan_two_steps(), {"s1": "part1", "s2": "part2"})
    assert out == "part1\n\npart2"


def test_fake_finalizer_single_step_passthrough() -> None:
    """A single-step plan is returned verbatim by the fake finalizer."""
    plan = Plan(steps=[Step(id="only", agent="translate", instruction="hi")])
    assert FakeFinalizer().finalize(plan, {"only": "перевод"}) == "перевод"


def test_llm_finalizer_merges_parts_via_llm() -> None:
    """The LLM finalizer sends every part to the model for merging."""
    llm = StubLLM()
    out = LLMFinalizer(llm).finalize(_plan_two_steps(), {"s1": "r1", "s2": "r2"})
    assert out == "stub:1"
    assert len(llm.calls) == 1
    _system, user = llm.calls[0]
    assert "s1" in user
    assert "s2" in user


def test_llm_finalizer_single_step_skips_llm() -> None:
    """A single-step plan short-circuits the LLM merge call."""
    llm = StubLLM()
    plan = Plan(steps=[Step(id="only", agent="translate", instruction="hi")])
    assert LLMFinalizer(llm).finalize(plan, {"only": "перевод"}) == "перевод"
    assert llm.calls == []


def test_llm_agent_uses_role_prompt_and_context() -> None:
    """A real agent sends its role prompt and the dependency context to the LLM."""
    llm = StubLLM()
    agent = LLMAgent("code_review", llm)
    out = agent.run("find bugs", "file.py has an off-by-one")
    assert out == "stub:1"
    system, user = llm.calls[0]
    assert "code reviewer" in system
    assert "find bugs" in user
    assert "off-by-one" in user


def test_llm_agent_rejects_unknown_type() -> None:
    """Unknown agent types are rejected during construction."""
    with pytest.raises(KeyError):
        LLMAgent("nope", StubLLM())


def test_orchestrator_run_returns_openai_response() -> None:
    """The orchestrator pipeline returns a properly assembled OpenAI response."""
    response = run("Найди баги в коде и напиши документацию")
    assert response.object == "chat.completion"
    assert response.choices[0].message.role == "assistant"
    content = response.choices[0].message.content
    assert "code_review" in content
    assert "doc_gen" in content


def test_build_response_contract() -> None:
    """The OpenAI response contract is filled correctly."""
    response = build_response("hello")
    assert response.model == "agentic-mesh"
    assert response.choices[0].message.content == "hello"
    assert response.object == "chat.completion"
