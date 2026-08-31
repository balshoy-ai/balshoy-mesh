from typing import Callable, Dict

# Mock agents: stand-ins for real GPU models. Signature is the contract
# every future real agent must satisfy: (instruction, dependency_context) -> result.
MockAgent = Callable[[str, str], str]

REGISTRY: Dict[str, MockAgent] = {
    "code_review": lambda instr, ctx: f"[code_review] done: {instr} (ctx: {ctx or 'none'})",
    "doc_gen": lambda instr, ctx: f"[doc_gen] done: {instr} (ctx: {ctx or 'none'})",
    "translate": lambda instr, ctx: f"[translate] done: {instr}",
}


def run(agent: str, instruction: str, context: str) -> str:
    fn = REGISTRY.get(agent)
    if not fn:
        raise KeyError(f"no agent registered for '{agent}'")
    return fn(instruction, context)
