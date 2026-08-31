import json
import os
import sys

from planner import PlanError, _topo, OpenAIPlanner

PROMPTS = [
    "Найди баги в коде и напиши документацию",
    "Переведи этот заголовок на английский",
    "Проверь пул-реквест на ошибки и опиши их в чат",
    "Напиши readme для проекта",
    "Переведи и вычитай текст договора",
    "Посмотри, есть ли утечки в этом модуле",
    "Сгенерируй документацию и переведи её на английский",
    "Просто напиши приветствие",
    "Отрефактори функцию и добавь юнит-тесты",
    "Составь отзыв на дизайн-документ",
]

KNOWN = {"code_review", "doc_gen", "translate"}


def evaluate() -> None:
    planner = OpenAIPlanner()
    report = {"planner_model": planner.model, "items": []}
    n_ok = 0
    for i, p in enumerate(PROMPTS, 1):
        item = {"request": p}
        try:
            plan = planner.plan_safe(p)
            order = _topo(plan.steps)
            item["plan"] = plan.model_dump()
            item["topo_ok"] = True
            item["unknown_agents"] = sorted({s.agent for s in plan.steps} - KNOWN)
            item["redundant_duplicate_ids"] = (
                len({s.id for s in plan.steps}) != len(plan.steps)
            )
            item["task_count"] = len(plan.steps)
            n_ok += 1
        except PlanError as e:
            item["error"] = str(e)
        report["items"].append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    report["valid_plans"] = n_ok
    report["total"] = len(PROMPTS)
    report["passed"] = n_ok == len(PROMPTS)
    out = os.getenv("REPORT_PATH", "eval_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRESULT: {n_ok}/{len(PROMPTS)} valid plans -> {out}")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    evaluate()
