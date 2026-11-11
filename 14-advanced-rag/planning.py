"""Small, validated dependency-aware retrieval plans; gold labels not available."""
from __future__ import annotations

import re

from contracts import Analysis, Budgets, Request, SearchPlan, SearchStep
from routing import SourceRouter

STRATEGIES = frozenset({
    "baseline", "rewrite", "multiquery", "decompose", "planned", "adaptive", "hyde"
})


def _year(question: str) -> str:
    found = re.findall(r"\b20\d{2}\b", question)
    return found[-1] if found else "2025"


def aspect_query(analysis: Analysis, aspect: str, question: str, *, recovery: bool = False) -> str:
    year = _year(question)
    if analysis.intent == "finance_compare":
        queries = {
            "q1_revenue": f"FPT international software Q1/{year} quarterly revenue billion VND",
            "q2_revenue": f"FPT international software Q2/{year} quarterly revenue billion VND",
            "drivers": (
                f"FPT international software Q1/{year} Q2/{year} management "
                + ("contract renewals delivery volume Japan business drivers"
                   if recovery else "explanation why revenue changed")
            ),
        }
    elif analysis.intent == "refund_policy":
        queries = {
            "deadline": "retail refund deadline time to request after delivery",
            "conditions": "retail refund eligibility unused product packaging proof of purchase",
            "exceptions": "refund policy exclusions activated digital subscriptions",
        }
    elif analysis.intent == "technical_serving":
        queries = {"mitigation": "vLLM Qwen3 CUDA out of memory max_model_len KV cache mitigation"}
    elif analysis.intent == "authorization":
        queries = {"diagnosis": "ERR_AUTH_403 administrator role authorization handler"}
    elif analysis.intent == "customer_revenue":
        match = re.search(
            r"customer(?:_id)?[=\s-]*(\d+)", question, re.IGNORECASE
        )
        customer = match.group(1) if match else ""
        queries = {
            "revenue": f"customer {customer} paid orders total revenue aggregation"
        }
    else:
        queries = {"answer": question}
    if aspect not in queries:
        raise ValueError(f"Unknown aspect {aspect} for intent {analysis.intent}")
    return queries[aspect]


def _levels(plan: SearchPlan) -> tuple[tuple[SearchStep, ...], ...]:
    """Stable topological batches; independent retrieve steps can run in parallel."""
    steps = {step.id: step for step in plan.steps}
    if len(steps) != len(plan.steps):
        raise ValueError("Duplicate step ID")
    processed: set[str] = set()
    batches: list[tuple[SearchStep, ...]] = []
    while len(processed) < len(steps):
        available = tuple(
            step for step in plan.steps
            if step.id not in processed
            and all(dep in processed for dep in step.depends_on)
        )
        if not available:
            raise ValueError("Plan contains missing dependencies or a cycle")
        batches.append(available)
        processed.update(step.id for step in available)
    return tuple(batches)


def validate_plan(plan: SearchPlan, request: Request, budgets: Budgets) -> tuple[tuple[SearchStep, ...], ...]:
    budgets.validate()
    if not plan.steps or plan.strategy not in STRATEGIES:
        raise ValueError("Invalid plan")
    router = SourceRouter()
    for step in plan.steps:
        if not step.id or not step.aspect or not step.query.strip():
            raise ValueError("Every step needs ID, aspect and query")
        if step.action not in {"retrieve", "calculate"}:
            raise ValueError("Unsupported plan action")
        if step.action == "calculate":
            if step.source or step.aspect != "calculation":
                raise ValueError("Calculation must be source-free and typed")
        elif step.source not in router.route(
            _intent_for_request(request), step.aspect, request
        ):
            raise PermissionError("Planner route disagrees with authorized router")
        if len(step.query) > 600:
            raise ValueError("Plan query exceeds allowed length")
    if sum(step.action == "retrieve" for step in plan.steps) > budgets.max_subqueries:
        raise ValueError("Plan exceeds max_subqueries")
    return _levels(plan)


def _intent_for_request(request: Request) -> str:
    from query_analysis import analyze

    return analyze(request).intent


class RetrievalPlanner:
    def plan(self, request: Request, analysis: Analysis, *,
             strategy: str, budgets: Budgets) -> SearchPlan:
        if strategy not in STRATEGIES:
            raise ValueError("Unsupported strategy")
        router = SourceRouter()
        if strategy in {"baseline", "rewrite", "multiquery", "hyde"}:
            source = router.route(analysis.intent, analysis.aspects[0], request)[0]
            steps = (
                SearchStep("S1", "answer", request.question, source),
            )
        else:
            steps_list = []
            for idx, aspect in enumerate(analysis.aspects, 1):
                query = aspect_query(analysis, aspect, request.question)
                for source_idx, source in enumerate(
                    router.route(analysis.intent, aspect, request), 1
                ):
                    steps_list.append(
                        SearchStep(
                            id=f"S{idx}_{source_idx}", aspect=aspect,
                            query=query, source=source,
                        )
                    )
            if (analysis.intent == "finance_compare"
                and {"q1_revenue", "q2_revenue"}.issubset(analysis.aspects)
                and strategy in {"planned", "adaptive"}):
                q1_id = next(step.id for step in steps_list if step.aspect == "q1_revenue")
                q2_id = next(step.id for step in steps_list if step.aspect == "q2_revenue")
                steps_list.append(SearchStep(
                    id="CALC", aspect="calculation",
                    query="Compute verified Q2 vs Q1 quarterly revenue growth",
                    source="", depends_on=(q1_id, q2_id), action="calculate",
                ))
            steps = tuple(steps_list)
        plan = SearchPlan(steps=steps, strategy=strategy)
        validate_plan(plan, request, budgets)
        return plan

    @staticmethod
    def batches(plan: SearchPlan) -> tuple[tuple[SearchStep, ...], ...]:
        return _levels(plan)