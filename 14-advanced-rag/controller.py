"""Bounded retrieval controller: plan, fan-out, fuse, validate, targeted recover.

All budgets are enforced before expensive/search operations. Stable trace order
is preserved even though independent steps search concurrently.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from contracts import Analysis, Budgets, Request, SearchHit
from corpus import Corpus
from evidence import EvidenceBuilder, grounded_answer, supports
from planning import RetrievalPlanner, aspect_query
from query_analysis import analyze
from retrieval import LocalRetriever, rrf_fuse
from routing import SourceRouter
from transform import DeterministicTransformer, LLMTransformer

SCENARIOS = frozenset({"normal", "delayed-drivers", "no-drivers"})


class AdvancedRAG:
    def __init__(
        self, *, corpus: Corpus | None = None,
        transformer_provider: str = "mock",
        transformer=None,
    ):
        self.corpus = corpus or Corpus.load()
        self.retriever = LocalRetriever(self.corpus)
        self.builder = EvidenceBuilder(self.corpus)
        self.planner = RetrievalPlanner()
        self.router = SourceRouter()
        if transformer is not None:
            self.transformer = transformer
        elif transformer_provider == "mock":
            self.transformer = DeterministicTransformer()
        elif transformer_provider == "llm":
            self.transformer = LLMTransformer()
        else:
            raise ValueError("Unknown query transformer provider")
        self.provider = transformer_provider

    def run(
        self, request: Request, *, strategy: str = "adaptive",
        budgets: Budgets = Budgets(), scenario: str = "normal",
    ) -> dict[str, object]:
        if scenario not in SCENARIOS:
            raise ValueError("Unknown fault scenario")
        if not request.tenant_id or not request.tenant_id.strip():
            raise ValueError("A trusted tenant is required")
        if not request.allowed_sources:
            raise ValueError("At least one trusted source is required")
        budgets.validate()
        analysis = analyze(request)
        plan = self.planner.plan(
            request, analysis, strategy=strategy, budgets=budgets
        )
        events: list[dict[str, object]] = []
        groups: list[list[SearchHit]] = []
        calls = 0
        generated_subqueries = 0
        llm_calls = 0
        rounds = 0

        def execute_round(steps, round_no: int) -> bool:
            nonlocal calls, generated_subqueries, llm_calls
            retrieves = [step for step in steps if step.action == "retrieve"]
            if generated_subqueries + len(retrieves) > budgets.max_subqueries:
                events.append({
                    "event": "budget_stop", "reason": "max_subqueries",
                    "round": round_no,
                })
                return False
            expected_queries = (
                3 if strategy == "multiquery" else 1
            ) * len(retrieves)
            if calls + expected_queries > budgets.max_search_calls:
                events.append({
                    "event": "budget_stop", "reason": "max_search_calls",
                    "round": round_no,
                })
                return False
            if (self.provider == "llm"
                    and strategy in {"rewrite", "multiquery", "hyde"}
                    and llm_calls + len(retrieves) > budgets.max_llm_calls):
                events.append({
                    "event": "budget_stop", "reason": "max_llm_calls",
                    "round": round_no,
                })
                return False
            queries: list[tuple[str, str, str, str]] = []
            for step in steps:
                if step.action != "retrieve":
                    continue
                if generated_subqueries >= budgets.max_subqueries:
                    events.append({
                        "event": "budget_stop", "reason": "max_subqueries",
                        "round": round_no,
                    })
                    return False
                generated_subqueries += 1
                if self.provider == "llm" and strategy in {
                    "rewrite", "multiquery", "hyde"
                }:
                    if llm_calls >= budgets.max_llm_calls:
                        events.append({
                            "event": "budget_stop", "reason": "max_llm_calls",
                            "round": round_no,
                        })
                        return False
                    llm_calls += 1
                variants = self.transformer.transform(
                    step.query, aspect=step.aspect, mode=strategy
                )
                if not variants or len(set(variants)) != len(variants):
                    raise ValueError("Transformer must return unique nonempty queries")
                from query_analysis import validate_rewrite
                for query in variants:
                    validate_rewrite(step.query, query)
                    queries.append((step.id, step.aspect, step.source, query))
            if not queries:
                return True
            if calls + len(queries) > budgets.max_search_calls:
                events.append({
                    "event": "budget_stop", "reason": "max_search_calls",
                    "round": round_no,
                })
                return False
            calls += len(queries)
            events.append({
                "event": "parallel_retrieval_batch", "round": round_no,
                "step_ids": list(dict.fromkeys(q[0] for q in queries)),
                "query_count": len(queries),
            })
            with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
                futures = [
                    pool.submit(
                        self.retriever.search,
                        request, query=query, source=source, aspect=aspect,
                        limit=budgets.max_hits_per_search,
                        scenario=scenario, round_number=round_no,
                    )
                    for _, aspect, source, query in queries
                ]
                # Results collected in stable plan order, not completion order.
                retrieved = [future.result() for future in futures]
            groups.extend(retrieved)
            for (step_id, aspect, source, query), hits in zip(queries, retrieved):
                events.append({
                    "event": "search", "round": round_no,
                    "step_id": step_id, "aspect": aspect,
                    "source": source, "query": query,
                    "doc_ids": [hit.doc_id for hit in hits],
                })
            return True

        for batch in self.planner.batches(plan):
            retrieves = [step for step in batch if step.action == "retrieve"]
            if retrieves:
                if not execute_round(retrieves, 1):
                    break
                rounds = 1
            for step in batch:
                if step.action == "calculate":
                    ranked_now = rrf_fuse(groups)
                    has_q1 = any(supports(
                        "q1_revenue", self.corpus.verify(request, h.doc_id).text,
                        request.question,
                    ) for h in ranked_now)
                    has_q2 = any(supports(
                        "q2_revenue", self.corpus.verify(request, h.doc_id).text,
                        request.question,
                    ) for h in ranked_now)
                    events.append({
                        "event": "calculation_dependency",
                        "step_id": step.id,
                        "status": "ready" if has_q1 and has_q2 else "blocked",
                        "depends_on": list(step.depends_on),
                    })

        def pack():
            ranked = rrf_fuse(groups)
            if len({hit.doc_id for hit in ranked}) != len(ranked):
                raise AssertionError("Fusion must deduplicate documents")
            evidence, missing, dropped = self.builder.build(
                request, ranked, analysis.aspects, budgets,
                coverage_aware=strategy in {"decompose", "planned", "adaptive"},
            )
            return ranked, evidence, missing, dropped

        ranked, evidence, missing, dropped = pack()
        events.append({
            "event": "coverage_check", "round": rounds,
            "present_aspects": [a for a in analysis.aspects if a not in missing],
            "missing_aspects": list(missing),
        })

        stop = "sufficient" if not missing else "no_adaptive_retry"
        if strategy == "adaptive":
            def needs_more_retrieval() -> tuple[str, ...]:
                return tuple(
                    aspect for aspect in missing
                    if not any(
                        supports(
                            aspect, self.corpus.verify(request, hit.doc_id).text,
                            request.question,
                        )
                        for hit in ranked
                    )
                )

            while missing and rounds < budgets.max_rounds:
                retrieval_gaps = needs_more_retrieval()
                if not retrieval_gaps:
                    stop = "context_budget_limited"
                    events.append({
                        "event": "context_budget_gap",
                        "missing_aspects": list(missing),
                        "message": "Evidence exists in retrieved hits but was not packed",
                    })
                    break
                targeted = []
                for index, aspect in enumerate(retrieval_gaps, 1):
                    for source in self.router.route(analysis.intent, aspect, request):
                        from contracts import SearchStep
                        targeted.append(SearchStep(
                            id=f"R{rounds + 1}_{index}_{source}",
                            aspect=aspect, source=source,
                            query=aspect_query(
                                analysis, aspect, request.question, recovery=True
                            ),
                        ))
                if not targeted:
                    stop = "no_authorized_recovery_source"
                    break
                events.append({
                    "event": "targeted_recovery", "round": rounds + 1,
                    "missing_aspects": list(retrieval_gaps),
                })
                previous_missing = set(missing)
                if not execute_round(targeted, rounds + 1):
                    stop = "budget_exhausted"
                    break
                rounds += 1
                ranked, evidence, missing, dropped = pack()
                events.append({
                    "event": "coverage_check", "round": rounds,
                    "present_aspects": [a for a in analysis.aspects if a not in missing],
                    "missing_aspects": list(missing),
                })
                if not missing:
                    stop = "sufficient_after_recovery"
                elif not needs_more_retrieval():
                    stop = "context_budget_limited"
                    break
                elif set(missing) == previous_missing:
                    # New irrelevant documents are NOT evidence progress.
                    stop = "no_new_evidence"
                    break
            if missing and stop in {"no_adaptive_retry", "sufficient"}:
                stop = (
                    "context_budget_limited" if not needs_more_retrieval()
                    else "max_rounds_exhausted" if rounds >= budgets.max_rounds
                    else "budget_exhausted"
                )

        answer = grounded_answer(analysis, evidence, missing)
        events.append({
            "event": "complete", "stop_reason": stop,
            "answer_status": answer["status"],
        })
        return {
            "question": request.question,
            "tenant_id": request.tenant_id,
            "as_of_date": request.as_of_date,
            "strategy": strategy,
            "provider": self.provider,
            "scenario": scenario,
            "analysis": asdict(analysis),
            "plan": [asdict(step) for step in plan.steps],
            "events": events,
            "retrieved": [
                {"id": hit.doc_id, "rank": hit.rank,
                 "rrf_score": hit.score, "source": hit.source}
                for hit in ranked
            ],
            "evidence": [asdict(item) for item in evidence],
            "missing_aspects": list(missing),
            "coverage_by_runtime_heuristics": (
                (len(analysis.aspects) - len(missing)) / len(analysis.aspects)
            ),
            "dropped_doc_ids": dropped,
            "answer": answer,
            "usage": {
                "retrieval_rounds": rounds,
                "search_calls": calls,
                "subqueries": generated_subqueries,
                "llm_transform_calls": llm_calls,
                "max_search_calls": budgets.max_search_calls,
                "max_rounds": budgets.max_rounds,
            },
            "stop_reason": stop,
            "evidence_provenance_note": (
                "Grounding uses explicitly narrow runtime text heuristics. "
                "Hypothetical search hints are not evidence. "
                "No claim-level entailment is automatically established."
            ),
        }