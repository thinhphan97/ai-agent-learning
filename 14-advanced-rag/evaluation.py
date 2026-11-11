"""Held-out synthetic gold benchmarking; gold never enters AdvancedRAG.run."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from contracts import Budgets, Request
from controller import AdvancedRAG
from corpus import Corpus
from planning import STRATEGIES

BASE = Path(__file__).resolve().parent / "data"


def _read_suite():
    cases = json.loads((BASE / "evaluation_cases.json").read_text(encoding="utf-8"))
    labels = json.loads((BASE / "gold_labels.json").read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases["cases"]]
    if len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError("Case ID mismatch between execution and gold labels")
    known = Corpus.load().documents
    for label in labels.values():
        if not set(label["relevant_docs"]).issubset(known):
            raise ValueError("Unknown gold document")
        for group in label["required_facts"].values():
            if not group or not set(group).issubset(label["relevant_docs"]):
                raise ValueError("Bad independent gold fact mapping")
    return cases, labels


def _fact_coverage(ids: set[str], required: dict[str, list[str]]) -> float | None:
    if not required:
        return None
    return sum(bool(ids.intersection(support)) for support in required.values()) / len(required)


def _mean_nullable(values: list[float | None]) -> float | None:
    scored = [v for v in values if v is not None]
    return round(mean(scored), 6) if scored else None


def evaluate_strategy(strategy: str, *,
                      budgets: Budgets = Budgets(),
                      cases_filter: tuple[str, ...] = ()) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")
    cases, gold = _read_suite()
    case_ids = {case["id"] for case in cases["cases"]}
    if not set(cases_filter).issubset(case_ids):
        raise ValueError("Unknown benchmark case")
    controller = AdvancedRAG()
    results = []
    for case in cases["cases"]:
        if cases_filter and case["id"] not in cases_filter:
            continue
        request = Request(
            question=case["question"], tenant_id=case["tenant_id"],
            as_of_date=case["as_of_date"],
            allowed_sources=tuple(case["sources"]),
        )
        trace = controller.run(
            request, strategy=strategy, budgets=budgets, scenario=case["scenario"]
        )
        labels = gold[case["id"]]
        retrieved = {hit["id"] for hit in trace["retrieved"]}
        evidence = {item["doc_id"] for item in trace["evidence"]}
        gold_ids = set(labels["relevant_docs"])
        recall = (
            len(retrieved & gold_ids) / len(gold_ids)
            if gold_ids else None
        )
        results.append({
            "case_id": case["id"],
            "scenario": case["scenario"],
            "recall": recall,
            "retrieval_gold_coverage": _fact_coverage(retrieved, labels["required_facts"]),
            "context_gold_coverage": _fact_coverage(evidence, labels["required_facts"]),
            "runtime_heuristic_coverage": trace["coverage_by_runtime_heuristics"],
            "status": trace["answer"]["status"],
            "stop_reason": trace["stop_reason"],
            "search_calls": trace["usage"]["search_calls"],
            "retrieval_rounds": trace["usage"]["retrieval_rounds"],
            "safe_scope": all(
                controller.corpus.verify(request, doc_id)
                for doc_id in retrieved | evidence
            ),
            "retrieved_doc_ids": sorted(retrieved),
            "context_doc_ids": [e["doc_id"] for e in trace["evidence"]],
        })
    if not results:
        raise ValueError("Selected benchmark must include cases")
    return {
        "dataset_version": cases["dataset_version"],
        "strategy": strategy,
        "benchmark_type": "synthetic_gold_document_presence_not_semantic_faithfulness",
        "cases": results,
        "mean_recall": _mean_nullable([r["recall"] for r in results]),
        "mean_retrieval_gold_coverage": _mean_nullable([
            r["retrieval_gold_coverage"] for r in results
        ]),
        "mean_context_gold_coverage": _mean_nullable([
            r["context_gold_coverage"] for r in results
        ]),
        "total_search_calls": sum(r["search_calls"] for r in results),
        "partial_or_abstained": sum(
            r["status"] != "answered" for r in results
        ),
        "unsafe_scope_count": sum(not r["safe_scope"] for r in results),
    }


def compare_strategies(*, budgets: Budgets = Budgets(),
                       cases_filter: tuple[str, ...] = ()) -> dict:
    strategies = (
        "baseline", "rewrite", "multiquery",
        "decompose", "planned", "adaptive", "hyde",
    )
    reports = [
        evaluate_strategy(s, budgets=budgets, cases_filter=cases_filter)
        for s in strategies
    ]
    return {
        "dataset_version": reports[0]["dataset_version"],
        "note": (
            "Equal corpus and gold cases, deterministic offline mock searches. "
            "No label access during runtime; not a production benchmark."
        ),
        "strategies": [
            {k: val for k, val in report.items() if k != "cases"}
            for report in reports
        ],
        "case_details": {
            report["strategy"]: report["cases"] for report in reports
        },
    }
