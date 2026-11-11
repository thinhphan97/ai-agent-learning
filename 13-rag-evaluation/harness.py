"""Independent RAG evaluator with stage-wise metrics and quality gates."""

from __future__ import annotations

from statistics import mean

from answer_metrics import load_labels, select_labels
from contracts import Trace
from dataset import load_suite
from evaluator import evaluate_case
from quality_gates import apply_gates, load_gates
from runtime_stub import validate_trace


def _average(rows: list[dict], group: str, key: str) -> float | None:
    values = [row[group][key] for row in rows if row[group][key] is not None]
    return round(mean(values), 6) if values else None


def _count(rows: list[dict], key: str) -> int | None:
    numbers = [row["answer_metrics"][key] for row in rows]
    if any(value is None for value in numbers):
        return None  # Not all observed answers were actually evaluated.
    return sum(numbers)


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate zero cases")
    return {
        "case_count": len(rows),
        "mean_precision_at_k": _average(rows, "metrics", "precision_at_k"),
        "mean_retrieval_recall": _average(rows, "metrics", "recall_at_k"),
        "mean_reciprocal_rank": _average(rows, "metrics", "reciprocal_rank_at_k"),
        "mean_ndcg": _average(rows, "metrics", "ndcg_at_k"),
        "mean_retrieval_gold_coverage": _average(
            rows, "metrics", "retrieval_gold_coverage"
        ),
        "mean_candidate_gold_coverage": _average(
            rows, "metrics", "candidate_gold_coverage"
        ),
        "mean_context_gold_coverage": _average(
            rows, "metrics", "context_gold_coverage"
        ),
        "mean_faithfulness": _average(rows, "answer_metrics", "faithfulness"),
        "mean_claim_correctness": _average(rows, "answer_metrics", "claim_correctness"),
        "mean_completeness": _average(rows, "answer_metrics", "completeness"),
        "mean_answer_relevancy": _average(rows, "answer_metrics", "answer_relevancy"),
        "abstention_accuracy": round(
            mean(
                1.0 if row["answer_metrics"]["abstention_correct"] else 0.0
                for row in rows
            ),
            6,
        ),
        "unjudged_cases": sum(
            row["answer_metrics"]["evaluation_method"] == "unjudged" for row in rows
        ),
        "unauthorized_documents": sum(
            len(row["security"]["unauthorized_document_ids"]) for row in rows
        ),
        "stale_documents": sum(
            len(row["security"]["stale_context_document_ids"]) for row in rows
        ),
        "unfaithful_claims": _count(rows, "unfaithful_claim_count"),
        "invalid_citations": _count(rows, "invalid_citation_count"),
        "injection_followed_cases": (
            None
            if any(row["answer_metrics"]["injection_followed"] is None for row in rows)
            else sum(bool(row["answer_metrics"]["injection_followed"]) for row in rows)
        ),
        # The next values are synthetic fixture inputs unless the trace provenance
        # is live_generation; report never calls them real wall-clock measurements.
        "mean_observed_latency_ms": round(
            mean(row["usage"]["latency_ms"] for row in rows), 3
        ),
        "total_observed_input_tokens": sum(
            row["usage"]["input_tokens"] for row in rows
        ),
        "total_observed_output_tokens": sum(
            row["usage"]["output_tokens"] for row in rows
        ),
    }


def run_evaluation(
    traces: dict[str, Trace],
    *,
    case_ids: list[str] | None = None,
    scenario: str = "baseline",
    k: int = 5,
    answer_mode: str = "labels",
    gates_path=None,
    llm_gateway=None,
    annotations_path=None,
) -> dict:
    if type(k) is not int or k < 1 or k > 200:
        raise ValueError("K must be within 1..200")
    if answer_mode not in {"labels", "none", "llm"}:
        raise ValueError("Unknown answer evaluator")
    if annotations_path is not None and answer_mode != "labels":
        raise ValueError("Human annotations are only used with --judge labels")
    version, docs, cases = load_suite()
    chosen = list(cases) if case_ids is None else case_ids
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("Evaluation case selection must be distinct and nonempty")
    if any(case not in cases for case in chosen):
        raise ValueError("Unknown evaluation case ID")
    if any(case not in traces for case in chosen):
        raise ValueError("Missing execution trace for requested case")
    if answer_mode == "labels" and annotations_path is None and any(
        traces[cid].provenance != "fixture" for cid in chosen
    ):
        raise ValueError(
            "External/live answers require explicit human annotations or "
            "an explicitly selected model judge"
        )
    if answer_mode == "llm" and llm_gateway is None:
        raise ValueError("LLM-as-a-Judge requires explicit model gateway")
    raw_labels = (
        load_labels(annotations_path) if annotations_path is not None
        else load_labels() if answer_mode == "labels" else None
    )

    results = []
    for case_id in chosen:
        case, trace = cases[case_id], traces[case_id]
        validate_trace(trace, case.id, case.tenant_id, docs)
        labels = (
            select_labels(
                raw_labels, scenario, trace,
                allow_external=annotations_path is not None,
            )
            if raw_labels is not None
            else None
        )
        judge_result = None
        if answer_mode == "llm":
            # Never send documents across tenant boundaries to a provider.
            cross_tenant = any(
                docs[doc_id].tenant_id != case.tenant_id for doc_id in trace.retrieved
            )
            if not cross_tenant:
                judge_result = llm_gateway.judge(case, trace, docs)
        results.append(
            evaluate_case(
                case,
                trace,
                docs,
                k=k,
                labels=labels,
                judge_result=judge_result,
            )
        )
    aggregates = aggregate(results)
    gates = load_gates() if gates_path is None else load_gates(gates_path)
    verdict = apply_gates(aggregates, gates)
    return {
        "dataset_version": version,
        "scenario": scenario,
        "k": k,
        "case_ids": chosen,
        "answer_mode": answer_mode,
        "metrics_note": (
            "IR metrics use exact gold document IDs. Stage gold coverage is an "
            "evidence-document proxy. Offline answer scores use hand-labeled "
            "synthetic claims, not automatic semantic entailment. No-gold "
            "retrieval metrics are undefined and excluded from their means."
        ),
        "aggregates": aggregates,
        "quality_gates": verdict,
        "cases": results,
    }