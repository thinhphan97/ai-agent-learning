"""Score each RAG stage, keep hard policy failures independent of averages."""
from __future__ import annotations

from contracts import AnswerLabels, Document, EvaluationCase, Trace
from answer_metrics import score_answer
from dataset import is_current
from retrieval_metrics import evidence_coverage, retrieval_metrics
from runtime_stub import validate_trace


def evaluate_case(
    case: EvaluationCase,
    trace: Trace,
    documents: dict[str, Document],
    *,
    k: int,
    labels: AnswerLabels | None = None,
    judge_result: dict[str, object] | None = None,
) -> dict[str, object]:
    validate_trace(trace, case.id, case.tenant_id, documents)
    if judge_result is not None and labels is not None:
        raise ValueError("Select hand annotations OR a model judge, not both")
    if judge_result is not None and any(
        documents[doc].tenant_id != case.tenant_id
        for doc in trace.retrieved
    ):
        raise PermissionError("Judge cannot receive a trace with unauthorized sources")

    metrics = retrieval_metrics(case, trace.retrieved, k)
    metrics["candidate_gold_coverage"] = evidence_coverage(case, trace.candidates)
    metrics["context_gold_coverage"] = evidence_coverage(
        case, [e.doc_id for e in trace.context]
    )
    # None indicates undefined, not 100% coverage, for no gold positives.
    metrics["retrieval_gold_coverage"] = evidence_coverage(case, trace.retrieved)
    answer = score_answer(case, trace, labels)
    if judge_result is not None:
        # Model scores are fallible, but invalid citation IDs are testable
        # deterministically and MUST NOT be overruled by the judge.
        mandatory_invalid = len(answer["unknown_citation_ids"])
        if case.answerable and not trace.abstained and not answer["citation_ids"]:
            mandatory_invalid += 1
        scored = dict(judge_result)
        scored["invalid_citation_count"] = max(
            mandatory_invalid, scored["invalid_citation_count"]
        )
        answer.update(scored)

    all_stages = {
        "retrieved": list(trace.retrieved),
        "candidates": list(trace.candidates),
        "reranked": list(trace.reranked),
        "context": [{"evidence_id": e.evidence_id, "doc_id": e.doc_id}
                    for e in trace.context],
    }
    unauthorized = sorted({
        doc_id for values in (trace.retrieved, trace.candidates, trace.reranked)
        for doc_id in values if documents[doc_id].tenant_id != case.tenant_id
    } | {
        e.doc_id for e in trace.context
        if documents[e.doc_id].tenant_id != case.tenant_id
    })
    stale = sorted({
        e.doc_id for e in trace.context
        if not is_current(documents[e.doc_id], case.as_of_date)
    })
    return {
        "case_id": case.id,
        "question": case.question,
        "tenant_id": case.tenant_id,
        "as_of_date": case.as_of_date,
        "answerable": case.answerable,
        "trace_provenance": trace.provenance,
        "stages": all_stages,
        "metrics": metrics,
        "answer": trace.answer,
        "answer_metrics": answer,
        "security": {
            "unauthorized_document_ids": unauthorized,
            "stale_context_document_ids": stale,
            "cross_tenant_violation": bool(unauthorized),
            "outdated_evidence_violation": bool(stale),
        },
        "usage": {
            "latency_ms": trace.latency_ms,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "measurement": "synthetic_fixture" if trace.provenance == "fixture"
                           else ("provider_generation" if trace.provenance == "live_generation"
                                 else "user_supplied_trace"),
        },
    }