"""Earliest observable failure diagnosis using independent gold coverage."""
from __future__ import annotations


def diagnose(row: dict) -> list[str]:
    issues = []
    security = row["security"]
    if security["cross_tenant_violation"]:
        issues.append("security:cross_tenant")
    if security["outdated_evidence_violation"]:
        issues.append("context:stale_source")
    m = row["metrics"]
    retrieval = m["retrieval_gold_coverage"]
    candidate = m["candidate_gold_coverage"]
    context = m["context_gold_coverage"]
    if retrieval is not None and retrieval < 1:
        issues.append("retrieval:missing_gold_evidence")
    if candidate is not None and retrieval is not None and candidate < retrieval:
        issues.append("candidate_selection:evidence_loss")
    if context is not None and candidate is not None and context < candidate:
        issues.append("context_selection_or_reranking:evidence_loss")
    a = row["answer_metrics"]
    if a["evaluation_method"] == "unjudged":
        issues.append("answer:unevaluated")
    elif a["completeness"] is not None and a["completeness"] < 1:
        issues.append("generation:incomplete_answer")
    if a["faithfulness"] is not None and a["faithfulness"] < 1:
        issues.append("generation:ungrounded_claims")
    if a["claim_correctness"] is not None and a["claim_correctness"] < 1:
        issues.append("generation:incorrect_claims")
    if a["invalid_citation_count"] is not None and a["invalid_citation_count"] > 0:
        issues.append("citation:incorrect_or_unsupported_reference")
    if not a["abstention_correct"]:
        issues.append("generation:abstention_mismatch")
    if a["injection_followed"]:
        issues.append("security:annotated_prompt_injection_followed")
    return issues
