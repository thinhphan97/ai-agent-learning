"""Fail-closed evaluation gates. Policy breaches never average away."""

from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "data" / "gates.json"

SCORES = {
    "min_mean_retrieval_recall": "mean_retrieval_recall",
    "min_mean_candidate_gold_coverage": "mean_candidate_gold_coverage",
    "min_mean_context_gold_coverage": "mean_context_gold_coverage",
    "min_mean_faithfulness": "mean_faithfulness",
    "min_mean_claim_correctness": "mean_claim_correctness",
    "min_mean_completeness": "mean_completeness",
    "min_mean_answer_relevancy": "mean_answer_relevancy",
    "min_abstention_accuracy": "abstention_accuracy",
}
COUNTERS = {
    "max_unauthorized_documents": "unauthorized_documents",
    "max_stale_documents": "stale_documents",
    "max_unfaithful_claims": "unfaithful_claims",
    "max_invalid_citations": "invalid_citations",
    "max_injection_followed": "injection_followed_cases",
}


def load_gates(path: Path = DEFAULT) -> dict:
    gates = json.loads(path.read_text(encoding="utf-8"))
    if set(gates) != set(SCORES) | set(COUNTERS) | {"require_answer_evaluation"}:
        raise ValueError("Unknown or missing gate fields")
    for name in SCORES:
        value = gates[name]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Minimum score gate must lie within [0,1]")
    for name in COUNTERS:
        if type(gates[name]) is not int or gates[name] < 0:
            raise ValueError("Maximum count gate must be nonnegative integer")
    if type(gates["require_answer_evaluation"]) is not bool:
        raise ValueError("require_answer_evaluation must be boolean")
    return gates


ANSWER_METRIC_RULES = {
    "min_mean_faithfulness", "min_mean_claim_correctness",
    "min_mean_completeness", "min_mean_answer_relevancy",
}
ANSWER_COUNT_RULES = {
    "max_unfaithful_claims", "max_invalid_citations", "max_injection_followed",
}


def apply_gates(aggregates: dict, gates: dict) -> dict:
    failures = []
    skipped = []
    for rule, metric in SCORES.items():
        observed = aggregates[metric]
        minimum = gates[rule]
        if observed is None:
            if (not gates["require_answer_evaluation"]
                    and rule in ANSWER_METRIC_RULES and minimum == 0):
                skipped.append(f"{metric}: NOT MEASURED (answer gate disabled)")
                continue
            failures.append(f"{metric}: NOT MEASURED (minimum {minimum})")
        elif observed + 1e-9 < minimum:
            failures.append(f"{metric}: {observed:.4f} below minimum {minimum:.4f}")
    for rule, metric in COUNTERS.items():
        observed = aggregates[metric]
        maximum = gates[rule]
        if observed is None:
            if not gates["require_answer_evaluation"] and rule in ANSWER_COUNT_RULES:
                skipped.append(f"{metric}: NOT MEASURED (answer gate disabled)")
                continue
            failures.append(f"{metric}: NOT MEASURED (maximum {maximum})")
        elif observed > maximum:
            failures.append(f"{metric}: {observed} exceeds maximum {maximum}")
    if gates["require_answer_evaluation"] and aggregates["unjudged_cases"] > 0:
        failures.append(
            f"unjudged_cases: {aggregates['unjudged_cases']} cases have no "
            "human annotations or selected judge"
        )
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "skipped": skipped,
        "scope": "full_rag" if gates["require_answer_evaluation"] else "retrieval_only",
    }