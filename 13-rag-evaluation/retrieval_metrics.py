"""Exact classical information retrieval metrics; no model-based guessing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from contracts import EvaluationCase


def _validate(ranked: Sequence[str], grades: Mapping[str, int], k: int):
    if type(k) is not int or k < 1:
        raise ValueError("K must be a positive integer")
    if len(set(ranked)) != len(ranked):
        raise ValueError("Ranked document IDs must be unique")
    if any(type(grade) is not int or not 0 <= grade <= 3 for grade in grades.values()):
        raise ValueError("Grades must be integers within 0..3")


def precision_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Fixed-k denominator; missing retrieval slots count as not relevant."""
    _validate(ranked, grades, k)
    return sum(grades.get(doc, 0) > 0 for doc in ranked[:k]) / k


def recall_at_k(
    ranked: Sequence[str], grades: Mapping[str, int], k: int
) -> float | None:
    _validate(ranked, grades, k)
    gold = {doc for doc, grade in grades.items() if grade > 0}
    if not gold:
        return None  # Undefined for an unanswerable case with zero gold positives.
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank_at_k(
    ranked: Sequence[str], grades: Mapping[str, int], k: int
) -> float | None:
    _validate(ranked, grades, k)
    if not any(grade > 0 for grade in grades.values()):
        return None
    for rank, doc in enumerate(ranked[:k], 1):
        if grades.get(doc, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float | None:
    """Graded nDCG using (2**grade - 1) / log2(rank+1)."""
    _validate(ranked, grades, k)
    positive = sorted((g for g in grades.values() if g > 0), reverse=True)
    if not positive:
        return None

    def gain(values):
        return sum(
            (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(values, 1)
        )

    dcg = gain([grades.get(doc, 0) for doc in ranked[:k]])
    idcg = gain(positive[:k])
    return dcg / idcg if idcg else None


def evidence_coverage(
    case: EvaluationCase, selected_ids: Sequence[str]
) -> float | None:
    """Separate required FACT coverage from document relevance/Recall@K.

    It is a gold-document proxy: the presence of a supporting document
    does not prove the answer fully covers its fact.
    """
    if not case.required_facts:
        return None
    selected = set(selected_ids)
    return sum(
        bool(selected.intersection(fact.supporting_doc_ids))
        for fact in case.required_facts
    ) / len(case.required_facts)


def retrieval_metrics(case: EvaluationCase, retrieved: tuple[str, ...], k: int):
    return {
        "precision_at_k": precision_at_k(retrieved, case.relevance_grades, k),
        "recall_at_k": recall_at_k(retrieved, case.relevance_grades, k),
        "reciprocal_rank_at_k": reciprocal_rank_at_k(
            retrieved, case.relevance_grades, k
        ),
        "ndcg_at_k": ndcg_at_k(retrieved, case.relevance_grades, k),
    }
