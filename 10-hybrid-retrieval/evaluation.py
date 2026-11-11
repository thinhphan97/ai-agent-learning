"""Offline retrieval evaluation using fixed labels independent of the retrievers.

No LLM judge and no final-answer accuracy claims.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from contracts import EvalCase, Hit
from pipeline import RetrievalPipeline
from reranker import Reranker


def score_case(
    hits: Sequence[Hit], expected_ids: Sequence[str], k: int
) -> dict[str, float]:
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Gold IDs must be distinct and nonempty")
    relevant = set(expected_ids)
    ranked = [hit.chunk.id for hit in hits[:k]]
    recalled = len(relevant.intersection(ranked)) / len(relevant)
    first = next((i for i, item in enumerate(ranked, 1) if item in relevant), None)
    mrr = 1.0 / first if first else 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked, 1)
        if item in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1)
    )
    return {"recall": recalled, "mrr": mrr, "ndcg": dcg / ideal}


def evaluate(
    pipeline: RetrievalPipeline,
    cases: Sequence[EvalCase],
    *,
    method: str,
    k: int = 3,
    reranker: Reranker | None = None,
) -> dict[str, object]:
    if not cases or type(k) is not int or not 1 <= k <= 100:
        raise ValueError("Need test cases and valid k")
    records = []
    for case in cases:
        hits = pipeline.search(
            case.question,
            method=method,
            tenant_id=case.tenant_id,
            filters=case.filters,
            top_k=k,
            candidate_k=max(k, 8),
            reranker=reranker,
        )
        metrics = score_case(hits, case.expected_ids, k)
        records.append(
            {
                "question": case.question,
                "tenant_id": case.tenant_id,
                "expected_ids": list(case.expected_ids),
                "retrieved_ids": [hit.chunk.id for hit in hits],
                **metrics,
            }
        )
    return {
        "method": method + ("+reranker" if reranker else ""),
        "cases": len(records),
        "recall_at_k": round(sum(c["recall"] for c in records) / len(records), 4),
        "mrr_at_k": round(sum(c["mrr"] for c in records) / len(records), 4),
        "ndcg_at_k": round(sum(c["ndcg"] for c in records) / len(records), 4),
        "details": records,
    }
