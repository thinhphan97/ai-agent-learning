"""Budgeted candidate selection: top-k, MMR diversity, or curated aspect coverage.

The aspect field is a synthetic metadata teaching hint. Production systems must
not rely on gold evaluation labels as runtime selection features.
"""
from __future__ import annotations

from contracts import Hit
from retrieval import tokenize


def _jaccard(a: str, b: str) -> float:
    aa, bb = set(tokenize(a)), set(tokenize(b))
    return len(aa & bb) / len(aa | bb) if aa | bb else 1.0


def select_candidates(
    retrieved: list[Hit],
    *,
    budget: int,
    strategy: str = "topk",
    required_aspects: tuple[str, ...] = (),
    mmr_lambda: float = 0.65,
) -> list[Hit]:
    if type(budget) is not int or budget < 1:
        raise ValueError("Candidate budget must be positive")
    if strategy not in {"topk", "mmr", "coverage"}:
        raise ValueError("Unknown selection strategy")
    if not 0 <= mmr_lambda <= 1:
        raise ValueError("MMR lambda must lie within [0,1]")
    if len({hit.child.id for hit in retrieved}) != len(retrieved):
        raise ValueError("Retrieved candidates must have distinct IDs")
    if strategy == "topk":
        return retrieved[:budget]
    remaining = list(retrieved)
    selected: list[Hit] = []
    if strategy == "coverage":
        # Choose best retrieved hit per aspect, then fill remaining slots.
        for aspect in dict.fromkeys(required_aspects):
            if len(selected) >= budget:
                break
            match = next((hit for hit in remaining if hit.child.aspect == aspect), None)
            if match is not None:
                selected.append(match)
                remaining.remove(match)
        selected.extend(remaining[:max(0, budget - len(selected))])
        return selected

    # MMR: relevance comes from RETRIEVAL RANK, not incomparable raw scores.
    while remaining and len(selected) < budget:
        def mmr_score(hit: Hit) -> tuple[float, int, str]:
            relevance = 1.0 / hit.rank
            redundancy = max(
                (_jaccard(hit.child.text, picked.child.text) for picked in selected),
                default=0.0,
            )
            return (mmr_lambda * relevance - (1 - mmr_lambda) * redundancy,
                    -hit.rank, hit.child.id)

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected
