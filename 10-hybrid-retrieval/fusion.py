"""Reciprocal Rank Fusion: combine ranked results without mixing raw score units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from contracts import Hit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    *,
    rank_constant: int = 60,
    top_k: int = 5,
) -> list[Hit]:
    if type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("rank_constant must be a positive integer")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be in 1..100")
    scores: dict[str, float] = defaultdict(float)
    chunk_by_id = {}
    source_methods: dict[str, list[str]] = defaultdict(list)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id in chunk_by_id and chunk_by_id[chunk_id] != hit.chunk:
                raise ValueError(f"Inconsistent chunk payload for id={chunk_id}")
            chunk_by_id[chunk_id] = hit.chunk
            scores[chunk_id] += 1.0 / (rank_constant + rank)
            source_methods[chunk_id].append(hit.method)
    ranked = [
        Hit(chunk_by_id[key], score, "hybrid", tuple(source_methods[key]))
        for key, score in scores.items()
    ]
    return sorted(ranked, key=lambda item: (-item.score, item.chunk.id))[:top_k]
