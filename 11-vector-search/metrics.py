"""Vector similarity arithmetic; higher similarity score is always better.

For L2, returned search scores are NEGATIVE squared Euclidean distance.
This is deliberate: the display can expose true distance separately.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

METRICS = ("cosine", "dot", "l2")


def validate_vector(raw: Sequence[float]) -> tuple[float, ...]:
    if not raw:
        raise ValueError("Embedding cannot be empty")
    result = []
    for value in raw:
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("Embedding must contain finite numeric values")
        result.append(float(value))
    return tuple(result)


def normalize(raw: Sequence[float]) -> tuple[float, ...]:
    vector = validate_vector(raw)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return tuple(value / norm for value in vector)


def similarity(a: Sequence[float], b: Sequence[float], metric: str) -> float:
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    a, b = validate_vector(a), validate_vector(b)
    if len(a) != len(b):
        raise ValueError("Query and indexed vectors must have matching dimensions")
    if metric == "dot":
        return sum(x * y for x, y in zip(a, b))
    if metric == "l2":
        return -sum((x - y) ** 2 for x, y in zip(a, b))
    magnitude = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    if magnitude == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    return sum(x * y for x, y in zip(a, b)) / magnitude


def score_to_distance(score: float, metric: str) -> float:
    """Return Euclidean distance for l2; otherwise no geometric distance."""
    if metric != "l2":
        raise ValueError("Only l2 score converts to Euclidean distance")
    return math.sqrt(max(0.0, -score))


def validate_search(top_k: int, ef_search: int | None = None) -> None:
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if ef_search is not None and (type(ef_search) is not int or ef_search < top_k):
        raise ValueError("ef_search must be an integer >= top_k")
