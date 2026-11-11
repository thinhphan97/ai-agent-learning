"""Optional vectorized Exact KNN baseline using real NumPy operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from contracts import SearchHit, SearchStats, VectorRecord
from exact import ExactKNN
from metadata import matches, normalize_filters
from metrics import validate_search, validate_vector


class NumPyExactKNN:
    """Exact vectorized scores over every authorized candidate."""

    def __init__(self, records: Sequence[VectorRecord], metric: str = "cosine"):
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("Install optional NumPy: python -m pip install numpy") from exc
        self.np = np
        self.exact = ExactKNN(records, metric)
        self.records = self.exact.records
        self.metric = metric
        self.dimensions = self.exact.dimensions
        self.matrix = np.asarray(
            [r.vector for r in self.records], dtype=np.float64,
        )
        if metric == "cosine" and (np.linalg.norm(self.matrix, axis=1) == 0).any():
            raise ValueError("Cosine cannot index zero vectors")

    def search(
        self,
        vector: Sequence[float],
        *,
        tenant_id: str,
        filters: Mapping[str, str] | None = None,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> tuple[list[SearchHit], SearchStats]:
        validate_search(top_k)
        query = self.np.asarray(validate_vector(vector), dtype=self.np.float64)
        if query.shape != (self.dimensions,):
            raise ValueError("Query embedding dimensions mismatch")
        if self.metric == "cosine" and self.np.linalg.norm(query) == 0:
            raise ValueError("Cosine query must not be zero")
        canonical = normalize_filters(filters)
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Trusted tenant_id required")
        mask = self.np.asarray([
            matches(r.metadata, tenant_id, canonical) for r in self.records
        ], dtype=bool)
        indices = self.np.flatnonzero(mask)
        if len(indices) == 0:
            return [], SearchStats(0, 0, 0)
        values = self.matrix[indices]
        if self.metric == "dot":
            scores = values @ query
        elif self.metric == "l2":
            scores = -self.np.sum((values - query) ** 2, axis=1)
        else:
            scores = (values @ query) / (
                self.np.linalg.norm(values, axis=1) * self.np.linalg.norm(query)
            )
        matches_found = [
            SearchHit(self.records[int(idx)], float(score))
            for idx, score in zip(indices, scores)
            if min_score is None or float(score) >= min_score
        ]
        matches_found.sort(key=lambda hit: (-hit.score, hit.record.id))
        hits = [
            SearchHit(hit.record, hit.score, rank)
            for rank, hit in enumerate(matches_found[:top_k], 1)
        ]
        return hits, SearchStats(
            len(indices), len(indices), len(matches_found)
        )
