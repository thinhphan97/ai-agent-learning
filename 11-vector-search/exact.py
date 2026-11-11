"""Exact KNN baseline. All authorized candidates are scored (O(N*d))."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from contracts import SearchHit, SearchStats, VectorRecord
from metadata import matches, normalize_filters
from metrics import similarity, validate_search, validate_vector


class ExactKNN:
    def __init__(self, records: Sequence[VectorRecord], metric: str = "cosine"):
        from metrics import METRICS

        if metric not in METRICS:
            raise ValueError("Unsupported metric")
        self.records = tuple(records)
        self.metric = metric
        if not records:
            raise ValueError("Exact index needs at least one record")
        self.dimensions = len(records[0].vector)
        self.model_id = records[0].embedding_model
        if len({r.id for r in records}) != len(records):
            raise ValueError("Duplicate vector record IDs")
        for record in records:
            validate_vector(record.vector)
            if len(record.vector) != self.dimensions:
                raise ValueError("Inconsistent embedding dimensions")
            if record.embedding_model != self.model_id:
                raise ValueError("Mixed embedding model versions")

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
        query = validate_vector(vector)
        if len(query) != self.dimensions:
            raise ValueError("Query embedding dimensions mismatch")
        canonical = normalize_filters(filters)
        ranked = []
        for record in self.records:
            if matches(record.metadata, tenant_id, canonical):
                score = similarity(query, record.vector, self.metric)
                if min_score is None or score >= min_score:
                    ranked.append(SearchHit(record, score))
        ranked.sort(key=lambda hit: (-hit.score, hit.record.id))
        output = [
            SearchHit(hit.record, hit.score, rank)
            for rank, hit in enumerate(ranked[:top_k], 1)
        ]
        authorized_count = sum(
            matches(record.metadata, tenant_id, canonical) for record in self.records
        )
        return output, SearchStats(
            distance_evaluations=authorized_count,
            visited_nodes=authorized_count,
            candidates_considered=len(ranked),
        )
