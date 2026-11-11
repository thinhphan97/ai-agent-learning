"""Optional real hnswlib index; requires separately installed hnswlib + NumPy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from contracts import SearchHit, SearchStats, VectorRecord
from exact import ExactKNN
from metadata import matches, normalize_filters
from metrics import validate_search, validate_vector


class HnswlibIndex:
    """Real hnswlib ANN with explicit post-filter or exact filtered fallback."""

    def __init__(
        self,
        records: Sequence[VectorRecord],
        *,
        metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 100,
        seed: int = 17,
    ):
        try:
            import hnswlib
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "Install optional packages: python -m pip install numpy hnswlib"
            ) from exc
        self.np = np
        self.exact = ExactKNN(records, metric)
        self.records = self.exact.records
        self.metric = metric
        self.dimensions = self.exact.dimensions
        self.model_id = self.exact.model_id
        self.index = hnswlib.Index(
            space={"cosine": "cosine", "dot": "ip", "l2": "l2"}[metric],
            dim=self.dimensions,
        )
        self.index.init_index(
            max_elements=len(self.records),
            ef_construction=ef_construction,
            M=m,
            random_seed=seed,
        )
        self.index.add_items(
            np.asarray([r.vector for r in self.records], dtype=np.float32),
            np.arange(len(self.records), dtype=np.int64),
        )

    def search(
        self,
        vector: Sequence[float],
        *,
        tenant_id: str,
        filters: Mapping[str, str] | None = None,
        top_k: int = 5,
        ef_search: int = 32,
        filter_mode: str = "post",
        min_score: float | None = None,
    ) -> tuple[list[SearchHit], SearchStats]:
        validate_search(top_k, ef_search)
        query = validate_vector(vector)
        if len(query) != self.dimensions:
            raise ValueError("Vector dimensions mismatch")
        canonical = normalize_filters(filters)
        if filter_mode == "exact-filter":
            return self.exact.search(
                query,
                tenant_id=tenant_id,
                filters=canonical,
                top_k=top_k,
                min_score=min_score,
            )
        if filter_mode != "post":
            raise ValueError("Invalid filter mode")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Trusted tenant scope required")
        self.index.set_ef(ef_search)
        k = min(top_k, len(self.records))
        ids, distances = self.index.knn_query(
            self.np.asarray([query], dtype=self.np.float32),
            k=k,
            num_threads=1,
        )
        output = []
        for idx, distance in zip(ids[0], distances[0]):
            record = self.records[int(idx)]
            if not matches(record.metadata, tenant_id, canonical):
                continue
            score = float(-distance if self.metric == "l2" else 1.0 - distance)
            if min_score is None or score >= min_score:
                output.append(SearchHit(record, score, len(output) + 1))
        # hnswlib does not export its internal visited count through this API.
        return output, SearchStats(0, 0, k)
