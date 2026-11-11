"""Qdrant SDK adapter: explicit Docker server OR optional in-process local mode.

For a service backend use QdrantIndex(url="http://127.0.0.1:6333").
For SDK local mode use QdrantLocalIndex(records), kept for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from contracts import SearchHit, SearchStats, VectorRecord
from exact import ExactKNN
from metadata import matches, normalize_filters, normalize_metadata
from metrics import similarity, validate_search, validate_vector


def _point_id(record_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"ai-agent-learning-lab11:{record_id}"))


class QdrantIndex:
    """Explicitly scoped collection. No remote collection is created on a query."""

    def __init__(
        self,
        records: Sequence[VectorRecord],
        *,
        metric: str = "cosine",
        local_path: str | Path = ":memory:",
        url: str | None = None,
        api_key: str | None = None,
        collection: str = "lab11_documents",
        reindex: bool = False,
        initialize: bool = True,
    ):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise ImportError(
                "Install optional Qdrant client: python -m pip install -r requirements.txt"
            ) from exc

        if url is not None and str(local_path) != ":memory:":
            raise ValueError("Cannot select Qdrant server URL and local_path together")
        if not isinstance(collection, str) or not collection.startswith("lab11_"):
            raise ValueError("Only lab11_ prefixed collections are allowed")
        if reindex and not initialize:
            raise ValueError("reindex requires explicit collection initialization")

        self.models = models
        self.exact = ExactKNN(records, metric)
        self.records = {record.id: record for record in self.exact.records}
        self.metric = metric
        self.dimensions = self.exact.dimensions
        self.collection = collection
        self.model_id = self.exact.model_id
        self.remote = url is not None
        self.client = (
            QdrantClient(url=url, api_key=api_key, timeout=10)
            if self.remote
            else QdrantClient(":memory:")
            if str(local_path) == ":memory:"
            else QdrantClient(path=str(local_path))
        )

        try:
            expected_distance = {
                "cosine": models.Distance.COSINE,
                "dot": models.Distance.DOT,
                "l2": models.Distance.EUCLID,
            }[metric]
            exists = self.client.collection_exists(collection)
            if exists and reindex:
                # Destructive reset is deliberate, never triggered by a normal query.
                self.client.delete_collection(collection)
                exists = False
            if not exists:
                if not initialize:
                    raise ValueError(
                        f"Qdrant collection {collection!r} does not exist; "
                        "run --backend qdrant --index-only first"
                    )
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=self.dimensions, distance=expected_distance,
                    ),
                )
                self.upsert_many(records)
            else:
                info = self.client.get_collection(collection)
                params = info.config.params.vectors
                if (
                    not isinstance(params, models.VectorParams)
                    or params.size != self.dimensions
                    or params.distance != expected_distance
                ):
                    raise ValueError(
                        "Existing Qdrant collection has incompatible dimensions or metric; "
                        "reindex the dedicated lab collection"
                    )
        except Exception:
            self.close()
            raise

    def upsert_many(self, records: Sequence[VectorRecord]) -> None:
        points = []
        for record in records:
            if (
                record.embedding_model != self.model_id
                or len(record.vector) != self.dimensions
            ):
                raise ValueError("Embedding model/version or dimensions mismatch")
            metadata = normalize_metadata(record.metadata)
            points.append(
                self.models.PointStruct(
                    id=_point_id(record.id),
                    vector=list(validate_vector(record.vector)),
                    payload={
                        "chunk_id": record.id,
                        "text": record.text,
                        "embedding_model": record.embedding_model,
                        **metadata,
                    },
                )
            )
        if points:
            self.client.upsert(
                collection_name=self.collection, points=points, wait=True,
            )
            self.records.update({record.id: record for record in records})

    def delete(self, record_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=self.models.PointIdsList(
                points=[_point_id(record_id)]
            ),
            wait=True,
        )
        self.records.pop(record_id, None)

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
            raise ValueError("Query dimension mismatch")
        canonical = normalize_filters(filters)
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Trusted tenant scope required")
        conditions = [
            self.models.FieldCondition(
                key="tenant_id",
                match=self.models.MatchValue(value=tenant_id),
            )
        ]
        conditions.extend(
            self.models.FieldCondition(
                key=field, match=self.models.MatchValue(value=value),
            )
            for field, value in sorted(canonical.items())
        )
        result = self.client.query_points(
            collection_name=self.collection,
            query=list(query),
            query_filter=self.models.Filter(must=conditions),
            limit=top_k,
            with_payload=True,
        )
        hits = []
        for point in result.points:
            payload = point.payload or {}
            record_id = payload.get("chunk_id")
            record = self.records.get(record_id)
            if record is None:
                raise ValueError(
                    "Qdrant contains unknown records for this source snapshot; "
                    "reindex the dedicated lab collection"
                )
            # Protect against silently mixing old vectors, sources, model IDs,
            # or metadata during ingestion/query changes.
            expected = {"chunk_id": record.id, "text": record.text,
                        "embedding_model": record.embedding_model, **record.metadata}
            if any(payload.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "Qdrant payload differs from source records; "
                    "reindex the dedicated lab collection"
                )
            if not matches(record.metadata, tenant_id, canonical):
                raise ValueError("Qdrant returned a record outside trusted scope")
            score = similarity(query, record.vector, self.metric)
            if min_score is None or score >= min_score:
                hits.append(SearchHit(record, score, len(hits) + 1))
        return hits, SearchStats(0, 0, len(result.points))

    def close(self) -> None:
        self.client.close()


# Existing Lab 11 tests and local-mode exercises retain this import.
QdrantLocalIndex = QdrantIndex
