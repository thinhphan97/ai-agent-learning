"""Retrieval orchestration; trusted scope applies to EVERY retrieval branch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from analyzer import validate_query
from bm25 import BM25Index
from contracts import Chunk, Hit
from dense import DenseIndex, Embedder
from fusion import reciprocal_rank_fusion
from metadata import normalize_filters
from reranker import Reranker
from storage import SQLiteFTSIndex

METHODS = ("bm25", "sqlite", "dense", "hybrid")


class RetrievalPipeline:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        sqlite: SQLiteFTSIndex,
        *,
        embedder: Embedder | None = None,
    ):
        if not chunks:
            raise ValueError("Need indexed chunks")
        self.bm25 = BM25Index(chunks)
        self.sqlite = sqlite
        self.dense = DenseIndex(chunks, embedder) if embedder is not None else None
        ids = {chunk.id for chunk in chunks}
        if ids != {chunk.id for chunk in sqlite.all_chunks()}:
            raise ValueError("SQLite and in-memory indexes must have identical IDs")

    def search(
        self,
        question: str,
        *,
        method: str = "hybrid",
        tenant_id: str,
        filters: Mapping[str, str] | None = None,
        top_k: int = 5,
        candidate_k: int = 10,
        reranker: Reranker | None = None,
    ) -> list[Hit]:
        if method not in METHODS:
            raise ValueError(f"Unknown search method: {method}")
        validate_query(question, top_k)
        validate_query(question, candidate_k)
        if candidate_k < top_k:
            raise ValueError("candidate_k must be >= top_k")
        filters = normalize_filters(filters)
        kwargs = dict(tenant_id=tenant_id, filters=filters, top_k=candidate_k)

        if method == "bm25":
            candidates = self.bm25.search(question, **kwargs)
        elif method == "sqlite":
            candidates = self.sqlite.search(question, **kwargs)
        elif method == "dense":
            if self.dense is None:
                raise ValueError("Dense search requires an embedding provider")
            candidates = self.dense.search(question, **kwargs)
        else:
            if self.dense is None:
                raise ValueError("Hybrid search requires an embedding provider")
            lexical = self.sqlite.search(question, **kwargs)
            dense = self.dense.search(question, **kwargs)
            candidates = reciprocal_rank_fusion(
                [lexical, dense],
                top_k=candidate_k,
            )

        # Never relax filters for zero results, during fusion or during reranking.
        if reranker is not None:
            candidates = reranker.rerank(question, candidates)
        return candidates[:top_k]
