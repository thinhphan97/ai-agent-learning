"""ANN benchmark: exact neighbors as ground truth, NOT RAG relevance labels."""

from __future__ import annotations

from statistics import median
from time import perf_counter

from exact import ExactKNN
from fixtures import synthetic_vectors
from hnsw import EducationalHNSW
from hnswlib_adapter import HnswlibIndex

DEFAULT_EF_VALUES = (8, 24, 64)


def ann_recall_at_k(exact_ids: list[str], approximate_ids: list[str]) -> float:
    if not exact_ids or len(set(exact_ids)) != len(exact_ids):
        raise ValueError("Exact IDs must contain unique neighbors")
    return len(set(exact_ids).intersection(approximate_ids)) / len(exact_ids)


def benchmark(
    *,
    count: int = 240,
    dim: int = 12,
    k: int = 5,
    backend: str = "educational",
    ef_values: tuple[int, ...] = DEFAULT_EF_VALUES,
) -> dict[str, object]:
    from metrics import validate_search

    validate_search(k)
    if backend not in ("educational", "hnswlib"):
        raise ValueError("Benchmark supports educational HNSW or hnswlib")
    if any(type(ef) is not int or ef < k for ef in ef_values):
        raise ValueError("Each ef must be >= k")

    records, queries = synthetic_vectors(count=count, dimensions=dim)
    start = perf_counter()
    exact = ExactKNN(records, metric="cosine")
    exact_build_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    ann = (
        EducationalHNSW(records, metric="cosine", m=8, ef_construction=32)
        if backend == "educational"
        else HnswlibIndex(records, metric="cosine", m=8, ef_construction=32)
    )
    ann_build_ms = (perf_counter() - start) * 1000
    expected = []
    exact_times = []
    for vector in queries:
        t = perf_counter()
        hits, _ = exact.search(vector, tenant_id="tenant_a", top_k=k)
        exact_times.append((perf_counter() - t) * 1000)
        expected.append([hit.record.id for hit in hits])
    reports = []
    for ef in ef_values:
        recall, latency, comparisons = [], [], []
        for vector, gold in zip(queries, expected):
            t = perf_counter()
            hits, stats = ann.search(
                vector,
                tenant_id="tenant_a",
                top_k=k,
                ef_search=ef,
                filter_mode="post",
            )
            latency.append((perf_counter() - t) * 1000)
            recall.append(ann_recall_at_k(gold, [hit.record.id for hit in hits]))
            comparisons.append(stats.distance_evaluations)
        reports.append(
            {
                "ef_search": ef,
                "ann_recall_at_k": round(sum(recall) / len(recall), 4),
                "median_latency_ms": round(median(latency), 3),
                "mean_distance_evaluations": round(
                    sum(comparisons) / len(comparisons), 2
                ),
            }
        )
    return {
        "benchmark_type": "ann_vs_exact_not_rag_quality",
        "backend": backend,
        "vectors": len(records),
        "dimensions": dim,
        "queries": len(queries),
        "k": k,
        "exact_build_ms": round(exact_build_ms, 3),
        "ann_build_ms": round(ann_build_ms, 3),
        "exact_median_latency_ms": round(median(exact_times), 3),
        "levels": ann.level_counts() if backend == "educational" else None,
        "ef_sweep": reports,
    }
