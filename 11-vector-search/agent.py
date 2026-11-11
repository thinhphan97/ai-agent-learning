"""Lab 11 — vector similarity, exact KNN, educational HNSW and optional engines.

No remote API calls or third-party dependencies in default mode.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from contracts import VectorRecord
from embeddings import HashingEmbedder, OpenAIEmbedder
from evaluation import benchmark
from exact import ExactKNN
from fixtures import load_documents
from hnsw import EducationalHNSW
from hnswlib_adapter import HnswlibIndex
from numpy_adapter import NumPyExactKNN
from metrics import METRICS, score_to_distance, similarity, validate_vector
from metadata import normalize_filters
from qdrant_adapter import QdrantIndex
from storage import VectorStore


def parse_vector(text: str) -> tuple[float, ...]:
    try:
        return validate_vector([float(x.strip()) for x in text.split(",")])
    except (ValueError, OverflowError) as exc:
        raise ValueError("Expected numeric --query-vector '0.1,0.2,...'") from exc


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 11 — Vector Search Internals")
    parser.add_argument(
        "--backend",
        choices=("exact", "numpy", "educational-hnsw", "hnswlib", "qdrant", "qdrant-local"),
        default="educational-hnsw",
    )
    parser.add_argument("--metric", choices=METRICS, default="cosine")
    parser.add_argument(
        "--embedding",
        choices=("toy", "api"),
        default="toy",
        help="toy = deterministic lexical hash; api = actual embedding provider",
    )
    parser.add_argument(
        "--question",
        default="How can I reduce CUDA out of memory when running vLLM?",
    )
    parser.add_argument(
        "--query-vector", help="Numeric vector; must match indexed dimensions"
    )
    parser.add_argument(
        "--tenant-id", default="tenant_a", help="Trusted-scope SIMULATION only"
    )
    parser.add_argument("--company", help="Optional metadata filter")
    parser.add_argument("--period", help="Optional metadata filter")
    parser.add_argument("--section", help="Optional metadata filter")
    parser.add_argument("--db", default=":memory:", help="SQLite source-record path")
    parser.add_argument(
        "--qdrant-path", default=":memory:", help="Qdrant local SDK path; only for qdrant-local"
    )
    parser.add_argument("--qdrant-url", help="Qdrant server URL; defaults to root QDRANT_URL or localhost:6333")
    parser.add_argument("--qdrant-collection", default="lab11_documents", help="Dedicated lab11_ collection")
    parser.add_argument(
        "--reindex", action="store_true", help="Re-embed/upsert source fixtures"
    )
    parser.add_argument(
        "--index-only", action="store_true", help="Persist source vectors; for Qdrant also upsert collection"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ef-search", type=int, default=32)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--ef-construction", type=int, default=32)
    parser.add_argument(
        "--filter-mode",
        choices=("post", "exact-filter"),
        default="post",
        help="Post-filter ANN Top-K or explicit exact filtered fallback",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum similarity; L2 uses NEGATIVE squared distance",
    )
    parser.add_argument(
        "--geometry", action="store_true", help="Show metrics on 2D example"
    )
    parser.add_argument(
        "--crud-demo",
        action="store_true",
        help="Insert/update/delete in an ephemeral DB",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Compare ANN with exact KNN"
    )
    parser.add_argument("--benchmark-size", type=int, default=240)
    parser.add_argument("--benchmark-dim", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    return parser


def _geometry() -> dict[str, object]:
    q = (1.0, 0.0)
    points = {"A": (0.8, 0.6), "B": (0.9, 0.1), "C": (2.0, 0.0)}
    return {
        metric: [
            {
                "id": label,
                "score": round(similarity(q, vector, metric), 5),
                "distance": (
                    round(score_to_distance(similarity(q, vector, metric), "l2"), 5)
                    if metric == "l2"
                    else None
                ),
            }
            for label, vector in sorted(
                points.items(),
                key=lambda p: (-similarity(q, p[1], metric), p[0]),
            )
        ]
        for metric in METRICS
    }


def _crud_demo(embedder: HashingEmbedder | OpenAIEmbedder) -> dict[str, object]:
    records = load_documents(embedder)
    first = records[0]
    with VectorStore() as store:
        store.upsert_many(records)
        before = store.count()
        updated_text = (
            "UPDATED: FPT international software revenue increased 17 percent."
        )
        vector = tuple(embedder.embed([updated_text])[0])
        changed = VectorRecord(
            first.id, vector, updated_text, first.metadata, first.embedding_model
        )
        store.upsert_many([changed])
        after_update = store.get(first.id)
        assert after_update is not None
        store.delete(first.id)
        return {
            "before_count": before,
            "updated_text": after_update.text,
            "updated_vector": after_update.vector == vector,
            "after_delete_count": store.count(),
            "deleted_record_absent": store.get(first.id) is None,
            "persistent_file_changed": False,
        }


def make_qdrant(
    args: argparse.Namespace,
    records: list[VectorRecord],
    *,
    initialize: bool,
) -> QdrantIndex:
    """Open Qdrant server or in-process local client, without implicit fallback."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    url = None
    if args.backend == "qdrant":
        url = args.qdrant_url or os.getenv("QDRANT_URL") or "http://127.0.0.1:6333"
    api_key = (os.getenv("QDRANT_API_KEY") or None) if url else None
    try:
        return QdrantIndex(
            records,
            metric=args.metric,
            local_path=args.qdrant_path,
            url=url,
            api_key=api_key,
            collection=args.qdrant_collection,
            reindex=args.reindex,
            initialize=initialize,
        )
    except (ValueError, ImportError):
        raise
    except Exception as exc:
        raise RuntimeError(
            "Qdrant service could not be reached. Start it with "
            "'docker compose -f 11-vector-search/compose.yaml up -d' "
            "and verify QDRANT_URL or --qdrant-url."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.backend == "qdrant" and args.reindex and not args.index_only:
        parser.error("--reindex on a Qdrant server requires --index-only to confirm a destructive reset")
    if args.backend not in {"qdrant", "qdrant-local"} and (args.qdrant_url or args.qdrant_path != ":memory:"):
        parser.error("Qdrant connection options require --backend qdrant or qdrant-local")
    if args.geometry:
        result = _geometry()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for metric, values in result.items():
                print(
                    f"{metric:>8}: "
                    + " > ".join(
                        f"{value['id']} ({value['score']:.4f})" for value in values
                    )
                )
        return 0
    if args.benchmark:
        if args.embedding != "toy" or args.query_vector or args.min_score is not None:
            parser.error(
                "Benchmark uses independent synthetic numerical vectors, no API"
            )
        if args.backend not in ("educational-hnsw", "hnswlib"):
            parser.error("Benchmark needs educational-hnsw or hnswlib backend")
        try:
            report = benchmark(
                count=args.benchmark_size,
                dim=args.benchmark_dim,
                k=args.top_k,
                backend=(
                    "educational" if args.backend == "educational-hnsw" else "hnswlib"
                ),
                ef_values=tuple(
                    sorted(
                        {max(args.top_k, 8), max(args.top_k, 24), max(args.top_k, 64)}
                    )
                ),
            )
        except (ValueError, ImportError) as exc:
            parser.error(str(exc))
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("ANN Recall@K versus exact KNN, NOT semantic RAG relevance")
            print(
                f"Backend={report['backend']} vectors={report['vectors']} "
                f"dim={report['dimensions']} k={report['k']}"
            )
            print(f"Exact median latency: {report['exact_median_latency_ms']} ms")
            print(f"ANN build: {report['ann_build_ms']} ms")
            print("ef_search  ANN Recall@K  Median latency ms  Distance evaluations")
            for item in report["ef_sweep"]:
                print(
                    f"{item['ef_search']:>9}  {item['ann_recall_at_k']:>12.3f}  "
                    f"{item['median_latency_ms']:>17.3f}  "
                    f"{item['mean_distance_evaluations']:>20.1f}"
                )
            if report["levels"] is not None:
                print(f"Educational graph levels: {report['levels']}")
        return 0

    try:
        embedder = OpenAIEmbedder() if args.embedding == "api" else HashingEmbedder()
        if args.crud_demo:
            output = _crud_demo(embedder)
            print(json.dumps(output, indent=2) if args.json else output)
            return 0

        if args.backend == "qdrant" and args.qdrant_path != ":memory:":
            raise ValueError("--qdrant-path only applies to --backend qdrant-local")
        if args.backend == "qdrant-local" and args.qdrant_url:
            raise ValueError("--qdrant-url only applies to --backend qdrant")
        with VectorStore(args.db) as store:
            if store.count() == 0 or args.reindex:
                store.upsert_many(load_documents(embedder))
            records = store.all_records()
        if args.index_only:
            if args.backend in {"qdrant", "qdrant-local"}:
                index = make_qdrant(args, records, initialize=True)
                try:
                    index.upsert_many(records)
                finally:
                    index.close()
                print(json.dumps({
                    "backend": args.backend,
                    "collection": args.qdrant_collection,
                    "count": len(records), "embedding_model": embedder.model_id,
                }))
            else:
                print(json.dumps({
                    "index": args.db, "count": len(records),
                    "embedding_model": embedder.model_id,
                }))
            return 0

        if {record.embedding_model for record in records} != {embedder.model_id}:
            raise ValueError(
                "Stored model differs from selected query embedder; use --reindex"
            )
        if args.query_vector is None:
            query = tuple(embedder.embed([args.question])[0])
        else:
            query = parse_vector(args.query_vector)
        if args.ef_search < args.top_k:
            raise ValueError("ef-search must be >= top-k")
        filters = normalize_filters({
            field: value
            for field, value in {
                "company_id": args.company,
                "period": args.period,
                "section": args.section,
            }.items()
            if value is not None
        })
        if args.backend == "exact":
            index = ExactKNN(records, args.metric)
            hits, stats = index.search(
                query,
                tenant_id=args.tenant_id,
                filters=filters,
                top_k=args.top_k,
                min_score=args.min_score,
            )
        elif args.backend == "numpy":
            index = NumPyExactKNN(records, args.metric)
            hits, stats = index.search(
                query, tenant_id=args.tenant_id,
                filters=filters, top_k=args.top_k,
                min_score=args.min_score,
            )
        elif args.backend == "educational-hnsw":
            index = EducationalHNSW(
                records,
                metric=args.metric,
                m=args.m,
                ef_construction=args.ef_construction,
            )
            hits, stats = index.search(
                query,
                tenant_id=args.tenant_id,
                filters=filters,
                top_k=args.top_k,
                ef_search=args.ef_search,
                filter_mode=args.filter_mode,
                min_score=args.min_score,
            )
        elif args.backend == "hnswlib":
            index = HnswlibIndex(
                records,
                metric=args.metric,
                m=args.m,
                ef_construction=args.ef_construction,
            )
            hits, stats = index.search(
                query,
                tenant_id=args.tenant_id,
                filters=filters,
                top_k=args.top_k,
                ef_search=args.ef_search,
                filter_mode=args.filter_mode,
                min_score=args.min_score,
            )
        else:
            index = make_qdrant(
                args, records, initialize=(args.backend == "qdrant-local"),
            )
            try:
                hits, stats = index.search(
                    query,
                    tenant_id=args.tenant_id,
                    filters=filters,
                    top_k=args.top_k,
                    min_score=args.min_score,
                )
            finally:
                index.close()

        response = {
            "backend": args.backend,
            "metric": args.metric,
            "embedding_model": embedder.model_id,
            "indexed": len(records),
            "filter_mode": (
                "qdrant-payload" if args.backend in {"qdrant", "qdrant-local"}
                else args.filter_mode
            ),
            "filters": filters,
            "scope": args.tenant_id,
            "stats": asdict(stats),
            "results": [
                {
                    "id": hit.record.id,
                    "rank": hit.rank,
                    "score": round(hit.score, 6),
                    "text": hit.record.text,
                    "metadata": dict(hit.record.metadata),
                }
                for hit in hits
            ],
        }
        if args.json:
            print(json.dumps(response, indent=2))
        else:
            print(
                f"backend={args.backend} metric={args.metric} indexed={len(records)} "
                f"model={embedder.model_id}"
            )
            print(f"scope={args.tenant_id} filters={filters} mode={args.filter_mode}")
            print(
                f"distance_evaluations={stats.distance_evaluations} "
                f"candidates={stats.candidates_considered}"
            )
            for hit in hits:
                print(
                    f"  {hit.rank}. [{hit.record.id}] score={hit.score:.5f} "
                    f"source={hit.record.metadata['source']}"
                )
                print(f"     {hit.record.text}")
            if not hits:
                print("No matching results under the current scope and filter.")
        return 0
    except (ValueError, ImportError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())