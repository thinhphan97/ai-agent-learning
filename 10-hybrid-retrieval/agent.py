"""Lab 10 CLI: BM25, persistent SQLite FTS5, dense retrieval, RRF and reranking.

Offline by default. --dense-provider api and --reranker cross-encoder are explicit opt-ins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyzer import tokenize
from dense import HashingEmbedder, OpenAIEmbedder
from evaluation import evaluate
from fixtures import CASES, load_corpus
from pipeline import METHODS, RetrievalPipeline
from reranker import LexicalOverlapReranker, LocalCrossEncoderReranker
from storage import SQLiteFTSIndex


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 10 — BM25 & Hybrid Retrieval")
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument(
        "--question",
        default="How much did FPT international software revenue grow in Q2 2025?",
    )
    parser.add_argument(
        "--tenant-id",
        default="tenant_a",
        help="Demo trusted application scope; NEVER accept directly from an untrusted client",
    )
    parser.add_argument("--company", help="Company metadata filter")
    parser.add_argument("--period", help="Period metadata filter (e.g. Q2/2025)")
    parser.add_argument("--segment", help="Business segment filter")
    parser.add_argument("--document-type", help="Document type filter")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument(
        "--db", default=":memory:", help="SQLite FTS5 file path or :memory:"
    )
    parser.add_argument(
        "--reindex", action="store_true", help="Re-upsert synthetic fixtures"
    )
    parser.add_argument(
        "--index-only", action="store_true", help="Create/update SQLite index only"
    )
    parser.add_argument(
        "--dense-provider",
        choices=("toy", "api"),
        default="toy",
        help="toy = hashed lexical vectors; api = real embedding model from root .env",
    )
    parser.add_argument(
        "--reranker",
        choices=("none", "lexical", "cross-encoder"),
        default="none",
    )
    parser.add_argument(
        "--cross-encoder-path",
        type=Path,
        help="Required existing LOCAL Sentence Transformers Cross-Encoder directory",
    )
    parser.add_argument(
        "--explain", action="store_true", help="Print analyzer terms and BM25 DF/IDF"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Offline labeled retrieval comparison"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable results"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.candidate_k < args.top_k or not (1 <= args.top_k <= 100):
        parser.error("Require 1 <= top-k <= candidate-k <= 100")
    if not 1 <= args.candidate_k <= 100:
        parser.error("candidate-k must be <= 100")
    if args.cross_encoder_path and args.reranker != "cross-encoder":
        parser.error("--cross-encoder-path requires --reranker cross-encoder")
    if args.reranker == "cross-encoder" and not args.cross_encoder_path:
        parser.error("Specify --cross-encoder-path for a local model")
    if args.benchmark and (args.dense_provider != "toy" or args.reranker != "none"):
        parser.error(
            "Benchmark is offline-only with no reranker; compare modes explicitly"
        )
    try:
        with SQLiteFTSIndex(args.db) as store:
            if store.count() == 0 or args.reindex:
                store.upsert_many(load_corpus())
            if args.index_only:
                if args.json:
                    print(
                        json.dumps(
                            {"sqlite_db": args.db, "indexed_chunks": store.count()}
                        )
                    )
                else:
                    print(f"Indexed {store.count()} chunks in SQLite FTS5: {args.db}")
                return 0

            requires_dense = args.benchmark or args.method in {"dense", "hybrid"}
            embedder = None
            if requires_dense:
                embedder = (
                    OpenAIEmbedder()
                    if args.dense_provider == "api"
                    else HashingEmbedder()
                )
            pipeline = RetrievalPipeline(store.all_chunks(), store, embedder=embedder)
            if args.benchmark:
                reports = [
                    evaluate(pipeline, CASES, method=m, k=args.top_k) for m in METHODS
                ]
                reports.append(
                    evaluate(
                        pipeline,
                        CASES,
                        method="hybrid",
                        k=args.top_k,
                        reranker=LexicalOverlapReranker(),
                    )
                )
                if args.json:
                    print(json.dumps(reports, indent=2))
                else:
                    print(
                        "Fixed synthetic labels; hashed lexical vectors, NOT neural semantics."
                    )
                    print(
                        f"Method                  Recall@{args.top_k}   MRR@{args.top_k}   nDCG@{args.top_k}"
                    )
                    for report in reports:
                        print(
                            f"{report['method']:<24} "
                            f"{report['recall_at_k']:.3f}       "
                            f"{report['mrr_at_k']:.3f}     "
                            f"{report['ndcg_at_k']:.3f}"
                        )
                return 0

            filters = {
                key: value
                for key, value in {
                    "company_id": args.company,
                    "period": args.period,
                    "segment": args.segment,
                    "document_type": args.document_type,
                }.items()
                if value is not None
            }
            reranker = (
                None
                if args.reranker == "none"
                else (
                    LexicalOverlapReranker()
                    if args.reranker == "lexical"
                    else LocalCrossEncoderReranker(args.cross_encoder_path)
                )
            )
            hits = pipeline.search(
                args.question,
                method=args.method,
                tenant_id=args.tenant_id,
                filters=filters,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                reranker=reranker,
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "query": args.question,
                            "method": args.method,
                            "tenant_id": args.tenant_id,
                            "filters": filters,
                            "results": [
                                {
                                    "rank": rank,
                                    "id": hit.chunk.id,
                                    "score": hit.score,
                                    "method": hit.method,
                                    "sources": hit.sources,
                                    "original_text": hit.chunk.text,
                                    "metadata": dict(hit.chunk.metadata),
                                }
                                for rank, hit in enumerate(hits, start=1)
                            ],
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print(
                    f"Method={args.method} | dense={args.dense_provider} | reranker={args.reranker}"
                )
                print(f"Question: {args.question}")
                print(
                    f"Scope: {args.tenant_id} | Filters: {filters} | indexed={store.count()}"
                )
                if args.explain:
                    print("Analyzed query terms and document frequencies:")
                    for term in dict.fromkeys(tokenize(args.question)):
                        df = len(pipeline.bm25.postings.get(term, []))
                        idf = pipeline.bm25.idf.get(term, 0.0)
                        print(f"  {term:<24} df={df:<3} idf={idf:.4f}")
                for rank, hit in enumerate(hits, start=1):
                    print(
                        f"\n{rank}. [{hit.chunk.id}] {hit.method} score={hit.score:.6f}"
                    )
                    print(f"   metadata={dict(hit.chunk.metadata)}")
                    if hit.sources:
                        print(f"   fused_sources={','.join(hit.sources)}")
                    print(f"   original={hit.chunk.text}")
                if not hits:
                    print("No matching chunks inside the enforced tenant/filter scope.")
            return 0
    except (ValueError, ImportError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
