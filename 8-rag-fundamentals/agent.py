"""Lab 8 — Basic RAG entry point (the actual logic lives in separate modules).

From the workspace root:
  .venv/bin/python 8-rag-fundamentals/agent.py
  .venv/bin/python 8-rag-fundamentals/agent.py --eval
  .venv/bin/python 8-rag-fundamentals/agent.py --live
  .venv/bin/python 8-rag-fundamentals/agent.py --retriever dense --live
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from documents import chunk_documents, load_documents
from evaluation import evaluate
from generation import ExtractiveMockGenerator, OpenAIGenerator
from pipeline import RAGPipeline
from retrieval import DenseRetriever, TfidfRetriever, openai_embedding_client

DEFAULT_DOCUMENTS = Path(__file__).resolve().parent / "data" / "documents"
DEFAULT_QUESTION = "How many days can I return unopened hardware?"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 8: RAG Fundamentals")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock", action="store_true", help="Force deterministic offline generator"
    )
    mode.add_argument(
        "--live", action="store_true", help="Use live LLM from shared root .env"
    )
    parser.add_argument(
        "--retriever",
        choices=("tfidf", "dense"),
        default="tfidf",
        help="TF-IDF lexical baseline (offline), or real provider embeddings",
    )
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate on 8 small fixture questions"
    )
    args = parser.parse_args()

    try:
        documents = load_documents(args.documents)
        chunks = chunk_documents(
            documents, chunk_size=args.chunk_size, overlap=args.overlap
        )
        retriever = (
            DenseRetriever(chunks, openai_embedding_client())
            if args.retriever == "dense"
            else TfidfRetriever(chunks)
        )
        use_live = args.live or (not args.mock and os.getenv("LIVE_MODE") == "1")
        generator = OpenAIGenerator() if use_live else ExtractiveMockGenerator()
        pipeline = RAGPipeline(
            retriever,
            generator,
            top_k=args.top_k,
            min_score=args.min_score,
        )
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))

    print(
        f"Documents: {len(documents)} | Chunks: {len(chunks)} | "
        f"Retriever: {args.retriever} | Generator: {'live' if use_live else 'extractive-mock'}"
    )

    if args.eval:
        summary, outcomes = evaluate(pipeline.answer)
        for item in outcomes:
            print(
                f"{'PASS' if item['answer_check'] else 'FAIL'} "
                f"| source_retrieved={item['retrieved']} "
                f"| {item['question']}"
            )
        print(f"Source Recall@{args.top_k}: {summary.source_recall_at_k:.1%}")
        print(f"Answer checks: {summary.grounded_answer_accuracy:.1%}")
        print(f"Negative abstention: {summary.negative_abstention_accuracy:.1%}")
        return 0

    try:
        result = pipeline.answer(args.question)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Question: {args.question}")
    for hit in result.hits:
        print(
            f"Retrieved: {hit.chunk.id} | score={hit.score:.3f} | "
            f"title={hit.chunk.title} | words={hit.chunk.word_start}:{hit.chunk.word_end}"
        )
    print(f"Status: {result.status} | Reason: {result.reason}")
    print(f"Answer: {result.text}")
    return 1 if result.reason == "GENERATION_FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
