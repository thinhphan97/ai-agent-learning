"""Lab 12: contextual indexing, evidence retrieval, reranking and context packing.

Default mode is deterministic/offline. Real LLM contextualization is explicitly
opt-in; local real Cross-Encoder needs existing model weights on disk.
"""
from __future__ import annotations

import argparse
import json

from evaluation import evaluate
from pipeline import RAGPipeline
from validation import check_citations

SCENARIOS = {
    "refund": (
        "What are the refund deadline, eligibility conditions, and exceptions for store purchases?",
        ("deadline", "conditions", "exceptions"),
    ),
    "vllm": (
        "How can I mitigate CUDA out of memory in vLLM Qwen3 inference?",
        ("mitigation",),
    ),
    "auth": (
        "Why does ERR_AUTH_403 occur in the administrator dashboard?",
        ("diagnosis",),
    ),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lab 12 — Reranking & Context Construction")
    p.add_argument("--scenario", choices=SCENARIOS, default="refund")
    p.add_argument("--question", help="Override scenario question; supply --aspects separately")
    p.add_argument("--aspects", default=None, help="Comma-separated required evidence aspects")
    p.add_argument("--tenant-id", default="tenant_a", help="Trusted tenant simulation only")
    p.add_argument("--contextualization", choices=("none", "metadata", "llm"), default="metadata")
    p.add_argument("--retrieval", choices=("bm25", "dense", "hybrid"), default="hybrid")
    p.add_argument("--dense-provider", choices=("toy", "api"), default="toy",
                   help="toy is lexical hash; api makes real external embeddings calls")
    p.add_argument("--retrieval-k", type=int, default=50)
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--candidate-selection", choices=("topk", "mmr", "coverage"), default="topk")
    p.add_argument("--reranker", choices=("toy", "cross-encoder"), default="toy")
    p.add_argument("--model-path", help="Existing local directory for real Cross-Encoder")
    p.add_argument("--context-selection", choices=("ranked", "coverage"), default="coverage")
    p.add_argument("--expansion", choices=("child", "parent"), default="child")
    p.add_argument("--max-evidence", type=int, default=5)
    p.add_argument("--token-budget", type=int, default=240)
    p.add_argument("--tokenizer", choices=("whitespace", "tiktoken"), default="whitespace")
    p.add_argument("--benchmark", action="store_true", help="Fixed-label offline evaluation")
    p.add_argument("--compare", action="store_true", help="Compare contextualization off/on")
    p.add_argument("--answer", help="Check explicit [E1] evidence IDs in a supplied answer")
    p.add_argument("--show-context", action="store_true", help="Print packed evidence context")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.dense_provider == "api" and args.retrieval == "bm25":
        p.error("API dense embeddings require --retrieval dense or hybrid")
    if args.benchmark and args.dense_provider != "toy":
        p.error("Fixed-label benchmark only uses offline toy vectors")
    if args.compare and not args.benchmark:
        p.error("--compare requires --benchmark")
    if args.benchmark and (args.reranker != "toy" or args.contextualization == "llm"):
        p.error("Offline benchmark requires --reranker toy and no --contextualization llm")
    try:
        if args.benchmark:
            scenarios = (
                ("none", "metadata") if args.compare
                else (args.contextualization,)
            )
            outputs = [
                evaluate(
                    contextualization=mode,
                    candidate_strategy=args.candidate_selection,
                    context_strategy=args.context_selection,
                    retrieval_method=args.retrieval,
                    retrieval_k=args.retrieval_k,
                    candidate_k=args.candidate_k,
                    max_evidence=args.max_evidence,
                    token_budget=args.token_budget,
                ) for mode in scenarios
            ]
            if args.json:
                print(json.dumps(outputs if args.compare else outputs[0], indent=2))
            else:
                for report in outputs:
                    print(
                        f"contextualization={report['contextualization']} "
                        f"retrieval_coverage={report['mean_retrieval_gold_coverage']} "
                        f"candidate_coverage={report['mean_candidate_gold_coverage']} "
                        f"context_coverage={report['mean_context_gold_coverage']}"
                    )
                    for case in report["cases"]:
                        print(f"  {case['case']}: {case['context_gold_coverage']}")
            return 0

        question, scenario_aspects = SCENARIOS[args.scenario]
        question = args.question or question
        if args.aspects is None:
            aspects = scenario_aspects if args.question is None else ()
        else:
            aspects = tuple(
                x.strip().casefold() for x in args.aspects.split(",") if x.strip()
            )
        pipeline = RAGPipeline(
            contextualization=args.contextualization,
            reranker=args.reranker, model_path=args.model_path,
            tokenizer=args.tokenizer,
            dense_provider=args.dense_provider,
            tenant_id=args.tenant_id if args.contextualization == "llm" else None,
        )
        result = pipeline.run(
            question,
            tenant_id=args.tenant_id,
            retrieval_k=args.retrieval_k,
            candidate_k=args.candidate_k,
            retrieval_method=args.retrieval,
            candidate_strategy=args.candidate_selection,
            context_strategy=args.context_selection,
            max_evidence=args.max_evidence,
            token_budget=args.token_budget,
            expansion=args.expansion,
            required_aspects=aspects,
        )
        if args.answer is not None:
            from contracts import ContextResult, Evidence

            raw_context = result["context"]
            context = ContextResult(
                text=raw_context["text"],
                evidence=tuple(Evidence(**e) for e in raw_context["evidence"]),
                tokens_used=raw_context["tokens_used"],
                token_budget=raw_context["token_budget"],
                dropped=tuple(tuple(d) for d in raw_context["dropped"]),
                missing_aspects=tuple(raw_context["missing_aspects"]),
                token_counter=raw_context["token_counter"],
            )
            result["citation_validation"] = check_citations(args.answer, context)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Query: {result['question']}")
            print(
                f"Retrieval: {len(result['retrieved'])} hits; "
                f"Candidate Selection: {len(result['candidates'])}; "
                f"Reranking: {len(result['reranked'])}"
            )
            print(
                f"Evidence: {len(result['context']['evidence'])} blocks; "
                f"budget: {result['context']['tokens_used']}/"
                f"{result['context']['token_budget']} "
                f"({result['context']['token_counter']})"
            )
            print(f"Metadata aspect coverage: {result['evidence_coverage_by_curated_metadata']}")
            print(f"Missing aspects: {result['context']['missing_aspects']}")
            for e in result["context"]["evidence"]:
                print(
                    f"  [{e['id']}] {e['aspect']} ({e['source']}#{e['section']}) "
                    f"tokens={e['tokens']}"
                )
                print(f"    {e['content'][:160]}")
            if args.show_context:
                print("\n--- CONTEXT (untrusted source text) ---")
                print(result["context"]["text"])
            if args.answer is not None:
                print("Citation ID validation:", result["citation_validation"])
            if result["context"]["dropped"]:
                print(f"Dropped: {result['context']['dropped']}")
        return 0
    except (ValueError, ImportError, OSError, RuntimeError, PermissionError) as exc:
        p.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())