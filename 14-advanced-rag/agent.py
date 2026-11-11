"""Lab 14 — advanced RAG retrieval planning, offline by default.

The optional LLM provides SEARCH QUERY proposals only, not fabricated evidence
or user-facing answers. The deterministic answerer uses source excerpts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import Budgets, Request
from controller import AdvancedRAG
from evaluation import compare_strategies, evaluate_strategy
from planning import STRATEGIES

SAMPLE_QUESTIONS = {
    "finance": (
        "Compare FPT international software revenue in Q1 and Q2 2025, "
        "calculate growth and explain the main reasons for the change."
    ),
    "refund": "What are the refund deadline, conditions and exceptions?",
    "vllm": "How can vLLM Qwen3 reduce CUDA out of memory errors?",
    "auth": "Why does ERR_AUTH_403 occur in the administrator dashboard code?",
    "sql": "What is paid order revenue for customer 1?",
    "unknown": "What is the unlisted confidential CFO salary amount?",
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 14 — Advanced RAG")
    parser.add_argument("--scenario", choices=("normal", "delayed-drivers", "no-drivers"),
                        default="normal", help="Controlled corpus fault injection")
    parser.add_argument("--example", choices=tuple(SAMPLE_QUESTIONS), default="finance")
    parser.add_argument("--question", help="Override the synthetic example question")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="adaptive")
    parser.add_argument("--transform-provider", choices=("mock", "llm"), default="mock",
                        help="LLM mode makes explicit possibly billable rewrite calls")
    parser.add_argument("--tenant-id", default="tenant_a",
                        help="Synthetic trusted application scope")
    parser.add_argument("--as-of-date", default="2026-11-10")
    parser.add_argument("--sources", default="docs,code,sql",
                        help="Trusted allowlist simulation: docs,code,sql")
    parser.add_argument("--max-subqueries", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--max-search-calls", type=int, default=12)
    parser.add_argument("--max-hits-per-search", type=int, default=5)
    parser.add_argument("--max-evidence", type=int, default=6)
    parser.add_argument("--max-llm-calls", type=int, default=3)
    parser.add_argument("--benchmark", action="store_true",
                        help="Run independent gold-document evaluation")
    parser.add_argument("--compare", action="store_true",
                        help="Run all offline retrieval strategy variants (requires --benchmark)")
    parser.add_argument("--case", action="append",
                        help="Filter benchmark case IDs (repeatable)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON report target; not overwritten")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.compare and not args.benchmark:
        parser.error("--compare requires --benchmark")
    if args.benchmark and args.transform_provider != "mock":
        parser.error("Independent gold benchmark is offline mock only")
    if args.transform_provider == "llm" and args.strategy not in {
        "rewrite", "multiquery", "hyde"
    }:
        parser.error("LLM query proposals are only used by rewrite, multiquery and hyde")
    if args.case and not args.benchmark:
        parser.error("--case is for benchmarking only")
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    budgets = Budgets(
        max_subqueries=args.max_subqueries,
        max_rounds=args.max_rounds,
        max_search_calls=args.max_search_calls,
        max_hits_per_search=args.max_hits_per_search,
        max_evidence=args.max_evidence,
        max_llm_calls=args.max_llm_calls,
    )
    try:
        budgets.validate()
        if args.benchmark:
            report = (
                compare_strategies(budgets=budgets, cases_filter=tuple(args.case or ()))
                if args.compare
                else evaluate_strategy(
                    args.strategy, budgets=budgets,
                    cases_filter=tuple(args.case or ()),
                )
            )
        else:
            request = Request(
                question=args.question or SAMPLE_QUESTIONS[args.example],
                tenant_id=args.tenant_id, as_of_date=args.as_of_date,
                allowed_sources=sources,
            )
            report = AdvancedRAG(
                transformer_provider=args.transform_provider
            ).run(
                request, strategy=args.strategy, budgets=budgets,
                scenario=args.scenario,
            )
        if args.output is not None:
            if args.output.exists():
                raise ValueError("Refusing to overwrite an existing report")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        elif args.benchmark:
            rows = report["strategies"] if args.compare else [report]
            for r in rows:
                print(
                    f"Strategy {r['strategy']}: "
                    f"gold recall={r['mean_recall']} "
                    f"context coverage={r['mean_context_gold_coverage']} "
                    f"search_calls={r['total_search_calls']} "
                    f"partial/abstain={r['partial_or_abstained']}"
                )
        else:
            print(f"Strategy: {report['strategy']} | Intent: {report['analysis']['intent']}")
            print(f"Coverage (heuristic): {report['coverage_by_runtime_heuristics']:.1%}")
            print(
                f"Calls: {report['usage']['search_calls']} | "
                f"Rounds: {report['usage']['retrieval_rounds']} | "
                f"Stop: {report['stop_reason']}"
            )
            print("Evidence:", ", ".join(
                f"[{e['id']}] {e['doc_id']}" for e in report["evidence"]
            ) or "none")
            print("Missing aspects:", ", ".join(report["missing_aspects"]) or "none")
            print(f"Answer ({report['answer']['status']}): {report['answer']['answer']}")
            if args.show_trace:
                print("\nPLAN:")
                for step in report["plan"]:
                    print(
                        f"  {step['id']} {step['action']} {step['aspect']}: "
                        f"{step['query']} (source={step['source']}, "
                        f"depends_on={step['depends_on']})"
                    )
                print("\nEVENTS:")
                for event in report["events"]:
                    print(" ", json.dumps(event, ensure_ascii=False))
        if args.output:
            print("Report saved:", args.output)
        return 0
    except (ValueError, KeyError, TypeError, OSError, PermissionError,
            ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())