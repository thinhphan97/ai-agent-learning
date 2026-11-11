"""Lab 13 CLI — deterministic RAG evaluation and opt-in real model judging.

All default cases are synthetic. Quality gate failures exit 1, input errors 2.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dataset import load_suite
from harness import run_evaluation
from reporting import compare_reports, prepare_report, save_report
from runtime_stub import load_external_traces, load_scripted_traces, validate_trace

LAB_DIR = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lab 13 — RAG Evaluation")
    p.add_argument(
        "--scenario",
        default="baseline",
        choices=(
            "baseline",
            "candidate-loss",
            "hallucination",
            "stale-evidence",
            "tenant-leak",
            "injection-followed",
            "abstention-failure",
        ),
    )
    p.add_argument(
        "--case", action="append", help="Evaluate only this case ID (repeatable)"
    )
    p.add_argument("--k", type=int, default=5, help="IR metric cutoff K (default 5)")
    p.add_argument(
        "--traces-file",
        type=Path,
        help="Import a JSON list of external RAG traces; no Lab 12 imports",
    )
    p.add_argument(
        "--judge",
        choices=("labels", "none", "llm"),
        default="labels",
        help="labels: independent fixture annotations, none: IR only, llm: API",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Real model GENERATION over scripted retrieval, one case only",
    )
    p.add_argument(
        "--annotations-file", type=Path,
        help="Independent hand-labeled judgments for imported/live answers",
    )
    p.add_argument("--gates-file", type=Path, default=LAB_DIR / "data" / "gates.json")
    p.add_argument("--output-root", type=Path, default=LAB_DIR / "reports")
    p.add_argument(
        "--run-id", help="Safe report folder name; auto-generated if omitted"
    )
    p.add_argument("--no-save", action="store_true", help="Do not write reports")
    p.add_argument("--compare-with", type=Path, help="Prior summary.json")
    p.add_argument("--json", action="store_true", help="Print full JSON report")
    p.add_argument("--explain", action="store_true", help="Show stage diagnostics")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.traces_file and args.scenario != "baseline":
        p.error("--traces-file cannot be combined with a scripted failure scenario")
    if args.live and (args.traces_file or args.scenario != "baseline"):
        p.error("--live uses the baseline scripted retrieval/context stages only")
    if args.live and (not args.case or len(args.case) != 1):
        p.error("--live requires exactly one --case (bound real provider cost)")
    if args.judge == "llm" and (not args.case or len(args.case) != 1):
        p.error("--judge llm requires exactly one --case (bound model calls)")
    if (args.live or args.traces_file) and args.judge == "labels" and not args.annotations_file:
        p.error("Live/external answers require --annotations-file or --judge none/llm")
    if args.annotations_file and args.judge != "labels":
        p.error("--annotations-file requires --judge labels")
    try:
        version, docs, cases = load_suite()
        selected = args.case if args.case is not None else list(cases)
        if any(cid not in cases for cid in selected):
            raise ValueError(
                "Unknown case: "
                + ", ".join(cid for cid in selected if cid not in cases)
            )
        traces = (
            load_external_traces(args.traces_file)
            if args.traces_file
            else load_scripted_traces(args.scenario)
        )
        gateway = None
        if args.live or args.judge == "llm":
            from llm_adapter import LLMGateway

            gateway = LLMGateway()
        if args.live:
            cid = selected[0]
            case, trace = cases[cid], traces[cid]
            validate_trace(trace, case.id, case.tenant_id, docs)
            traces[cid] = gateway.generate(
                case.question,
                tenant_id=case.tenant_id,
                as_of_date=case.as_of_date,
                trace=trace,
                docs=docs,
            )
        report = prepare_report(
            run_evaluation(
                traces,
                case_ids=selected,
                scenario=(
                    "imported"
                    if args.traces_file
                    else ("live-generation" if args.live else args.scenario)
                ),
                k=args.k,
                answer_mode=args.judge,
                gates_path=args.gates_file,
                llm_gateway=gateway if args.judge == "llm" else None,
                annotations_path=args.annotations_file,
            )
        )
        if args.compare_with:
            prior = json.loads(args.compare_with.read_text(encoding="utf-8"))
            report["comparison"] = compare_reports(report, prior)
        path = None
        if not args.no_save:
            run_id = args.run_id or (
                f"{args.scenario}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}"
            )
            path = save_report(report, args.output_root, run_id)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            metrics = report["aggregates"]
            print(f"Lab 13 | {report['scenario']} | {len(selected)} cases | K={args.k}")
            print(
                f"Recall={metrics['mean_retrieval_recall']} "
                f"Candidate coverage={metrics['mean_candidate_gold_coverage']} "
                f"Context coverage={metrics['mean_context_gold_coverage']} "
                f"Faithfulness={metrics['mean_faithfulness']} "
                f"Completeness={metrics['mean_completeness']}"
            )
            print(
                f"Quality Gates: {'PASS' if report['quality_gates']['passed'] else 'FAIL'}"
            )
            print("Gate scope:", report["quality_gates"]["scope"])
            if report["quality_gates"]["skipped"]:
                print("Skipped answer gates:", len(report["quality_gates"]["skipped"]))
            for failure in report["quality_gates"]["failures"]:
                print("  FAIL:", failure)
            if args.explain:
                for row in report["cases"]:
                    print(
                        f"  {row['case_id']}: {', '.join(row['diagnosis']) or 'no detected failure'}"
                    )
            if path:
                print("Saved:", path)
            if "comparison" in report:
                print("Baseline:", report["comparison"]["baseline_scenario"])
                for key in (
                    "mean_context_gold_coverage",
                    "mean_faithfulness",
                    "mean_completeness",
                ):
                    print(
                        f"  Delta {key}: {report['comparison']['delta_candidate_minus_baseline'][key]}"
                    )
        return 0 if report["quality_gates"]["passed"] else 1
    except (
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        OSError,
        PermissionError,
        IndexError,
    ) as exc:
        p.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())