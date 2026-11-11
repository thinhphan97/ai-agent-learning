"""Lab 6 CLI: run an evaluation suite and save reproducible local reports.

Examples:
    python 6-agent-evaluation/agent.py --mock
    python 6-agent-evaluation/agent.py --scenario wrong-metric
    python 6-agent-evaluation/agent.py --live --case revenue_lan
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cases import CASES, select_cases
from contracts import Limits
from harness import SuiteRunner, aggregate
from model import MockModel, openai_compatible_model
from reporting import compare_summaries, write_artifacts

DEFAULT_REPORTS = Path(__file__).resolve().parent / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 6: Agent Evaluation Harness")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="Force offline mock mode")
    mode.add_argument("--live", action="store_true", help="Use real LLM from root .env")
    parser.add_argument(
        "--scenario",
        choices=MockModel.SCENARIOS,
        help="Force mock scenario, even when LIVE_MODE=1 is exported",
    )
    parser.add_argument(
        "--case", choices=[c.id for c in CASES], help="Evaluate only this case"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory (default: this lab's reports/<mode>)",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="Compare with a prior summary.json from identical cases",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        help="Optional CI gate between 0 and 1 (exit code 1 on insufficient success)",
    )
    parser.add_argument("--max-model-calls", type=int, default=8)
    parser.add_argument("--max-tool-attempts", type=int, default=4)
    args = parser.parse_args()
    if args.live and args.scenario is not None:
        parser.error("--live and --scenario cannot be combined")
    if args.min_success_rate is not None and not 0 <= args.min_success_rate <= 1:
        parser.error("--min-success-rate must be between 0 and 1")
    try:
        limits = Limits(args.max_model_calls, args.max_tool_attempts)
    except ValueError as exc:
        parser.error(str(exc))

    live = args.live or (
        not args.mock and args.scenario is None and os.getenv("LIVE_MODE") == "1"
    )
    scenario = args.scenario or "normal"
    factory = openai_compatible_model if live else lambda: MockModel(scenario)
    cases = select_cases(args.case)
    output_dir = args.output_dir or DEFAULT_REPORTS / (
        "live" if live else f"mock-{scenario}"
    )
    print(
        f"Mode: {'live' if live else 'mock'} | Scenario: {scenario} | Cases: {len(cases)}"
    )
    runs = SuiteRunner(factory, limits=limits).run(cases)
    summary = aggregate(runs, mode="live" if live else "mock", scenario=scenario)
    summary["limits"] = {
        "max_model_calls": limits.max_model_calls,
        "max_tool_attempts": limits.max_tool_attempts,
    }
    outputs = write_artifacts(output_dir, summary, runs)
    for result in runs:
        score = result.score
        answer = score.observed.as_dict() if score.observed else None
        print(
            f"{score.id}: {'PASS' if score.task_success else 'FAIL'} "
            f"({score.status}; exact={score.answer_exact}; "
            f"grounded={score.evidence_grounded}; answer={answer})"
        )
    print(
        f"Strict success: {summary['passed']}/{summary['cases']} "
        f"({summary['task_success_rate'] * 100:.1f}%) | "
        f"Runtime completed: {summary['runtime_completion_rate'] * 100:.1f}%"
    )
    for output in outputs:
        print(f"Report: {output}")

    if args.compare_with:
        try:
            baseline = json.loads(args.compare_with.read_text(encoding="utf-8"))
            deltas = compare_summaries(summary, baseline)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Could not compare reports: {exc}")
        print("Change from baseline (percentage points):")
        for name, delta in deltas.items():
            print(f"  {name}: {delta * 100:+.1f}")

    if (
        args.min_success_rate is not None
        and summary["task_success_rate"] < args.min_success_rate
    ):
        print(f"CI gate FAILED: required {args.min_success_rate * 100:.1f}% success.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
