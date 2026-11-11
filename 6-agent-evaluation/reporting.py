"""Machine-readable evaluation artifacts plus a human-readable Markdown report."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from harness import CaseRun


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".eval-",
            delete=False,
        ) as file:
            temporary = file.name
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def markdown_report(summary: dict[str, Any], runs: Sequence[CaseRun]) -> str:
    percent = lambda k: f"{summary[k] * 100:.1f}%"
    lines = [
        "# Lab 6 — Agent Evaluation Report",
        "",
        f"Dataset: `{summary['dataset_version']}` | "
        f"Mode: `{summary['mode']}` | Scenario: `{summary['scenario']}`",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Strict task success | **{summary['passed']}/{summary['cases']}** "
        f"({percent('task_success_rate')}) |",
        f"| Runtime completion | {percent('runtime_completion_rate')} |",
        f"| Exact answer | {percent('answer_exact_rate')} |",
        f"| Evidence grounded | {percent('evidence_grounded_rate')} |",
        f"| Tool policy OK | {percent('tool_policy_ok_rate')} |",
        f"| Budget OK | {percent('budget_ok_rate')} |",
        f"| Average model calls | {summary['avg_model_calls']} |",
        f"| Average tool attempts | {summary['avg_tool_attempts']} |",
        f"| Average runtime (ms) | {summary['avg_elapsed_ms']} |",
        "",
        "Strict success requires: run completed, exact answer, grounded evidence, "
        "tool policy OK, and budgets OK. A completed run alone is **not** a pass.",
        "",
        "## Case Results",
        "",
        "| Case | Category | Status | Answer exact | Grounded | Success | Failure reasons |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        s = run.score
        reasons = ", ".join(s.failure_reasons) if s.failure_reasons else "—"
        lines.append(
            f"| `{s.id}` | {s.category} | {s.status} | "
            f"{'PASS' if s.answer_exact else 'FAIL'} | "
            f"{'PASS' if s.evidence_grounded else 'FAIL'} | "
            f"{'PASS' if s.task_success else 'FAIL'} | {reasons} |"
        )

    lines += [
        "",
        "## Category Breakdown",
        "",
        "| Category | Passed / Cases | Success rate |",
        "| --- | ---: | ---: |",
    ]
    for category, entry in summary["by_category"].items():
        lines.append(
            f"| {category} | {entry['passed']}/{entry['cases']} | "
            f"{entry['success_rate'] * 100:.1f}% |"
        )
    lines += [
        "",
        "## Failure Taxonomy",
        "",
    ]
    if summary["failure_reasons"]:
        for reason, count in summary["failure_reasons"].items():
            lines.append(f"- `{reason}`: {count} case(s)")
    else:
        lines.append("No failures.")

    lines += [
        "",
        "## Reading the trace",
        "",
        "Open `cases.jsonl` to see the exact expected/observed answer, "
        "successful tool evidence, model/tool call counters and numbered events. "
        "Only the evaluator sees the fixture's expected result; "
        "the agent receives only its question.",
        "",
        "## Limitations",
        "",
        "This small dataset is an educational deterministic fixture suite, "
        "not a statistically representative benchmark. The evaluator checks "
        "tool evidence but does not independently prove external business data "
        "accuracy. Provider token usage and cost are not measured here.",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(
    output_dir: Path, summary: dict[str, Any], runs: Sequence[CaseRun]
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    cases_path = output_dir / "cases.jsonl"
    report_path = output_dir / "report.md"
    _write_atomic(
        summary_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    records = [
        json.dumps(
            {
                "case": run.score.as_dict(),
                "episode": run.episode.as_dict(),
            },
            ensure_ascii=False,
        )
        for run in runs
    ]
    _write_atomic(cases_path, "\n".join(records) + "\n")
    _write_atomic(report_path, markdown_report(summary, runs))
    return summary_path, cases_path, report_path


def compare_summaries(
    current: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    if current.get("dataset_version") != baseline.get("dataset_version") or current.get(
        "case_ids"
    ) != baseline.get("case_ids"):
        raise ValueError("Cannot compare differing evaluation datasets or selections")
    metrics = (
        "task_success_rate",
        "answer_exact_rate",
        "evidence_grounded_rate",
        "tool_policy_ok_rate",
    )
    return {key: round(current[key] - baseline[key], 4) for key in metrics}
