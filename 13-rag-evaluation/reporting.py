"""Stable evaluation reports, comparison and regression deltas."""
from __future__ import annotations

import json
import re
from pathlib import Path

from diagnostics import diagnose


def compare_reports(current: dict, prior: dict) -> dict:
    for key in ("dataset_version", "k", "case_ids", "answer_mode"):
        if current.get(key) != prior.get(key):
            raise ValueError(f"Cannot compare reports: {key} differs")
    now = current["aggregates"]
    before = prior["aggregates"]
    changes = {}
    for key, val in now.items():
        prev = before.get(key)
        if type(val) in (int, float) and type(prev) in (int, float):
            changes[key] = round(val - prev, 6)
        else:
            changes[key] = None
    return {
        "baseline_scenario": prior.get("scenario"),
        "candidate_scenario": current.get("scenario"),
        "delta_candidate_minus_baseline": changes,
        "baseline_gates_passed": prior["quality_gates"]["passed"],
        "candidate_gates_passed": current["quality_gates"]["passed"],
    }


def _print_score(value) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def markdown_report(report: dict) -> str:
    g = report["aggregates"]
    lines = [
        "# Lab 13 — RAG Evaluation Report", "",
        f"Dataset: {report['dataset_version']} | Scenario: {report['scenario']} "
        f"| K={report['k']} | Answer evaluator: {report['answer_mode']}",
        "",
        f"**Quality gates: {'PASS' if report['quality_gates']['passed'] else 'FAIL'}**",
        "",
        "| Metric | Value |", "| --- | ---: |",
    ]
    for name in (
        "mean_precision_at_k", "mean_retrieval_recall", "mean_reciprocal_rank",
        "mean_ndcg", "mean_candidate_gold_coverage",
        "mean_context_gold_coverage", "mean_faithfulness",
        "mean_claim_correctness", "mean_completeness",
        "abstention_accuracy", "unjudged_cases",
        "unauthorized_documents", "stale_documents",
        "unfaithful_claims", "invalid_citations",
        "injection_followed_cases",
    ):
        val = g[name]
        lines.append(
            f"| {name} | {val if type(val) is int else _print_score(val)} |"
        )
    lines.extend([
        "",
        f"Evaluation gate scope: {report['quality_gates'].get('scope', 'full_rag')}",
        "",
        "## Unmeasured/skipped gates", "",
    ])
    lines.extend(
        f"- {name}" for name in report["quality_gates"].get("skipped", [])
    )
    if not report["quality_gates"].get("skipped"):
        lines.append("None.")
    lines.extend(["", "## Failed gates", ""])
    if report["quality_gates"]["failures"]:
        lines.extend(f"- {problem}" for problem in report["quality_gates"]["failures"])
    else:
        lines.append("None.")
    lines.extend([
        "", "## Case diagnosis", "",
        "| Case | Retrieval | Candidates | Context | Faithfulness | Completeness | Issues |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for case in report["cases"]:
        m = case["metrics"]
        a = case["answer_metrics"]
        issues = ", ".join(case["diagnosis"]) or "none"
        lines.append(
            f"| {case['case_id']} | {_print_score(m['retrieval_gold_coverage'])} "
            f"| {_print_score(m['candidate_gold_coverage'])} "
            f"| {_print_score(m['context_gold_coverage'])} "
            f"| {_print_score(a['faithfulness'])} "
            f"| {_print_score(a['completeness'])} | {issues} |"
        )
    lines.extend([
        "", "## Important limits", "",
        report["metrics_note"],
        "",
        "Synthetic fixture usage is illustrative; do not treat it as measured "
        "provider latency, token usage or cost.",
        "",
        "A matching citation ID is not semantic verification. Hand-labeled "
        "claim support is valid only for the exact scripted answers. LLM "
        "judge evaluations, when selected, are fallible model opinions.",
        "",
    ])
    return "\n".join(lines)


def save_report(report: dict, root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run_id must be a safe name with up to 64 characters")
    path = root.resolve() / run_id
    path.mkdir(parents=True, exist_ok=False)
    (path / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (path / "cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in report["cases"]),
        encoding="utf-8",
    )
    (path / "report.md").write_text(markdown_report(report), encoding="utf-8")
    return path


def prepare_report(report: dict) -> dict:
    for row in report["cases"]:
        row["diagnosis"] = diagnose(row)
    return report