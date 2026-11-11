"""Quality gates, reports, CLI exit statuses and external inputs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import load_suite
from harness import run_evaluation
from quality_gates import apply_gates, load_gates
from reporting import compare_reports, markdown_report, prepare_report, save_report
from runtime_stub import (
    load_external_traces, load_scripted_traces, validate_trace,
)

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent


def cli(*args: str):
    return subprocess.run(
        [sys.executable, str(LAB / "agent.py"), *args],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
        env=dict(os.environ, LIVE_MODE="0"),
    )


class HarnessTests(unittest.TestCase):
    def test_fail_closed_on_external_answers_without_independent_labels(self):
        traces = load_scripted_traces()
        external = {
            id: replace(t, provenance="external")
            for id, t in traces.items()
        }
        with self.assertRaises(ValueError):
            run_evaluation(external, answer_mode="labels")
        report = run_evaluation(external, answer_mode="none")
        self.assertEqual(report["aggregates"]["unjudged_cases"], 6)
        self.assertIsNone(report["aggregates"]["mean_faithfulness"])
        self.assertFalse(report["quality_gates"]["passed"])
        self.assertTrue(any("NOT MEASURED" in e
                            for e in report["quality_gates"]["failures"]))

    def test_external_annotations_must_exactly_match_observed_answer(self):
        traces = load_scripted_traces()
        external = {id: replace(t, provenance="external") for id, t in traces.items()}
        source = LAB / "data" / "answer_labels.json"
        report = run_evaluation(
            external, answer_mode="labels", annotations_path=source,
        )
        self.assertTrue(report["quality_gates"]["passed"])
        self.assertTrue(all(
            c["answer_metrics"]["evaluation_method"] == "hand_labeled_external"
            for c in report["cases"]
        ))
        external["auth_denied"] = replace(
            external["auth_denied"], answer="Unrelated message"
        )
        with self.assertRaises(ValueError):
            run_evaluation(external, answer_mode="labels", annotations_path=source)

    def test_report_save_and_compare_requires_same_dataset(self):
        baseline = prepare_report(run_evaluation(load_scripted_traces("baseline")))
        candidate = prepare_report(run_evaluation(
            load_scripted_traces("candidate-loss"), scenario="candidate-loss"
        ))
        delta = compare_reports(candidate, baseline)
        self.assertLess(delta["delta_candidate_minus_baseline"][
            "mean_context_gold_coverage"], 0)
        with self.assertRaises(ValueError):
            compare_reports(candidate, {**baseline, "k": 100})
        with tempfile.TemporaryDirectory() as tmp:
            path = save_report(baseline, Path(tmp), "baseline-test")
            self.assertTrue((path / "summary.json").exists())
            self.assertEqual(len((path / "cases.jsonl").read_text().splitlines()), 6)
            self.assertIn("Quality gates: PASS", (path / "report.md").read_text())
            with self.assertRaises(ValueError):
                save_report(baseline, Path(tmp), "../traversal")
            with self.assertRaises(FileExistsError):
                save_report(baseline, Path(tmp), "baseline-test")

    def test_quality_gates_validate_config(self):
        gates = load_gates()
        self.assertTrue(apply_gates(
            run_evaluation(load_scripted_traces())["aggregates"], gates
        )["passed"])
        broken = {**gates, "min_mean_faithfulness": 2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                load_gates(path)

    def test_skipped_judge_if_cross_tenant_trace(self):
        traces = load_scripted_traces("tenant-leak")
        gateway = MagicMock()
        gateway.judge.return_value = {
            "faithfulness": 1, "claim_correctness": 1,
            "completeness": 1, "answer_relevancy": 1,
            "claim_count": 1, "unfaithful_claim_count": 0,
            "invalid_citation_count": 0, "injection_followed": False,
            "evaluation_method": "llm_judge_unverified",
        }
        result = run_evaluation(
            traces, scenario="tenant-leak", case_ids=["refund_multi"],
            answer_mode="llm", llm_gateway=gateway,
        )
        gateway.judge.assert_not_called()
        self.assertEqual(result["aggregates"]["unjudged_cases"], 1)
        self.assertFalse(result["quality_gates"]["passed"])

    def test_answer_relevancy_rubric_is_separate_from_completeness(self):
        baseline = run_evaluation(load_scripted_traces())
        loss = run_evaluation(
            load_scripted_traces("candidate-loss"), scenario="candidate-loss"
        )
        self.assertEqual(baseline["aggregates"]["mean_answer_relevancy"], 1)
        row = next(x for x in loss["cases"] if x["case_id"] == "refund_multi")
        self.assertEqual(row["answer_metrics"]["answer_relevancy"], 1)
        self.assertLess(row["answer_metrics"]["completeness"], 1)

    def test_retrieval_only_gates_explicitly_skip_unjudged_answer_metrics(self):
        report = run_evaluation(
            load_scripted_traces(), answer_mode="none",
            gates_path=LAB / "data" / "retrieval_gates.json",
        )
        self.assertTrue(report["quality_gates"]["passed"])
        self.assertEqual(report["quality_gates"]["scope"], "retrieval_only")
        self.assertTrue(report["quality_gates"]["skipped"])
        self.assertEqual(report["aggregates"]["unjudged_cases"], 6)
        self.assertIsNone(report["aggregates"]["mean_faithfulness"])

    def test_external_trace_json_import(self):
        traces = load_scripted_traces()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.json"
            path.write_text(json.dumps([
                {k: v for k, v in asdict(trace).items()
                 if k != "provenance"} for trace in traces.values()
            ]))
            got = load_external_traces(path)
            self.assertEqual(set(got), set(traces))
            self.assertTrue(all(x.provenance == "external" for x in got.values()))
            report = run_evaluation(got, answer_mode="none")
            self.assertFalse(report["quality_gates"]["passed"])
            self.assertEqual(report["aggregates"]["mean_retrieval_recall"], 1)


class CLITests(unittest.TestCase):
    def test_baseline_pass_and_failure_exit_codes(self):
        ok = cli("--scenario", "baseline", "--no-save", "--json")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        payload = json.loads(ok.stdout)
        self.assertTrue(payload["quality_gates"]["passed"])
        for scenario in ("candidate-loss", "hallucination", "stale-evidence",
                         "tenant-leak", "injection-followed", "abstention-failure"):
            bad = cli("--scenario", scenario, "--no-save")
            self.assertEqual(bad.returncode, 1, bad.stderr)
            self.assertIn("Quality Gates: FAIL", bad.stdout)

    def test_live_requires_single_case_and_explicit_evaluator(self):
        missing = cli("--live", "--no-save")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("exactly one --case", missing.stderr)
        missing_eval = cli("--live", "--case", "refund_multi", "--no-save")
        self.assertEqual(missing_eval.returncode, 2)
        self.assertIn("annotations-file", missing_eval.stderr)
        judge_limit = cli("--judge", "llm", "--no-save")
        self.assertEqual(judge_limit.returncode, 2)

    def test_all_offline_modes_do_not_need_llm_credentials(self):
        out = cli("--case", "refund_multi", "--no-save", "--json")
        self.assertEqual(out.returncode, 0, out.stderr)
        metrics = json.loads(out.stdout)["aggregates"]
        self.assertEqual(metrics["mean_faithfulness"], 1)

    def test_cli_report_and_prior_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = cli(
                "--run-id", "control", "--output-root", tmp,
                "--scenario", "baseline",
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            prior = str(Path(tmp) / "control" / "summary.json")
            newer = cli(
                "--run-id", "candidate", "--output-root", tmp,
                "--scenario", "candidate-loss", "--compare-with", prior,
                "--json",
            )
            self.assertEqual(newer.returncode, 1, newer.stderr)
            payload = json.loads(newer.stdout)
            self.assertLess(payload["comparison"]["delta_candidate_minus_baseline"][
                "mean_context_gold_coverage"], 0)
            self.assertTrue((Path(tmp) / "candidate" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()