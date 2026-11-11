"""Report artifacts, reproducibility checks, and CLI integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

import agent
from cases import CASES
from harness import SuiteRunner, aggregate
from model import MockModel
from reporting import compare_summaries, markdown_report, write_artifacts


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.runs = SuiteRunner(lambda: MockModel()).run(CASES)
        self.summary = aggregate(self.runs, mode="mock", scenario="normal")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "report"

    def test_write_machine_and_markdown_reports(self):
        paths = write_artifacts(self.out, self.summary, self.runs)
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(path.exists() for path in paths))
        summary_json = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(summary_json["passed"], 9)
        records = [
            json.loads(line) for line in paths[1].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 9)
        self.assertEqual(records[0]["case"]["expected"]["value"], 200000)
        self.assertEqual(records[0]["episode"]["answer"]["value"], 200000)
        self.assertTrue(records[0]["episode"]["events"])
        markdown = paths[2].read_text(encoding="utf-8")
        self.assertIn("Strict task success", markdown)
        self.assertIn("9/9", markdown)
        self.assertIn("revenue_lan", markdown)
        self.assertIn("Evidence grounded", markdown)

    def test_reports_can_be_overwritten_without_partial_json(self):
        write_artifacts(self.out, self.summary, self.runs)
        paths = write_artifacts(self.out, self.summary, self.runs)
        for path in paths[:2]:
            self.assertTrue(path.stat().st_size > 0)
        self.assertEqual(json.loads(paths[0].read_text())["cases"], 9)

    def test_comparison_rejects_mismatched_case_ids(self):
        baseline = dict(self.summary, case_ids=["other"])
        with self.assertRaises(ValueError):
            compare_summaries(self.summary, baseline)

    def test_comparison_rejects_different_dataset_versions(self):
        baseline = dict(self.summary, dataset_version="other")
        with self.assertRaises(ValueError):
            compare_summaries(self.summary, baseline)

    def test_wrong_metric_delta_from_baseline(self):
        wrong = aggregate(
            SuiteRunner(lambda: MockModel("wrong-metric")).run(CASES),
            mode="mock", scenario="wrong-metric",
        )
        deltas = compare_summaries(wrong, self.summary)
        self.assertEqual(deltas["task_success_rate"], -0.3333)
        self.assertEqual(deltas["evidence_grounded_rate"], 0)
        self.assertEqual(deltas["tool_policy_ok_rate"], 0)

    def test_markdown_reports_failures(self):
        wrong = SuiteRunner(lambda: MockModel("wrong-metric")).run(CASES)
        summary = aggregate(wrong, mode="mock", scenario="wrong-metric")
        text = markdown_report(summary, wrong)
        self.assertIn("ANSWER_MISMATCH", text)
        self.assertIn("6/9", text)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "reports"

    def run_cli(self, args, *, env_live="0"):
        cmd = [sys.executable, str(LAB / "agent.py"), *args,
               "--output-dir", str(self.out)]
        env = dict(os.environ, LIVE_MODE=env_live)
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=20)

    def test_default_mock_writes_artifacts(self):
        result = self.run_cli(["--mock"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Strict success: 9/9", result.stdout)
        self.assertTrue((self.out / "summary.json").exists())

    def test_scenario_forces_mock_even_exported_live(self):
        result = self.run_cli(["--scenario", "wrong-metric"], env_live="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Mode: mock | Scenario: wrong-metric", result.stdout)
        self.assertIn("Strict success: 6/9", result.stdout)

    def test_minimum_success_ci_gate(self):
        result = self.run_cli([
            "--scenario", "wrong-metric", "--min-success-rate", "1",
        ])
        self.assertEqual(result.returncode, 1)
        self.assertIn("CI gate FAILED", result.stdout)
        self.assertTrue((self.out / "summary.json").exists())

    def test_specific_case_and_baseline_comparison(self):
        baseline_out = Path(self.tmp.name) / "baseline"
        first = subprocess.run(
            [sys.executable, str(LAB / "agent.py"), "--mock", "--case",
             "revenue_lan", "--output-dir", str(baseline_out)],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.run_cli([
            "--scenario", "wrong-metric", "--case", "revenue_lan",
            "--compare-with", str(baseline_out / "summary.json"),
        ])
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("Change from baseline", second.stdout)
        self.assertIn("task_success_rate: -100.0", second.stdout)

    def test_live_and_scenario_conflict(self):
        result = self.run_cli(["--live", "--scenario", "wrong-metric"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined", result.stderr)

    def test_invalid_minimum_success_rate(self):
        for value in ("-1", "1.1"):
            with self.subTest(value=value):
                result = self.run_cli(["--mock", "--min-success-rate", value])
                self.assertEqual(result.returncode, 2)

    def test_live_adapter_can_be_injected_for_offline_test(self):
        with (
            patch("sys.argv", ["agent.py", "--live", "--case", "revenue_lan",
                               "--output-dir", str(self.out)]),
            patch.object(agent, "openai_compatible_model", return_value=MockModel()),
        ):
            exit_code = agent.main()
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            json.loads((self.out / "summary.json").read_text())["mode"], "live"
        )

    def test_invalid_limits(self):
        result = self.run_cli(["--mock", "--max-model-calls", "0"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integers", result.stderr)


if __name__ == "__main__":
    unittest.main()
