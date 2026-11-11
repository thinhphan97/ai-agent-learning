"""Isolated gold scoring, benchmark strategy comparison and process CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Budgets
from evaluation import _read_suite, compare_strategies, evaluate_strategy

LAB = Path(__file__).resolve().parents[1]


def cli(*argv: str):
    return subprocess.run(
        [sys.executable, str(LAB / "agent.py"), *argv],
        capture_output=True, text=True, timeout=30,
        cwd=LAB.parent, env=dict(os.environ, LIVE_MODE="0"),
    )


class BenchmarkTests(unittest.TestCase):
    def test_gold_labels_separated_and_offline_comparison(self):
        cases, gold = _read_suite()
        self.assertEqual(cases["dataset_version"], "lab14-synthetic-v1")
        self.assertEqual(len(cases["cases"]), 8)
        self.assertIn("finance_delayed", gold)
        report = compare_strategies()
        self.assertEqual(len(report["strategies"]), 7)
        by_strategy = {row["strategy"]: row for row in report["strategies"]}
        self.assertGreater(
            by_strategy["adaptive"]["mean_context_gold_coverage"],
            by_strategy["planned"]["mean_context_gold_coverage"],
        )
        self.assertGreater(
            by_strategy["adaptive"]["total_search_calls"],
            by_strategy["planned"]["total_search_calls"],
        )
        self.assertEqual(by_strategy["adaptive"]["unsafe_scope_count"], 0)
        self.assertEqual(by_strategy["planned"]["unsafe_scope_count"], 0)

    def test_adaptive_helps_faulted_case_but_no_reasons_remains_partial(self):
        report = evaluate_strategy(
            "adaptive",
            cases_filter=("finance_delayed", "finance_unavailable"),
        )
        row = {case["case_id"]: case for case in report["cases"]}
        self.assertEqual(row["finance_delayed"]["context_gold_coverage"], 1)
        self.assertEqual(row["finance_delayed"]["status"], "answered")
        self.assertLess(row["finance_unavailable"]["context_gold_coverage"], 1)
        self.assertEqual(row["finance_unavailable"]["status"], "partial")

    def test_unanswerable_has_undefined_gold_recall(self):
        report = evaluate_strategy("baseline", cases_filter=("unanswerable",))
        self.assertIsNone(report["cases"][0]["recall"])
        self.assertIsNone(report["mean_recall"])
        self.assertIsNone(report["mean_context_gold_coverage"])

    def test_invalid_benchmark_case_fails(self):
        with self.assertRaises(ValueError):
            evaluate_strategy("baseline", cases_filter=("not_a_case",))


class CLITests(unittest.TestCase):
    def test_adaptive_demo_and_plan_trace(self):
        p = cli("--scenario", "delayed-drivers", "--strategy", "adaptive", "--show-trace")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("targeted_recovery", p.stdout)
        self.assertIn("20.00%", p.stdout)
        self.assertIn("sufficient_after_recovery", p.stdout)

    def test_no_drivers_partial_and_no_fabrication(self):
        p = cli("--scenario", "no-drivers", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        payload = json.loads(p.stdout)
        self.assertEqual(payload["answer"]["status"], "partial")
        self.assertIn("drivers", payload["missing_aspects"])
        self.assertNotIn("Reported reasons:", payload["answer"]["answer"])

    def test_benchmark_compare_cli(self):
        p = cli("--benchmark", "--compare", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        data = json.loads(p.stdout)
        self.assertEqual(len(data["strategies"]), 7)

    def test_no_provider_calls_in_default_offline(self):
        p = cli("--strategy", "multiquery", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        result = json.loads(p.stdout)
        self.assertEqual(result["provider"], "mock")
        self.assertEqual(result["usage"]["llm_transform_calls"], 0)
        self.assertEqual(result["usage"]["search_calls"], 3)

    def test_explicit_llm_provider_restricted(self):
        p = cli("--transform-provider", "llm", "--strategy", "adaptive")
        self.assertEqual(p.returncode, 2)
        self.assertIn("only used by", p.stderr)
        p = cli("--transform-provider", "llm", "--benchmark")
        self.assertEqual(p.returncode, 2)

    def test_gates_and_output_file_no_overwrite(self):
        p = cli("--max-search-calls", "0")
        self.assertEqual(p.returncode, 2)
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "report.json")
            ok = cli("--benchmark", "--strategy", "adaptive", "--output", target)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertTrue(Path(target).exists())
            retry = cli("--benchmark", "--strategy", "adaptive", "--output", target)
            self.assertEqual(retry.returncode, 2)
            self.assertIn("Refusing to overwrite", retry.stderr)


if __name__ == "__main__":
    unittest.main()