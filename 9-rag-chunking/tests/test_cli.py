"""All strategies are runnable using only stdlib and local sample data."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


LAB = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, str(LAB / "agent.py"), *args],
            capture_output=True, text=True, timeout=25,
            env=dict(os.environ, LIVE_MODE="0"),
        )

    def test_all_strategies_offline(self):
        r = self.invoke("--strategy", "all", "--limit", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[fixed]", r.stdout)
        self.assertIn("[semantic]", r.stdout)
        self.assertIn("[late]", r.stdout)
        self.assertIn("[raptor]", r.stdout)

    def test_individual_source_metadata(self):
        r = self.invoke("--strategy", "structure")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Annual Leave", r.stdout)
        self.assertIn("parent=policies:section:", r.stdout)

    def test_benchmark_reports_not_truth_accuracy(self):
        r = self.invoke("--benchmark")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Evidence benchmark", r.stdout)
        self.assertIn("NOT semantic relevance or answer accuracy", r.stdout)

    def test_json_mode_is_valid_json(self):
        import json
        r = self.invoke("--strategy", "table", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertEqual(result[0]["method"], "table")
        self.assertEqual(result[0]["parents"][0]["kind"], "table")

    def test_invalid_advanced_provider_fails_before_external_call(self):
        r = self.invoke("--strategy", "fixed", "--provider", "api")
        self.assertEqual(r.returncode, 2)
        self.assertIn("API provider only applies", r.stderr)

    def test_single_rag_strategy_options(self):
        r = self.invoke("--strategy", "semantic", "--size", "18", "--threshold", "0.3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("transcript:semantic:", r.stdout)

    def test_bad_document_file(self):
        r = self.invoke("--strategy", "table", "--file", "/missing/custom.md")
        self.assertEqual(r.returncode, 2)

    def test_invalid_strategy(self):
        r = self.invoke("--strategy", "unknown")
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
