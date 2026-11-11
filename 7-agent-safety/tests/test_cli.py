"""CLI integration, audit export and real-adapter wiring without network in tests."""

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
from model import MockModel
from policy import parse_intent


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_cli(self, *args, live="0"):
        return subprocess.run(
            [sys.executable, str(LAB / "agent.py"), *args],
            text=True, capture_output=True,
            env=dict(os.environ, LIVE_MODE=live), timeout=15,
        )

    def test_default_read_passes_without_write(self):
        result = self.run_cli("--mock")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Status: completed; code: VERIFIED", result.stdout)
        self.assertIn("simulated writes: 0", result.stdout)

    def test_mock_scenario_overrides_live_env(self):
        result = self.run_cli("--scenario", "prompt-injection", live="1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Mode: mock", result.stdout)
        self.assertIn("UNKNOWN_TOOL", result.stdout)

    def test_approval_denial_exit_code_and_no_side_effect(self):
        result = self.run_cli(
            "--mock", "--question",
            "Set note for customer_id=1 to 'Call Friday'.",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("APPROVAL_REQUIRED", result.stdout)
        self.assertIn("simulated writes: 0", result.stdout)

    def test_explicit_approval_uses_exact_note(self):
        result = self.run_cli(
            "--mock", "--question",
            "Set note for customer_id=1 to 'Call Friday'.",
            "--approve",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("set_customer_note -> ALLOWED", result.stdout)
        self.assertIn("simulated writes: 1", result.stdout)

    def test_scope_is_checked_before_llm_calls(self):
        result = self.run_cli("--mock", "--principal", "bob")
        self.assertEqual(result.returncode, 1)
        self.assertIn("SCOPE_DENIED", result.stdout)
        self.assertIn("Model calls: 0", result.stdout)

    def test_approval_on_read_rejected_by_cli(self):
        result = self.run_cli("--mock", "--approve")
        self.assertEqual(result.returncode, 2)
        self.assertIn("only for explicit note-write", result.stderr)

    def test_live_scenario_conflict(self):
        result = self.run_cli("--live", "--scenario", "normal")
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined", result.stderr)

    def test_negative_budget_validation(self):
        result = self.run_cli("--mock", "--max-tool-attempts", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integers", result.stderr)

    def test_write_audit_file_is_redacted(self):
        path = Path(self.tmp.name) / "reports" / "audit.json"
        result = self.run_cli(
            "--scenario", "prompt-injection",
            "--audit-file", str(path),
        )
        self.assertEqual(result.returncode, 1)
        audit = json.loads(path.read_text())
        self.assertEqual(audit["summary"]["code"], "UNKNOWN_TOOL")
        self.assertEqual([e["seq"] for e in audit["events"]],
                         list(range(1, len(audit["events"]) + 1)))
        self.assertNotIn("SYSTEM OVERRIDE", path.read_text())
        self.assertNotIn("Returning customer", path.read_text())

    def test_live_wiring_uses_real_mode_adapter_interface(self):
        with (
            patch("sys.argv", ["agent.py", "--live", "--question", "Show customer_id=1."]),
            patch.object(
                agent, "openai_compatible_model",
                return_value=MockModel(parse_intent("Show customer_id=1.")),
            ),
        ):
            self.assertEqual(agent.main(), 0)


if __name__ == "__main__":
    unittest.main()
