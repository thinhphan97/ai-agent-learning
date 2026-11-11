"""Run CLI in offline subprocesses; never call external model services in tests."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


LAB = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def execute(self, *args, live_mode="0"):
        return subprocess.run(
            [sys.executable, str(LAB / "agent.py"), *args],
            capture_output=True, text=True,
            env=dict(os.environ, LIVE_MODE=live_mode),
            timeout=15,
        )

    def test_default_run_explains_retrieved_source(self):
        result = self.execute("--mock")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Retriever: tfidf", result.stdout)
        self.assertIn("Retrieved: returns:0", result.stdout)
        self.assertIn("30 days", result.stdout)
        self.assertIn("Status: answered", result.stdout)

    def test_no_relevant_document_abstains(self):
        result = self.execute(
            "--mock", "--question", "What is the employee cafeteria menu?"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NO_RETRIEVAL_EVIDENCE", result.stdout)
        self.assertIn("Status: abstain", result.stdout)

    def test_eval_mock_reports_recall(self):
        result = self.execute("--mock", "--eval")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Source Recall@3", result.stdout)
        self.assertIn("Answer checks", result.stdout)
        self.assertIn("Negative abstention", result.stdout)

    def test_mock_overrides_live_mode_environment(self):
        result = self.execute("--mock", live_mode="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("extractive-mock", result.stdout)

    def test_invalid_chunk_arguments_are_user_friendly(self):
        result = self.execute("--mock", "--chunk-size", "6", "--overlap", "6")
        self.assertEqual(result.returncode, 2)
        self.assertIn("overlap < chunk_size", result.stderr)

    def test_invalid_retriever_selected(self):
        result = self.execute("--retriever", "imaginary")
        self.assertEqual(result.returncode, 2)

    def test_dense_requires_separate_embedding_model(self):
        # A missing model cannot silently fall back to the LLM/chat model.
        # Run with a blank root .env embedding key via env override.
        result = subprocess.run(
            [sys.executable, str(LAB / "agent.py"), "--mock", "--retriever", "dense"],
            capture_output=True, text=True,
            env=dict(os.environ, LIVE_MODE="0", EMBEDDING_MODEL=""),
            timeout=15,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("EMBEDDING_MODEL", result.stderr)


if __name__ == "__main__":
    unittest.main()
