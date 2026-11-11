"""Offline CLI smoke tests and explicitly unsupported-mode failures."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def execute(*args: str):
    return subprocess.run(
        [sys.executable, str(LAB / "agent.py"), *args],
        capture_output=True,
        text=True,
        timeout=35,
        env=dict(os.environ, LIVE_MODE="0"),
    )


class CLITests(unittest.TestCase):
    def test_geometry_has_correct_metric_rankings(self):
        p = execute("--geometry", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        result = json.loads(p.stdout)
        self.assertEqual(result["cosine"][0]["id"], "C")
        self.assertEqual(result["dot"][0]["id"], "C")
        self.assertEqual(result["l2"][0]["id"], "B")

    def test_exact_search(self):
        p = execute(
            "--backend",
            "exact",
            "--question",
            "CUDA out of memory in vLLM",
            "--json",
            "--top-k",
            "5",
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        result = json.loads(p.stdout)
        self.assertEqual(result["results"][0]["id"], "vllm_oom")
        self.assertTrue(
            all(x["metadata"]["tenant_id"] == "tenant_a" for x in result["results"])
        )

    def test_hnsw_and_filters(self):
        p = execute(
            "--backend",
            "educational-hnsw",
            "--question",
            "FPT international software revenue Q2 2025",
            "--company",
            "FPT Corporation",
            "--period",
            "Q2/2025",
            "--filter-mode",
            "exact-filter",
            "--json",
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        result = json.loads(p.stdout)
        self.assertEqual(result["results"][0]["id"], "fpt_q2_sw")

    def test_crud_demo_does_not_mutate_file(self):
        p = execute("--crud-demo", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        result = json.loads(p.stdout)
        self.assertTrue(result["updated_vector"])
        self.assertTrue(result["deleted_record_absent"])
        self.assertFalse(result["persistent_file_changed"])

    def test_persistence_across_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "local.sqlite3")
            write = execute("--db", path, "--index-only")
            self.assertEqual(write.returncode, 0, write.stderr)
            read = execute("--db", path, "--backend", "exact", "--json")
            self.assertEqual(read.returncode, 0, read.stderr)
            self.assertEqual(json.loads(read.stdout)["indexed"], 10)

    def test_unsupported_filters_and_bad_dimensions_rejected(self):
        wrong = execute(
            "--backend",
            "exact",
            "--query-vector",
            "0.1,0.2,0.3",
        )
        self.assertEqual(wrong.returncode, 2)
        self.assertIn("dimensions", wrong.stderr)

    def test_benchmark_runs_offline(self):
        p = execute("--benchmark", "--benchmark-size", "64", "--top-k", "3")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("ANN Recall@K", p.stdout)

    def test_optional_qdrant_and_hnswlib_fail_early_if_missing(self):
        for name, package in (("hnswlib", "hnswlib"), ("qdrant", "qdrant_client")):
            if importlib.util.find_spec(package) is not None:
                continue
            p = execute("--backend", name)
            self.assertEqual(p.returncode, 2)
            self.assertIn("Install optional", p.stderr)

    def test_bad_benchmark_provider_fails_before_network(self):
        p = execute("--benchmark", "--embedding", "api")
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
