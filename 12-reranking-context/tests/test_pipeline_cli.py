"""Full pipeline stage tracing, fixed-label evaluation and offline CLI smoke."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import evaluate
from pipeline import RAGPipeline

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent


def cli(*args: str):
    return subprocess.run(
        [sys.executable, str(LAB / "agent.py"), *args],
        cwd=str(ROOT),
        capture_output=True, text=True, timeout=30,
        env=dict(os.environ, LIVE_MODE="0"),
    )


class PipelineTests(unittest.TestCase):
    def test_pipeline_contains_every_observable_stage(self):
        result = RAGPipeline().run(
            "What are refund deadline, conditions and exceptions?",
            tenant_id="tenant_a",
            retrieval_k=30, candidate_k=9, max_evidence=3,
            required_aspects=("deadline", "conditions", "exceptions"),
            token_budget=200,
        )
        self.assertTrue(result["retrieved"])
        self.assertEqual(len(result["candidates"]), 9)
        self.assertEqual(len(result["reranked"]), 9)
        self.assertEqual(len(result["context"]["evidence"]), 3)
        self.assertEqual(result["context"]["missing_aspects"], ())
        self.assertEqual(result["evidence_coverage_by_curated_metadata"], 1)
        self.assertTrue(all(e["source"].endswith(".md")
                            for e in result["context"]["evidence"]))

    def test_candidate_budget_loss_is_observable(self):
        result = RAGPipeline().run(
            "What are the refund deadline, conditions and exceptions?",
            tenant_id="tenant_a",
            retrieval_k=30, candidate_k=1,
            max_evidence=1, token_budget=200,
            required_aspects=("deadline", "conditions", "exceptions"),
        )
        self.assertEqual(len(result["candidates"]), 1)
        self.assertLess(result["evidence_coverage_by_curated_metadata"], 1)
        self.assertEqual(len(result["context"]["missing_aspects"]), 2)

    def test_tenant_scope_does_not_relax_on_zero_hits(self):
        result = RAGPipeline().run(
            "45 day Tenant B refund", tenant_id="tenant_a",
            retrieval_k=30, candidate_k=10, token_budget=200,
        )
        ids = [e["parent_id"] for e in result["context"]["evidence"]]
        self.assertNotIn("tenant_b_refund", ids)
        self.assertFalse(any("tenant_b_private" in e["source"]
                             for e in result["context"]["evidence"]))

    def test_incompatible_candidate_limit_fails(self):
        with self.assertRaises(ValueError):
            RAGPipeline().run("test", tenant_id="tenant_a",
                              retrieval_k=5, candidate_k=6)
        with self.assertRaises(ValueError):
            RAGPipeline().run("test", tenant_id="tenant_a",
                              required_aspects=("deadline", "deadline"))

    def test_llm_contextualization_requires_scoped_index(self):
        with self.assertRaises(ValueError):
            RAGPipeline(contextualization="llm")

    def test_fixed_gold_labels_independent_and_comparable(self):
        result = evaluate(
            contextualization="none", retrieval_k=20,
            candidate_k=3, max_evidence=3, token_budget=220,
        )
        self.assertEqual(result["benchmark"], "fixed_synthetic_parent_evidence_labels")
        self.assertEqual(len(result["cases"]), 3)
        for case in result["cases"]:
            for metric in ("retrieval_gold_coverage", "candidate_gold_coverage",
                           "context_gold_coverage"):
                self.assertTrue(0 <= case[metric] <= 1)

    def test_no_false_claim_of_contextual_improvement(self):
        a = evaluate(contextualization="none")
        b = evaluate(contextualization="metadata")
        self.assertEqual(a["benchmark"], b["benchmark"])
        self.assertTrue(0 <= b["mean_context_gold_coverage"] <= 1)

    def test_optional_model_requires_explicit_local_path(self):
        with self.assertRaises(ValueError):
            RAGPipeline(reranker="cross-encoder")


class CLITests(unittest.TestCase):
    def test_default_demo(self):
        process = cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Evidence:", process.stdout)
        self.assertIn("Missing aspects: ()", process.stdout)

    def test_json_and_provenance(self):
        process = cli("--scenario", "refund", "--json", "--max-evidence", "3")
        self.assertEqual(process.returncode, 0, process.stderr)
        output = json.loads(process.stdout)
        self.assertEqual(len(output["context"]["evidence"]), 3)
        self.assertTrue(all(e["source"] for e in output["context"]["evidence"]))
        self.assertEqual(output["context"]["token_counter"],
                         "whitespace_proxy_not_model_tokens")

    def test_compare_reports_both_modes(self):
        p = cli("--benchmark", "--compare", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        report = json.loads(p.stdout)
        self.assertEqual([r["contextualization"] for r in report],
                         ["none", "metadata"])

    def test_real_cross_encoder_is_not_silently_simulated(self):
        p = cli("--reranker", "cross-encoder")
        self.assertEqual(p.returncode, 2)
        self.assertIn("--model-path", p.stderr)

    def test_explicit_citation_id_check(self):
        p = cli("--answer", "The request deadline is 30 days [E1].", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(json.loads(p.stdout)["citation_validation"]["citation_ids_valid"])

    def test_unknown_tenant_has_no_evidence(self):
        p = cli("--tenant-id", "tenant_nonexistent", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["context"]["evidence"], [])

    def test_readme_explains_external_service_policy(self):
        root = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("External services and local development", root)


if __name__ == "__main__":
    unittest.main()
