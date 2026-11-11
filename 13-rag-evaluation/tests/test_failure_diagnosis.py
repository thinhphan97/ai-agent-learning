"""Reproduce controlled failure cases and localize the first broken stage."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answer_metrics import load_labels, score_answer, select_labels
from dataset import load_suite
from harness import run_evaluation
from runtime_stub import load_scripted_traces, validate_trace


def evaluate(scenario: str):
    return run_evaluation(load_scripted_traces(scenario), scenario=scenario)


class EvaluationScenarios(unittest.TestCase):
    def test_baseline_is_full_and_passes_all_quality_gates(self):
        report = evaluate("baseline")
        self.assertTrue(report["quality_gates"]["passed"])
        g = report["aggregates"]
        self.assertEqual(g["case_count"], 6)
        self.assertEqual(g["mean_retrieval_recall"], 1)
        self.assertEqual(g["mean_candidate_gold_coverage"], 1)
        self.assertEqual(g["mean_context_gold_coverage"], 1)
        self.assertEqual(g["mean_faithfulness"], 1)
        self.assertEqual(g["mean_claim_correctness"], 1)
        self.assertEqual(g["mean_completeness"], 1)
        self.assertEqual(g["abstention_accuracy"], 1)
        self.assertEqual(g["unauthorized_documents"], 0)
        self.assertEqual(g["unjudged_cases"], 0)

    def test_candidate_selection_drops_correct_evidence(self):
        report = evaluate("candidate-loss")
        self.assertFalse(report["quality_gates"]["passed"])
        row = next(c for c in report["cases"] if c["case_id"] == "refund_multi")
        self.assertEqual(row["metrics"]["retrieval_gold_coverage"], 1)
        self.assertAlmostEqual(row["metrics"]["candidate_gold_coverage"], 2 / 3)
        self.assertAlmostEqual(row["metrics"]["context_gold_coverage"], 2 / 3)
        self.assertEqual(row["answer_metrics"]["faithfulness"], 1)
        self.assertAlmostEqual(row["answer_metrics"]["completeness"], 2 / 3)
        self.assertIn("candidate_selection:evidence_loss", __import__(
            "diagnostics").diagnose(row))

    def test_supported_citation_is_not_proof_of_faithfulness(self):
        report = evaluate("hallucination")
        row = next(c for c in report["cases"] if c["case_id"] == "refund_multi")
        a = row["answer_metrics"]
        self.assertTrue(a["citation_ids_valid"])
        self.assertEqual(a["faithfulness"], 0.75)
        self.assertEqual(a["claim_correctness"], 0.75)
        self.assertEqual(a["completeness"], 1)
        self.assertEqual(a["unfaithful_claim_count"], 1)
        self.assertEqual(a["invalid_citation_count"], 1)
        self.assertFalse(report["quality_gates"]["passed"])

    def test_stale_document_can_be_faithful_but_factually_incorrect(self):
        report = evaluate("stale-evidence")
        row = next(c for c in report["cases"] if c["case_id"] == "refund_current")
        self.assertEqual(row["answer_metrics"]["faithfulness"], 1)
        self.assertEqual(row["answer_metrics"]["claim_correctness"], 0)
        self.assertEqual(row["answer_metrics"]["completeness"], 0)
        self.assertEqual(row["security"]["stale_context_document_ids"],
                         ["refund_deadline_v1"])
        self.assertFalse(report["quality_gates"]["passed"])

    def test_tenant_leak_is_hard_failure_even_if_coverage_is_perfect(self):
        report = evaluate("tenant-leak")
        row = next(c for c in report["cases"] if c["case_id"] == "refund_multi")
        self.assertEqual(row["metrics"]["context_gold_coverage"], 1)
        self.assertEqual(row["security"]["unauthorized_document_ids"],
                         ["tenant_b_confidential"])
        self.assertEqual(report["aggregates"]["unauthorized_documents"], 1)
        self.assertFalse(report["quality_gates"]["passed"])

    def test_injection_followed_is_annotated_not_claimed_auto_detected(self):
        report = evaluate("injection-followed")
        row = next(c for c in report["cases"] if c["case_id"] == "reset_injection")
        self.assertTrue(row["answer_metrics"]["injection_followed"])
        self.assertEqual(report["aggregates"]["injection_followed_cases"], 1)
        self.assertFalse(report["quality_gates"]["passed"])

    def test_unanswerable_must_abstain(self):
        baseline = evaluate("baseline")
        row = next(c for c in baseline["cases"] if c["case_id"] == "unanswerable_salary")
        self.assertIsNone(row["metrics"]["recall_at_k"])
        self.assertIsNone(row["answer_metrics"]["faithfulness"])
        self.assertEqual(row["answer_metrics"]["completeness"], 1)
        report = evaluate("abstention-failure")
        bad = next(c for c in report["cases"] if c["case_id"] == "unanswerable_salary")
        self.assertFalse(bad["answer_metrics"]["abstention_correct"])
        self.assertEqual(bad["answer_metrics"]["completeness"], 0)
        self.assertFalse(report["quality_gates"]["passed"])

    def test_annotation_is_exact_only_for_scripted_answers(self):
        labels = load_labels()
        trace = load_scripted_traces()["refund_multi"]
        self.assertIsNotNone(select_labels(labels, "baseline", trace))
        self.assertIsNone(select_labels(
            labels, "baseline", replace(trace, provenance="external")))
        with self.assertRaises(ValueError):
            select_labels(labels, "baseline", replace(trace, answer="Different answer"))

    def test_invalid_or_forged_stage_trace_rejected(self):
        _, docs, cases = load_suite()
        original = load_scripted_traces()["refund_multi"]
        with self.assertRaises(ValueError):
            validate_trace(
                replace(original, candidates=("tenant_b_confidential",)),
                cases["refund_multi"].id, "tenant_a", docs,
            )
        with self.assertRaises(ValueError):
            validate_trace(
                replace(original, retrieved=("refund_deadline_v2",) * 2),
                "refund_multi", "tenant_a", docs,
            )
        with self.assertRaises(ValueError):
            validate_trace(original, "refund_multi", "tenant_b", docs)


if __name__ == "__main__":
    unittest.main()
