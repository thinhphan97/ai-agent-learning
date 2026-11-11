"""Classical IR metrics validated against hand-worked examples."""
from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import load_cases, load_documents
from retrieval_metrics import (
    evidence_coverage, ndcg_at_k, precision_at_k, recall_at_k,
    reciprocal_rank_at_k,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_precision_and_recall_have_different_denominators(self):
        ranking = ["A", "X", "B", "Y", "C"]
        grades = {"A": 3, "B": 3, "C": 3}
        self.assertAlmostEqual(recall_at_k(ranking, grades, 3), 2 / 3)
        self.assertAlmostEqual(precision_at_k(ranking, grades, 3), 2 / 3)
        self.assertEqual(recall_at_k(ranking, grades, 5), 1.0)
        self.assertEqual(precision_at_k(ranking, grades, 5), 3 / 5)
        self.assertEqual(precision_at_k(["A"], grades, 5), 1 / 5)

    def test_reciprocal_rank_not_all_relevant_ranked(self):
        grades = {"A": 1, "B": 1}
        self.assertEqual(reciprocal_rank_at_k(["X", "Y", "B", "A"], grades, 4), 1 / 3)
        self.assertEqual(reciprocal_rank_at_k(["X", "Y"], grades, 2), 0)
        self.assertEqual(reciprocal_rank_at_k(["A"], grades, 5), 1)

    def test_ndcg_is_graduated_and_discounted(self):
        grades = {"A": 3, "B": 2, "C": 1}
        self.assertAlmostEqual(ndcg_at_k(["A", "B", "C"], grades, 3), 1)
        self.assertLess(ndcg_at_k(["C", "B", "A"], grades, 3), 1)
        dcg = 7 / math.log2(2) + 3 / math.log2(3)
        ideal = dcg
        self.assertAlmostEqual(ndcg_at_k(["A", "B"], grades, 2), dcg / ideal)

    def test_no_gold_positives_is_undefined_not_perfect(self):
        self.assertIsNone(recall_at_k(["X"], {}, 5))
        self.assertIsNone(reciprocal_rank_at_k(["X"], {}, 5))
        self.assertIsNone(ndcg_at_k(["X"], {}, 5))
        self.assertEqual(precision_at_k(["X"], {}, 5), 0)

    def test_invalid_k_and_duplicate_results_fail(self):
        with self.assertRaises(ValueError):
            recall_at_k(["A"], {"A": 1}, 0)
        with self.assertRaises(ValueError):
            precision_at_k(["A", "A"], {"A": 1}, 2)
        with self.assertRaises(ValueError):
            ndcg_at_k(["A"], {"A": 4}, 1)

    def test_fact_coverage_separate_from_document_relevance(self):
        docs = load_documents()
        _, cases = load_cases(docs)
        case = cases["refund_multi"]
        self.assertEqual(evidence_coverage(case, ["refund_deadline_v2"]), 1 / 3)
        self.assertEqual(evidence_coverage(
            case, ["refund_deadline_v2", "refund_conditions", "refund_exceptions"]
        ), 1)
        self.assertIsNone(evidence_coverage(cases["unanswerable_salary"], []))

    def test_gold_labels_must_only_reference_current_scoped_documents(self):
        docs = load_documents()
        _, cases = load_cases(docs)
        self.assertEqual(cases["refund_multi"].relevance_grades["refund_deadline_v2"], 3)
        self.assertNotIn("tenant_b_confidential", cases["refund_multi"].relevance_grades)
        from dataset import is_current
        self.assertFalse(is_current(docs["refund_deadline_v1"], "2026-11-10"))
        self.assertTrue(is_current(docs["refund_deadline_v2"], "2026-11-10"))


if __name__ == "__main__":
    unittest.main()
