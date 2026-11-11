"""Candidate budgets, MMR, coverage and reranker correctness."""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Hit
from corpus import Corpus
from retrieval import Retriever
from reranking import LexicalReranker, LocalCrossEncoder, rerank
from selection import select_candidates


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        corpus = Corpus.load()
        by_parent = {}
        for child in corpus.children:
            by_parent.setdefault(child.parent_id, child)
        self.by_parent = by_parent
        self.hits = [
            Hit(by_parent[id], score=1.0 / rank, rank=rank, stage="hybrid")
            for rank, id in enumerate((
                "refund_deadline_1", "refund_deadline_2",
                "refund_eligibility", "refund_exceptions", "refund_process",
            ), 1)
        ]

    def test_topk_can_lose_required_aspects(self):
        selected = select_candidates(self.hits, budget=2, strategy="topk")
        self.assertEqual([h.child.aspect for h in selected], ["deadline", "deadline"])

    def test_coverage_keeps_requested_evidence_aspects(self):
        selected = select_candidates(
            self.hits, budget=3, strategy="coverage",
            required_aspects=("deadline", "conditions", "exceptions"),
        )
        self.assertEqual(
            [h.child.aspect for h in selected],
            ["deadline", "conditions", "exceptions"],
        )
        self.assertEqual(len(selected), 3)

    def test_mmr_promotes_diversity_against_duplicate_text(self):
        first, second = self.hits[:2]
        repeated = replace(second, child=replace(second.child, text=first.child.text))
        candidates = [first, repeated, self.hits[2]]
        topk = select_candidates(candidates, budget=2, strategy="topk")
        mmr = select_candidates(candidates, budget=2, strategy="mmr", mmr_lambda=0.05)
        self.assertEqual(topk[1].child.id, repeated.child.id)
        self.assertNotEqual(mmr[1].child.id, repeated.child.id)

    def test_candidate_budget_bounds(self):
        with self.assertRaises(ValueError):
            select_candidates(self.hits, budget=0)
        with self.assertRaises(ValueError):
            select_candidates(self.hits, budget=2, strategy="something")
        with self.assertRaises(ValueError):
            select_candidates(self.hits, budget=2, mmr_lambda=1.5)
        with self.assertRaises(ValueError):
            select_candidates([self.hits[0], self.hits[0]], budget=2)

    def test_reranker_scores_are_not_retrieval_scores(self):
        hit = Hit(self.hits[0].child, score=1000, rank=1, stage="bm25")
        result = rerank("calendar days", [hit], LexicalReranker())
        self.assertEqual(result[0].retrieval_score, 1000)
        self.assertLessEqual(result[0].score, 1)
        self.assertEqual(result[0].rank, 1)

    def test_reranker_rejects_nonfinite_and_wrong_count(self):
        fake = MagicMock()
        fake.scores.return_value = [float("nan")]
        with self.assertRaises(ValueError):
            rerank("refund", self.hits[:1], fake)
        fake.scores.return_value = [1.0, 0.5]
        with self.assertRaises(ValueError):
            rerank("refund", self.hits[:1], fake)

    def test_real_cross_encoder_requires_local_directory(self):
        with self.assertRaises(ValueError):
            LocalCrossEncoder("/nonexistent/path/not/a/model")


if __name__ == "__main__":
    unittest.main()
