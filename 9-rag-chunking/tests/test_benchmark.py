"""Benchmark is evidence-coverage diagnostic, not an ungrounded quality metric."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import CASES, assess, retrieval_context
from contracts import ChunkSet, Document, make_chunk
from runner import run, sample_file
from shared import load_document


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.doc = load_document(sample_file("fixed"))

    def test_fixture_evidence_exists_in_document(self):
        for case in CASES:
            self.assertIn(case.required_quote, self.doc.text)

    def test_empty_or_wrong_evidence_not_invented(self):
        bad = ChunkSet("fixed", (make_chunk(
            Document("x", "The canteen serves no food."),
            "fixed", 0, "The canteen serves no food.",
        ),))
        report = assess(bad)
        self.assertEqual(report["evidence_coverage_at_3"], 0)

    def test_parent_expansion_respects_context_limit(self):
        collection = run("parent-child", self.doc, size=12, overlap=2)
        hits = retrieval_context(collection, CASES[0].question, top_k=3, max_words=1)
        self.assertEqual(hits, [])

    def test_fixed_strategy_evidence_retrieval(self):
        collection = run("fixed", self.doc, size=35, overlap=5)
        report = assess(collection)
        self.assertEqual(report["questions"], 4)
        self.assertTrue(0 <= report["evidence_coverage_at_3"] <= 1)
        self.assertEqual(len(report["cases"]), 4)

    def test_duplicate_parent_not_counted_twice(self):
        collection = run("parent-child", self.doc, size=10, overlap=2)
        hits = retrieval_context(collection, CASES[0].question, top_k=6)
        self.assertEqual(len({c.id for c in hits}), len(hits))

    def test_invalid_retrieval_budget(self):
        collection = run("fixed", self.doc)
        with self.assertRaises(ValueError):
            retrieval_context(collection, CASES[0].question, top_k=0)


if __name__ == "__main__":
    unittest.main()
