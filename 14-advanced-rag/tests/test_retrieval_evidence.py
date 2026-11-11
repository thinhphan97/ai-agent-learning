"""Search, fusion, evidence validation, safe arithmetic, no unsupported explanations."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Budgets, Request, SearchHit
from controller import AdvancedRAG
from corpus import Corpus
from evidence import EvidenceBuilder, grounded_answer, supports
from query_analysis import analyze
from retrieval import LocalRetriever, rrf_fuse


QUESTION = (
    "Compare FPT international software revenue in Q1 and Q2 2025, "
    "calculate growth and explain the main reasons for change."
)


class RetrievalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.load()
        self.request = Request(QUESTION, "tenant_a", "2026-11-10", ("docs",))
        self.retriever = LocalRetriever(self.corpus)

    def test_bm25_search_with_source_and_date_filters(self):
        hits = self.retriever.search(
            self.request, query="FPT Q1/2025 quarterly revenue",
            source="docs", aspect="q1_revenue", limit=5,
        )
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, "fpt_q1")
        self.assertFalse(any(h.doc_id in {"fpt_old", "tenant_b_secret"} for h in hits))

    def test_rrf_fusion_is_deterministic_and_deduplicates(self):
        first = self.retriever.search(
            self.request, query="FPT Q1/2025 quarterly revenue",
            source="docs", aspect="q1_revenue", limit=5,
        )
        second = self.retriever.search(
            self.request, query="FPT Q2/2025 quarterly revenue",
            source="docs", aspect="q2_revenue", limit=5,
        )
        fused = rrf_fuse([first, second])
        self.assertEqual(len({h.doc_id for h in fused}), len(fused))
        self.assertEqual([h.doc_id for h in fused], [
            h.doc_id for h in rrf_fuse([first, second])
        ])
        self.assertEqual(fused[0].rank, 1)
        with self.assertRaises(ValueError):
            rrf_fuse([first + first], k=60)

    def test_original_query_2026_rejects_2025_evidence(self):
        text = self.corpus.documents["fpt_q1"].text
        self.assertTrue(supports("q1_revenue", text, QUESTION))
        self.assertFalse(supports(
            "q1_revenue", text, "FPT international software revenue Q1/2026"
        ))
        result = AdvancedRAG().run(Request(
            "Compare FPT international software revenue Q1 and Q2 2026",
            "tenant_a", "2026-11-10", ("docs",)
        ), strategy="planned")
        self.assertEqual(result["answer"]["status"], "abstained")
        self.assertIsNone(result["answer"]["calculation"])

    def test_cannot_reuse_customer_1_evidence_for_customer_2(self):
        result = AdvancedRAG().run(Request(
            "What is paid order revenue for customer 2?",
            "tenant_a", "2026-11-10", ("sql",)
        ), strategy="planned")
        self.assertEqual(result["answer"]["status"], "abstained")
        self.assertFalse(result["evidence"])

    def test_verified_calculation_and_original_doc_citations(self):
        result = AdvancedRAG().run(
            self.request, strategy="adaptive", scenario="delayed-drivers"
        )
        self.assertEqual(result["answer"]["calculation"], 20)
        self.assertIn("20.00% (calculated from [E1], [E2])", result["answer"]["answer"])
        self.assertIn("contracts and higher delivery volumes", result["answer"]["answer"])
        self.assertEqual([e["doc_id"] for e in result["evidence"]],
                         ["fpt_q1", "fpt_q2", "fpt_drivers"])
        self.assertNotIn("Hypothetical", result["answer"]["answer"])

    def test_missing_driver_is_partial_without_inventing_a_cause(self):
        result = AdvancedRAG().run(
            self.request, strategy="adaptive", scenario="no-drivers"
        )
        self.assertEqual(result["answer"]["status"], "partial")
        self.assertEqual(result["missing_aspects"], ["drivers"])
        self.assertIn("No unsupported conclusions are inferred", result["answer"]["answer"])
        self.assertNotIn("contracts", result["answer"]["answer"])
        self.assertEqual(result["answer"]["calculation"], 20)

    def test_hyde_hypothetical_text_never_enters_evidence(self):
        result = AdvancedRAG().run(self.request, strategy="hyde")
        for e in result["evidence"]:
            doc = self.corpus.documents[e["doc_id"]]
            self.assertEqual(e["content"], doc.text)
            self.assertNotIn("hypothetical", e["content"].lower())


if __name__ == "__main__":
    unittest.main()
