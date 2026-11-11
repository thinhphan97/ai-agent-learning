"""Full offline retrieval/generation flow and refusal of unsupported citations."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documents import chunk_documents, load_documents
from evaluation import CASES, evaluate
from generation import ExtractiveMockGenerator
from pipeline import RAGPipeline
from retrieval import TfidfRetriever


ROOT = Path(__file__).resolve().parents[1]


class FixedGenerator:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, question, hits):
        self.calls += 1
        return self.text


class PipelineTests(unittest.TestCase):
    def setUp(self):
        chunks = chunk_documents(
            load_documents(ROOT / "data" / "documents"),
            chunk_size=40, overlap=10,
        )
        self.retriever = TfidfRetriever(chunks)
        self.pipeline = RAGPipeline(self.retriever, ExtractiveMockGenerator())

    def test_expected_answer_has_real_source_citation(self):
        result = self.pipeline.answer("How many days can I return unopened hardware?")
        self.assertEqual(result.status, "answered")
        self.assertIn("30 days", result.text)
        self.assertIn("returns:0", result.citations)
        self.assertTrue(all("[" + source + "]" in result.text for source in result.citations))
        self.assertEqual(result.reason, "CITATIONS_VALID")

    def test_missing_information_abstains_without_generator_call(self):
        fake = FixedGenerator("Fabricated answer [returns:0]")
        pipeline = RAGPipeline(self.retriever, fake)
        result = pipeline.answer("What is the employee cafeteria menu?")
        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.reason, "NO_RETRIEVAL_EVIDENCE")
        self.assertEqual(result.citations, ())
        self.assertEqual(fake.calls, 0)

    def test_real_source_but_missing_citation_is_rejected(self):
        fake = FixedGenerator("30 days")
        pipeline = RAGPipeline(self.retriever, fake)
        result = pipeline.answer("How many days can I return unopened hardware?")
        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.reason, "INVALID_CITATIONS")

    def test_hallucinated_citation_is_rejected(self):
        fake = FixedGenerator("30 days [fiction:0]")
        result = RAGPipeline(self.retriever, fake).answer(
            "How many days can I return unopened hardware?"
        )
        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.reason, "INVALID_CITATIONS")

    def test_known_citation_and_unknown_citation_combination_rejected(self):
        fake = FixedGenerator("30 days [returns:0] [imaginary:999]")
        result = RAGPipeline(self.retriever, fake).answer(
            "How many days can I return unopened hardware?"
        )
        self.assertEqual(result.reason, "INVALID_CITATIONS")

    def test_explicit_dont_know_is_abstention(self):
        fake = FixedGenerator("I don't know based on the documents.")
        result = RAGPipeline(self.retriever, fake).answer(
            "How many days can I return unopened hardware?"
        )
        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.reason, "INSUFFICIENT_CONTEXT")

    def test_generator_exception_is_reported_without_provider_detail(self):
        class BrokenGenerator:
            def generate(self, question, hits):
                raise RuntimeError("secret provider token")
        result = RAGPipeline(self.retriever, BrokenGenerator()).answer(
            "How many days can I return unopened hardware?"
        )
        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.reason, "GENERATION_FAILED")
        self.assertNotIn("secret", result.text)

    def test_context_budget_skips_oversize_chunk(self):
        result = RAGPipeline(
            self.retriever, ExtractiveMockGenerator(), max_context_chars=100
        ).answer("How many days can I return unopened hardware?")
        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.reason, "NO_RETRIEVAL_EVIDENCE")

    def test_reject_invalid_context_budget(self):
        for limit in (0, -1, 99, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                RAGPipeline(self.retriever, ExtractiveMockGenerator(), max_context_chars=limit)

    def test_evaluation_is_separate_and_has_a_negative_case(self):
        self.assertEqual(len(CASES), 8)
        self.assertTrue(any(case.expected_document is None for case in CASES))
        summary, outcomes = evaluate(self.pipeline.answer)
        self.assertEqual(summary.questions, len(CASES))
        self.assertEqual(summary.negative_abstention_accuracy, 1)
        self.assertEqual(len(outcomes), 8)
        self.assertTrue(all("expected_document" in outcome for outcome in outcomes))

    def test_empty_suite_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.pipeline.answer, ())


if __name__ == "__main__":
    unittest.main()
