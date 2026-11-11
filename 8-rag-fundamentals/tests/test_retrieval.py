"""Sparse lexical retrieval and fake dense embeddings: exact data contracts."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documents import Chunk, chunk_documents, load_documents
from retrieval import DenseRetriever, TfidfRetriever, tokenize


ROOT = Path(__file__).resolve().parents[1]


class LexicalRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_documents(
            load_documents(ROOT / "data" / "documents"),
            chunk_size=40, overlap=10,
        )
        self.index = TfidfRetriever(self.chunks)

    def test_returns_policy_ranks_first(self):
        hits = self.index.retrieve(
            "How many days can I return unopened hardware?", top_k=3
        )
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.document_id, "returns")
        self.assertGreater(hits[0].score, 0)

    def test_support_and_security_ranks(self):
        for question, source in (
            ("When is the service desk open?", "support"),
            ("When do password reset links expire?", "security"),
            ("How much does expedited shipping cost?", "shipping"),
        ):
            with self.subTest(question=question):
                hits = self.index.retrieve(question, top_k=1)
                self.assertEqual(hits[0].chunk.document_id, source)

    def test_irrelevant_query_has_no_hits(self):
        hits = self.index.retrieve("What is the employee cafeteria menu?", top_k=3)
        self.assertEqual(hits, [])

    def test_score_threshold_can_force_abstention(self):
        hits = self.index.retrieve(
            "How many days can I return unopened hardware?",
            top_k=3, min_score=1,
        )
        self.assertEqual(hits, [])

    def test_top_k_and_stable_ordering(self):
        hits = self.index.retrieve("account support", top_k=2, min_score=0)
        self.assertLessEqual(len(hits), 2)
        self.assertEqual(hits, sorted(
            hits, key=lambda hit: (-hit.score, hit.chunk.id),
        ))

    def test_english_stopwords_are_not_keys(self):
        self.assertEqual(tokenize("What is THE return policy?"), ["return", "policy"])

    def test_unsupported_inputs_rejected(self):
        for k, score in ((0, 0.1), (True, 0.1), (11, 0.1), (2, -0.1), (2, math.nan), (2, 1.1)):
            with self.subTest(k=k, score=score), self.assertRaises(ValueError):
                self.index.retrieve("return policy", top_k=k, min_score=score)
        with self.assertRaises(ValueError):
            self.index.retrieve(" ")

    def test_cannot_build_empty_index(self):
        with self.assertRaises(ValueError):
            TfidfRetriever([])


class FakeEmbeddings:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


class DenseRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("shipping:0", "shipping", "Shipping", "Shipping policy", 0, 2),
            Chunk("returns:0", "returns", "Returns", "Return policy", 0, 2),
        ]
        self.embedder = FakeEmbeddings({
            "Shipping policy": [1, 0],
            "Return policy": [0, 1],
            "How do I return an item?": [0.1, 0.95],
            "How do I ship an item?": [0.95, 0.1],
        })

    def test_dense_cosine_search(self):
        index = DenseRetriever(self.chunks, self.embedder)
        hits = index.retrieve("How do I return an item?", top_k=1)
        self.assertEqual([h.chunk.id for h in hits], ["returns:0"])
        self.assertGreater(hits[0].score, 0.9)
        self.assertEqual(len(self.embedder.calls), 2)
        self.assertEqual(self.embedder.calls[0], ["Shipping policy", "Return policy"])

    def test_dimension_mismatch_rejected(self):
        embedder = FakeEmbeddings({
            "Shipping policy": [1, 0],
            "Return policy": [0, 1],
            "How do I return an item?": [1, 0, 0],
        })
        index = DenseRetriever(self.chunks, embedder)
        with self.assertRaises(ValueError):
            index.retrieve("How do I return an item?")

    def test_invalid_document_vectors_rejected(self):
        for bad in ([0, 0], [math.nan, 1], [1, float("inf")], ["x", 0], [True, 1]):
            with self.subTest(bad=bad):
                embedder = FakeEmbeddings({
                    "Shipping policy": bad,
                    "Return policy": [0, 1],
                })
                with self.assertRaises(ValueError):
                    DenseRetriever(self.chunks, embedder)

    def test_dimension_mismatch_between_documents(self):
        embedder = FakeEmbeddings({
            "Shipping policy": [1, 0],
            "Return policy": [0, 1, 0],
        })
        with self.assertRaises(ValueError):
            DenseRetriever(self.chunks, embedder)

    def test_bad_embed_count_rejected(self):
        class CountBad:
            def embed(self, texts):
                return [[1, 0]]
        with self.assertRaises(ValueError):
            DenseRetriever(self.chunks, CountBad())

    def test_empty_index_rejected(self):
        with self.assertRaises(ValueError):
            DenseRetriever([], self.embedder)


if __name__ == "__main__":
    unittest.main()
