"""Dense provider contracts, RRF, reranking, and trusted scope in all branches."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Hit
from dense import DenseIndex, HashingEmbedder
from fixtures import load_corpus
from fusion import reciprocal_rank_fusion
from pipeline import RetrievalPipeline
from reranker import LexicalOverlapReranker
from storage import SQLiteFTSIndex


class FixedEmbeddings:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.corpus = load_corpus()
        self.store = SQLiteFTSIndex()
        self.store.upsert_many(self.corpus)
        self.pipeline = RetrievalPipeline(
            self.store.all_chunks(), self.store, embedder=HashingEmbedder(),
        )

    def tearDown(self):
        self.store.close()

    def test_dense_returns_searchable_chunks(self):
        results = self.pipeline.search(
            "FPT software revenue", method="dense", tenant_id="tenant_a",
            filters={"period": "Q2/2025", "segment": "software"},
        )
        self.assertEqual([hit.chunk.id for hit in results], ["fpt_sw_q2"])

    def test_hybrid_preserves_source_rankings(self):
        results = self.pipeline.search(
            "ERR_AUTH_403 admin dashboard", method="hybrid", tenant_id="tenant_a",
        )
        self.assertEqual(results[0].chunk.id, "auth_403")
        self.assertEqual(results[0].method, "hybrid")
        self.assertIn("sqlite_fts5", results[0].sources)

    def test_hybrid_never_leaks_other_tenants(self):
        for method in ("bm25", "sqlite", "dense", "hybrid"):
            hits = self.pipeline.search(
                "refund for eligible subscriptions after 45 days",
                method=method, tenant_id="tenant_a", top_k=10, candidate_k=10,
            )
            self.assertNotIn("tenant_b_refund", [hit.chunk.id for hit in hits])

    def test_reranker_outputs_stable_hits_and_no_extra_documents(self):
        original = self.pipeline.search(
            "password reset links expire", method="hybrid",
            tenant_id="tenant_a", top_k=8, candidate_k=8,
        )
        ranked = self.pipeline.search(
            "password reset links expire", method="hybrid",
            tenant_id="tenant_a", top_k=5, candidate_k=8,
            reranker=LexicalOverlapReranker(),
        )
        self.assertTrue({h.chunk.id for h in ranked}.issubset({h.chunk.id for h in original}))
        self.assertEqual(ranked[0].chunk.id, "password_reset")
        self.assertEqual(ranked[0].method, "lexical_rerank")

    def test_rrf_deduplicates_and_uses_rank_not_raw_score(self):
        a, b = self.corpus[:2]
        ranks = reciprocal_rank_fusion([
            [Hit(a, 9000, "bm25"), Hit(b, 12, "bm25")],
            [Hit(b, 0.9, "dense"), Hit(a, 0.2, "dense")],
        ])
        self.assertEqual(len(ranks), 2)
        self.assertAlmostEqual(ranks[0].score, ranks[1].score)
        self.assertEqual(ranks[0].chunk.id, min(a.id, b.id))

    def test_duplicate_id_with_inconsistent_payload_is_rejected(self):
        first = self.corpus[0]
        from contracts import Chunk
        changed = Chunk(first.id, "different", "different", first.metadata)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([
                [Hit(first, 1.0, "lexical")],
                [Hit(changed, 1.0, "dense")],
            ])

    def test_bad_embedding_dimensions_rejected(self):
        class WrongDim:
            def embed(self, texts):
                return [[1.0] if i % 2 else [1.0, 2.0] for i, _ in enumerate(texts)]
        with self.assertRaises(ValueError):
            DenseIndex(self.corpus, WrongDim())

    def test_query_must_have_tenant_and_valid_candidate_budget(self):
        with self.assertRaises(ValueError):
            self.pipeline.search("refund", tenant_id="", method="hybrid")
        with self.assertRaises(ValueError):
            self.pipeline.search("refund", tenant_id="tenant_a", top_k=5, candidate_k=2)

    def test_model_real_vs_toy_contract_is_explicit(self):
        toy = HashingEmbedder()
        vector = toy.embed(["FPT software"])[0]
        self.assertEqual(len(vector), 256)
        self.assertTrue(any(x != 0 for x in vector))
        self.assertEqual(vector, toy.embed(["FPT software"])[0])


if __name__ == "__main__":
    unittest.main()