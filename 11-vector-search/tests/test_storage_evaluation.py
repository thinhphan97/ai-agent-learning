"""Vector persistence, CRUD, benchmark correctness and provider invariants."""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import VectorRecord
from embeddings import HashingEmbedder
from evaluation import ann_recall_at_k, benchmark
from fixtures import load_documents
from storage import VectorStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.records = load_documents(HashingEmbedder())

    def test_fixture_is_canonical_and_embedding_is_deterministic(self):
        self.assertEqual(len(self.records), 10)
        self.assertEqual(self.records[0].metadata["company_id"], "FPT")
        self.assertEqual(self.records[0].metadata["period"], "2025-Q2")
        a, b = HashingEmbedder(), HashingEmbedder()
        self.assertEqual(a.embed(["CUDA memory"]), b.embed(["CUDA memory"]))

    def test_persistent_sqlite_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "vectors.sqlite3"
            with VectorStore(db) as store:
                store.upsert_many(self.records)
            with VectorStore(db) as store:
                recovered = store.all_records()
                self.assertEqual(recovered, sorted(self.records, key=lambda r: r.id))
                self.assertEqual(store.count(), len(self.records))

    def test_upsert_updates_vector_text_and_metadata(self):
        record = self.records[0]
        changed = VectorRecord(
            record.id,
            record.vector,
            "NEW TEXT",
            record.metadata,
            record.embedding_model,
        )
        with VectorStore() as store:
            store.upsert_many(self.records)
            store.upsert_many([changed])
            self.assertEqual(store.get(record.id).text, "NEW TEXT")
            self.assertTrue(store.delete(record.id))
            self.assertIsNone(store.get(record.id))
            self.assertFalse(store.delete(record.id))

    def test_invalid_update_rolls_back(self):
        valid = self.records[0]
        invalid = VectorRecord(
            "bad",
            valid.vector,
            "bad text",
            {"tenant_id": "tenant_a"},
            valid.embedding_model,
        )
        with VectorStore() as store:
            with self.assertRaises(ValueError):
                store.upsert_many([valid, invalid])
            self.assertEqual(store.count(), 0)

    def test_duplicate_ids_rejected(self):
        with VectorStore() as store:
            with self.assertRaises(ValueError):
                store.upsert_many([self.records[0], self.records[0]])

    def test_ann_recall_metric(self):
        self.assertAlmostEqual(
            ann_recall_at_k(["a", "b", "c", "d", "e"], ["a", "b", "x", "d", "y"]),
            0.6,
        )
        with self.assertRaises(ValueError):
            ann_recall_at_k([], [])

    def test_offline_benchmark_has_fixed_contract(self):
        result = benchmark(count=100, dim=8, k=3, ef_values=(3, 8, 16))
        self.assertEqual(result["vectors"], 100)
        self.assertEqual(result["queries"], 16)
        for item in result["ef_sweep"]:
            self.assertGreaterEqual(item["ann_recall_at_k"], 0)
            self.assertLessEqual(item["ann_recall_at_k"], 1)
        self.assertEqual(result["benchmark_type"], "ann_vs_exact_not_rag_quality")


if __name__ == "__main__":
    unittest.main()
