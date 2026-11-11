"""Optional real libraries are tested if installed, with no external network."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import synthetic_vectors
from exact import ExactKNN
from metadata import normalize_metadata


class OptionalLibraryTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy not installed")
    def test_numpy_exact_matches_pure_python(self):
        from numpy_adapter import NumPyExactKNN

        records, queries = synthetic_vectors(count=50, dimensions=8)
        for metric in ("cosine", "dot", "l2"):
            numpy_index = NumPyExactKNN(records, metric)
            standard = ExactKNN(records, metric)
            for query in queries[:3]:
                a, _ = numpy_index.search(query, tenant_id="tenant_a", top_k=5)
                b, _ = standard.search(query, tenant_id="tenant_a", top_k=5)
                self.assertEqual(
                    [hit.record.id for hit in a],
                    [hit.record.id for hit in b],
                )

    @unittest.skipUnless(
        importlib.util.find_spec("hnswlib") and importlib.util.find_spec("numpy"),
        "hnswlib or numpy not installed",
    )
    def test_real_hnswlib_search_when_available(self):
        from hnswlib_adapter import HnswlibIndex

        records, queries = synthetic_vectors(count=80, dimensions=8)
        index = HnswlibIndex(records, m=8, ef_construction=32)
        hits, _ = index.search(
            queries[0], tenant_id="tenant_a", top_k=5, ef_search=32,
        )
        self.assertEqual(len(hits), 5)
        self.assertTrue(all(hit.record.metadata["tenant_id"] == "tenant_a" for hit in hits))

    @unittest.skipUnless(importlib.util.find_spec("qdrant_client"), "Qdrant client not installed")
    def test_real_qdrant_local_filter_and_update_when_available(self):
        from qdrant_adapter import QdrantLocalIndex
        from contracts import VectorRecord

        records, queries = synthetic_vectors(count=40, dimensions=8)
        index = QdrantLocalIndex(records)
        try:
            hits, _ = index.search(
                queries[0], tenant_id="tenant_a", filters={"company_id": "FPT"},
                top_k=5,
            )
            self.assertTrue(hits)
            self.assertTrue(all(
                h.record.metadata["company_id"] == "FPT" for h in hits
            ))
            record = records[0]
            changed = VectorRecord(
                record.id, record.vector, "updated",
                record.metadata, record.embedding_model,
            )
            index.upsert_many([changed])
            self.assertEqual(index.records[record.id].text, "updated")
            index.delete(record.id)
            self.assertNotIn(record.id, index.records)
        finally:
            index.close()


if __name__ == "__main__":
    unittest.main()
