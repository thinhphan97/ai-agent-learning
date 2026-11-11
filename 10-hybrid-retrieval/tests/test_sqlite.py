"""Persistent FTS5, metadata isolation, update consistency and injection safety."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import load_corpus
from metadata import canonical_chunk
from storage import SQLiteFTSIndex


def sample(id_: str, text: str):
    return canonical_chunk(id_, text, {
        "tenant_id": "tenant_a", "company_id": "NONE", "period": "none",
        "segment": "general", "document_type": "guide", "source": "test.md",
    })


class SQLiteFTSTests(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteFTSIndex()
        self.db.upsert_many(load_corpus())

    def tearDown(self):
        self.db.close()

    def test_fts5_ranks_and_filters(self):
        hits = self.db.search(
            "FPT software revenue", tenant_id="tenant_a",
            filters={"period": "Q2/2025", "segment": "software"},
        )
        self.assertEqual([h.chunk.id for h in hits], ["fpt_sw_q2"])
        self.assertEqual(hits[0].chunk.metadata["period"], "2025-Q2")

    def test_exact_error_code_search(self):
        hits = self.db.search("ERR_AUTH_403", tenant_id="tenant_a")
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.id, "auth_403")

    def test_fts_score_is_transformed_for_descending_rank(self):
        hits = self.db.search("FPT revenue Q2", tenant_id="tenant_a")
        self.assertGreater(len(hits), 1)
        self.assertEqual(hits, sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id)))

    def test_query_with_quotes_cannot_inject_sql_or_fts_operators(self):
        hits = self.db.search(
            '''" OR "refund" OR tenant_id:tenant_b --''', tenant_id="tenant_a",
        )
        self.assertNotIn("tenant_b_refund", [h.chunk.id for h in hits])

    def test_disallowed_filter_is_rejected(self):
        with self.assertRaises(ValueError):
            self.db.search("refund", tenant_id="tenant_a", filters={"tenant_id": "tenant_b"})

    def test_tenant_isolation(self):
        self.assertFalse(self.db.search(
            "refund for subscriptions 45", tenant_id="tenant_a",
            filters={"segment": "commerce"},
        ))
        hits = self.db.search(
            "refund for subscriptions 45", tenant_id="tenant_b",
            filters={"segment": "commerce"},
        )
        self.assertEqual([hit.chunk.id for hit in hits], ["tenant_b_refund"])

    def test_zero_results_does_not_relax_filters(self):
        self.assertEqual(
            self.db.search(
                "revenue", tenant_id="tenant_a",
                filters={"company_id": "UNLISTED"},
            ),
            [],
        )

    def test_replace_updates_fts_atomically(self):
        self.db.upsert_many([sample("replace_me", "olduncommonword exists")])
        self.assertEqual(
            self.db.search("olduncommonword", tenant_id="tenant_a")[0].chunk.id,
            "replace_me",
        )
        self.db.upsert_many([sample("replace_me", "newuncommonword exists")])
        self.assertFalse(self.db.search("olduncommonword", tenant_id="tenant_a"))
        self.assertEqual(
            self.db.search("newuncommonword", tenant_id="tenant_a")[0].chunk.id,
            "replace_me",
        )
        self.db.delete("replace_me")
        self.assertFalse(self.db.search("newuncommonword", tenant_id="tenant_a"))

    def test_persistence_between_connections(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.sqlite3"
            with SQLiteFTSIndex(path) as store:
                store.upsert_many([sample("durable", "persistencetest keyword")])
            with SQLiteFTSIndex(path) as reopened:
                hits = reopened.search("persistencetest", tenant_id="tenant_a")
                self.assertEqual(hits[0].chunk.id, "durable")

    def test_bad_batch_does_not_partially_update(self):
        good = sample("valid", "validunique")
        bad = sample("invalid", "badunique")
        from contracts import Chunk
        bad = Chunk(bad.id, bad.text, bad.contextual_text, {"tenant_id": "tenant_a"})
        before = self.db.count()
        with self.assertRaises(ValueError):
            self.db.upsert_many([good, bad])
        self.assertEqual(self.db.count(), before)
        self.assertFalse(self.db.search("validunique", tenant_id="tenant_a"))


if __name__ == "__main__":
    unittest.main()
