"""Math, tokenization and inverted index behavior."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer import tokenize
from bm25 import BM25Index
from fixtures import load_corpus
from metadata import canonical_chunk, normalize_filters, normalize_metadata


def chunk(id_: str, text: str):
    return canonical_chunk(id_, text, {
        "tenant_id": "tenant_a", "company_id": "NONE", "period": "none",
        "segment": "general", "document_type": "guide", "source": "test.md",
    })


class BM25Tests(unittest.TestCase):
    def test_analyzer_keeps_error_code_as_one_token(self):
        terms = tokenize("How to fix ERR_AUTH_403 in vLLM?")
        self.assertIn("err_auth_403", terms)
        self.assertIn("vllm", terms)
        self.assertNotIn("how", terms)

    def test_idf_is_larger_for_rarer_term(self):
        index = BM25Index([
            chunk("a", "cuda memory"),
            chunk("b", "memory"),
            chunk("c", "memory"),
        ])
        self.assertGreater(index.idf["cuda"], index.idf["memory"])
        self.assertEqual(len(index.postings["cuda"]), 1)

    def test_tf_saturation(self):
        index = BM25Index([
            chunk("a", "error"),
            chunk("b", "error error"),
            chunk("c", "error error error"),
        ], b=0)
        hits = index.search("error", tenant_id="tenant_a")
        scores = {h.chunk.id: h.score for h in hits}
        self.assertGreater(scores["b"], scores["a"])
        self.assertLess(scores["c"] - scores["b"], scores["b"] - scores["a"])

    def test_inverted_index_retrieves_matching_terms_only(self):
        index = BM25Index(load_corpus())
        hits = index.search("ERR_AUTH_403", tenant_id="tenant_a")
        self.assertEqual([h.chunk.id for h in hits], ["auth_403"])

    def test_tenant_filter_is_mandatory_and_not_overridable(self):
        index = BM25Index(load_corpus())
        self.assertFalse(index.search("refund 45", tenant_id="tenant_a"))
        self.assertEqual(
            index.search("refund 45", tenant_id="tenant_b")[0].chunk.id,
            "tenant_b_refund",
        )
        with self.assertRaises(ValueError):
            index.search("refund", tenant_id="tenant_a", filters={"tenant_id": "tenant_b"})

    def test_metadata_is_normalized_at_both_ends(self):
        corpus = load_corpus()
        target = next(chunk for chunk in corpus if chunk.id == "fpt_sw_q2")
        self.assertEqual(target.metadata["company_id"], "FPT")
        self.assertEqual(target.metadata["period"], "2025-Q2")
        self.assertEqual(target.metadata["segment"], "software")
        self.assertEqual(normalize_filters({
            "company_id": "Công ty FPT", "period": "Q2/2025",
            "segment": "Phần mềm",
        }), {
            "company_id": "FPT", "period": "2025-Q2", "segment": "software",
        })
        index = BM25Index(corpus)
        result = index.search(
            "international software revenue", tenant_id="tenant_a",
            filters={"company_id": "FPT Corporation", "period": "Q2/2025"},
        )
        self.assertEqual(result[0].chunk.id, "fpt_sw_q2")

    def test_bad_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_metadata({"company_id": "FPT"})
        with self.assertRaises(ValueError):
            normalize_filters({"unknown_column": "abc"})
        with self.assertRaises(ValueError):
            normalize_filters({"period": "invalid date"})

    def test_invalid_query_and_parameters(self):
        index = BM25Index([chunk("a", "hello world")])
        for query in ("", " " * 3, "x" * 1001):
            with self.assertRaises(ValueError):
                index.search(query, tenant_id="tenant_a")
        with self.assertRaises(ValueError):
            index.search("hello", tenant_id="tenant_a", top_k=0)
        with self.assertRaises(ValueError):
            BM25Index([chunk("a", "hi")], k1=0)
        with self.assertRaises(ValueError):
            BM25Index([chunk("a", "hi")], b=2)


if __name__ == "__main__":
    unittest.main()
