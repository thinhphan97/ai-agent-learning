"""Contextual retrieval indexing, tenant scope and source provenance."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextualization import MetadataContextualizer, contextualize
from contracts import Child
from corpus import Corpus
from retrieval import Retriever, tokenize


class ContextualRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.raw = Corpus.load()

    def test_original_chunk_unchanged_and_note_added(self):
        before = self.raw.children[0]
        indexed = contextualize(self.raw, "metadata")
        after = indexed.children[0]
        self.assertEqual(before.text, after.text)
        self.assertEqual(before.source, after.source)
        self.assertEqual(before.parent_id, after.parent_id)
        self.assertIn("Retail Refund Policy", after.contextual_note)
        self.assertIn("Request Deadline", after.indexed_text)
        self.assertTrue(after.indexed_text.endswith(before.text))
        self.assertEqual(before.indexed_text, before.text)

    def test_metadata_contextualization_improves_lexical_match_for_orphan_excerpt(self):
        query = "Retail Refund Policy Request Deadline"
        plain = Retriever(contextualize(self.raw, "none")).search(
            query, tenant_id="tenant_a", mode="bm25", top_k=50,
        )
        enhanced = Retriever(contextualize(self.raw, "metadata")).search(
            query, tenant_id="tenant_a", mode="bm25", top_k=50,
        )
        plain_ids = {h.child.id for h in plain}
        enhanced_ids = {h.child.id for h in enhanced}
        child_id = "refund_deadline_1:c1"
        self.assertNotIn(child_id, plain_ids)
        self.assertIn(child_id, enhanced_ids)

    def test_tenant_isolation_in_both_retrievers(self):
        corpus = contextualize(self.raw, "metadata")
        for mode in ("bm25", "dense", "hybrid"):
            tenant_a = Retriever(corpus).search(
                "refund 45 days confidential", tenant_id="tenant_a",
                mode=mode, top_k=50,
            )
            self.assertTrue(all(h.child.tenant_id == "tenant_a" for h in tenant_a))
            self.assertFalse(any(
                h.child.parent_id == "tenant_b_refund" for h in tenant_a
            ))

    def test_parent_child_provenance_and_scope(self):
        child = next(c for c in self.raw.children if c.parent_id == "refund_eligibility")
        self.assertIn("unused", self.raw.get_parent(child, "tenant_a").text)
        with self.assertRaises(PermissionError):
            self.raw.get_parent(child, "tenant_b")
        with self.assertRaises(ValueError):
            Corpus(list(self.raw.parents.values()), [
                replace(child, version="fake_version"),
            ])
        with self.assertRaises(ValueError):
            Corpus(list(self.raw.parents.values()), [
                replace(child, text="Invented evidence not present in parent"),
            ])

    def test_llm_mode_requires_trusted_scope_before_provider_creation(self):
        with self.assertRaises(ValueError):
            contextualize(self.raw, "llm")
        class FakeGenerator:
            def __init__(self):
                self.seen = []
            def describe(self, parent, child):
                self.seen.append(child.tenant_id)
                return f"grounded note for {child.id}"
        fake = FakeGenerator()
        with patch("contextualization.LLMContextualizer", return_value=fake):
            corpus = contextualize(self.raw, "llm", tenant_id="tenant_a")
        self.assertTrue(fake.seen)
        self.assertEqual(set(fake.seen), {"tenant_a"})
        for child in corpus.children:
            if child.tenant_id == "tenant_b":
                self.assertFalse(child.contextual_note)
            else:
                self.assertTrue(child.contextual_note)

    def test_unknown_contextualization_rejected(self):
        with self.assertRaises(ValueError):
            contextualize(self.raw, "magical")
        with self.assertRaises(ValueError):
            Corpus([], [])

    def test_real_api_dense_mode_filters_tenant_before_provider(self):
        from retrieval import Retriever, _embedding
        class FakeEmbeddingProvider:
            def __init__(self):
                self.seen = []
            def embed(self, texts):
                self.seen.extend(texts)
                return [_embedding(text) for text in texts]

        fake = FakeEmbeddingProvider()
        with patch("retrieval.OpenAIEmbedder", return_value=fake):
            retriever = Retriever(
                contextualize(self.raw, "metadata"), dense_provider="api"
            )
        hits = retriever.search(
            "refund subscription 45 days", tenant_id="tenant_a",
            mode="dense", top_k=20,
        )
        self.assertTrue(hits)
        self.assertEqual(set(h.child.tenant_id for h in hits), {"tenant_a"})
        self.assertFalse(any("Other Tenant Refund Terms" in text
                             for text in fake.seen))
        self.assertFalse(any("Tenant B" in text for text in fake.seen))

    def test_external_embedder_requires_configured_model(self):
        from retrieval import OpenAIEmbedder
        from unittest.mock import patch as mockpatch
        with mockpatch.dict("os.environ", {"EMBEDDING_MODEL": ""}):
            with self.assertRaises(ValueError):
                OpenAIEmbedder()

    def test_tokenize_keeps_technical_identifier(self):
        self.assertIn("err_auth_403", tokenize("ERR_AUTH_403 admin"))


if __name__ == "__main__":
    unittest.main()