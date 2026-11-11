"""Token budget, parent expansion, deduplication, citation IDs and coverage."""
from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import RerankedHit
from context_builder import ContextBuilder, WhitespaceTokenCounter
from corpus import Corpus
from validation import aspect_coverage, check_citations


def hits(*parent_ids):
    corpus = Corpus.load()
    by_parent = {}
    for child in corpus.children:
        by_parent.setdefault(child.parent_id, child)
    ranked = [
        RerankedHit(by_parent[id], retrieval_rank=i, retrieval_score=1 / i,
                    score=1 / i, rank=i)
        for i, id in enumerate(parent_ids, 1)
    ]
    return corpus, ranked


class BuilderTests(unittest.TestCase):
    def test_coverage_prioritizes_all_three_aspects(self):
        corpus, ranked = hits(
            "refund_deadline_1", "refund_deadline_2",
            "refund_eligibility", "refund_exceptions",
        )
        context = ContextBuilder(corpus).build(
            ranked, tenant_id="tenant_a", max_evidence=3,
            token_budget=200, strategy="coverage",
            required_aspects=("deadline", "conditions", "exceptions"),
        )
        self.assertEqual([e.aspect for e in context.evidence],
                         ["deadline", "conditions", "exceptions"])
        self.assertEqual(context.missing_aspects, ())
        self.assertEqual(aspect_coverage(context, ("deadline", "conditions", "exceptions")), 1)

    def test_ranked_selection_can_lack_aspects(self):
        corpus, ranked = hits(
            "refund_deadline_1", "refund_deadline_2",
            "refund_eligibility", "refund_exceptions",
        )
        context = ContextBuilder(corpus).build(
            ranked, tenant_id="tenant_a", max_evidence=2,
            token_budget=200, strategy="ranked",
            required_aspects=("deadline", "conditions", "exceptions"),
        )
        self.assertEqual(context.missing_aspects, ("conditions", "exceptions"))

    def test_actual_formatted_content_in_budget(self):
        corpus, ranked = hits("refund_deadline_1", "refund_eligibility", "refund_exceptions")
        counter = WhitespaceTokenCounter()
        context = ContextBuilder(corpus, counter).build(
            ranked, tenant_id="tenant_a", token_budget=85,
            max_evidence=5,
            required_aspects=("deadline", "conditions", "exceptions"),
        )
        self.assertLessEqual(context.tokens_used, context.token_budget)
        self.assertEqual(counter.count(context.text), context.tokens_used)
        self.assertTrue(context.dropped)
        self.assertNotIn("Contextual Description:", context.text)

    def test_parent_expansion_uses_original_parent_not_context_note(self):
        corpus, ranked = hits("refund_deadline_1")
        corpus_with_note = corpus.with_context({ranked[0].child.id: "FABRICATED NOTE"})
        ranked = [replace(ranked[0], child=corpus_with_note.children[0])]
        context = ContextBuilder(corpus_with_note).build(
            ranked, tenant_id="tenant_a", token_budget=180,
            expansion="parent",
        )
        self.assertEqual(context.evidence[0].content, corpus_with_note.parents[
            "refund_deadline_1"].text)
        self.assertNotIn("FABRICATED NOTE", context.text)
        self.assertIn("physical goods purchased", context.text)

    def test_parent_expansion_falls_back_to_shorter_child(self):
        corpus, ranked = hits("refund_deadline_1")
        parent = ContextBuilder(corpus).build(
            ranked, tenant_id="tenant_a", token_budget=200, expansion="parent",
        )
        child = ContextBuilder(corpus).build(
            ranked, tenant_id="tenant_a", token_budget=200, expansion="child",
        )
        self.assertGreater(parent.tokens_used, child.tokens_used)
        middle = (parent.tokens_used + child.tokens_used) // 2
        fallback = ContextBuilder(corpus).build(
            ranked, tenant_id="tenant_a", token_budget=middle,
            expansion="parent",
        )
        self.assertEqual(fallback.evidence[0].content, ranked[0].child.text)

    def test_parent_not_duplicated(self):
        corpus = Corpus.load()
        children = [x for x in corpus.children if x.parent_id == "refund_eligibility"]
        ranked = [
            RerankedHit(c, retrieval_rank=i, retrieval_score=1 / i,
                        score=1 / i, rank=i) for i, c in enumerate(children, 1)
        ]
        context = ContextBuilder(corpus).build(
            ranked, tenant_id="tenant_a", token_budget=250,
            expansion="parent",
        )
        self.assertEqual(len(context.evidence), 1)
        self.assertIn(("refund_eligibility:c2", "same_parent"), context.dropped)

    def test_cross_tenant_expansion_is_rejected(self):
        corpus, ranked = hits("refund_deadline_1")
        with self.assertRaises(PermissionError):
            ContextBuilder(corpus).build(
                ranked, tenant_id="tenant_b", token_budget=200,
                expansion="parent",
            )

    def test_no_budget_means_missing_evidence_and_no_truncation(self):
        corpus, ranked = hits("refund_deadline_1")
        result = ContextBuilder(corpus).build(
            ranked, tenant_id="tenant_a",
            token_budget=1, required_aspects=("deadline",),
        )
        self.assertEqual(result.text, "")
        self.assertEqual(result.missing_aspects, ("deadline",))
        self.assertEqual(result.dropped[0][1], "over_budget")

    def test_fallback_does_not_suppress_other_children_in_same_parent(self):
        from contracts import Parent, Child
        parent = Parent(
            id="p", source="same-source.md", section="Terms",
            title="Terms", version="v1", tenant_id="tenant_a",
            aspect="conditions",
            text=("First rule is mandatory. Second rule also applies. "
                  + "Additional explanatory wording " * 70),
        )
        first = Child(
            id="c1", parent_id="p", source=parent.source,
            section=parent.section, version=parent.version,
            tenant_id=parent.tenant_id, aspect=parent.aspect,
            text="First rule is mandatory.",
        )
        second = replace(first, id="c2", text="Second rule also applies.")
        corpus = Corpus([parent], [first, second])
        ranked = [
            RerankedHit(first, retrieval_rank=1, retrieval_score=1,
                        score=1, rank=1),
            RerankedHit(second, retrieval_rank=2, retrieval_score=0.5,
                        score=0.5, rank=2),
        ]
        context = ContextBuilder(corpus).build(
            ranked, tenant_id="tenant_a", token_budget=100,
            expansion="parent", max_evidence=2,
        )
        self.assertEqual([e.child_id for e in context.evidence], ["c1", "c2"])
        self.assertNotIn(("c2", "same_parent"), context.dropped)

    @unittest.skipUnless(
        os.getenv("RUN_TIKTOKEN_INTEGRATION") == "1",
        "tiktoken may download an encoding file on first use; run explicitly",
    )
    def test_optional_tiktoken_counts_actual_selected_encoding(self):
        import importlib.util
        if importlib.util.find_spec("tiktoken") is None:
            self.skipTest("Optional tiktoken not installed")
        from context_builder import TikTokenCounter

        counter = TikTokenCounter()
        corpus, ranked = hits("refund_deadline_1")
        context = ContextBuilder(corpus, counter).build(
            ranked, tenant_id="tenant_a", token_budget=120,
        )
        self.assertEqual(counter.count(context.text), context.tokens_used)
        self.assertLessEqual(context.tokens_used, context.token_budget)
        self.assertEqual(context.token_counter, "tiktoken:cl100k_base")

    def test_citation_validation_only_checks_id_existence(self):
        corpus, ranked = hits("refund_deadline_1")
        context = ContextBuilder(corpus).build(
            ranked, tenant_id="tenant_a", token_budget=200,
        )
        valid = check_citations("The deadline is 30 days [E1].", context)
        self.assertTrue(valid["citation_ids_valid"])
        self.assertFalse(valid["semantic_faithfulness_verified"])
        bad = check_citations("False claim [E123].", context)
        self.assertFalse(bad["citation_ids_valid"])
        self.assertEqual(bad["unknown_citations"], ["E123"])
        self.assertFalse(check_citations("Unsupported answer.", context)["citation_ids_valid"])


if __name__ == "__main__":
    unittest.main()