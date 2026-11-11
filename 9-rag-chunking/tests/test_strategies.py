"""Contract, source integrity and behavior tests for all 12 strategy families."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basic import fixed, recursive, sentence_or_paragraph
from code import python_ast
from contextual import contextual
from contracts import Document
from hierarchy import expand_to_parents, parent_child
from late import late
from layout import table_aware
from proposition import propositions
from providers import TopicLexicalEmbedder, WordContextToyEncoder
from raptor import ExtractiveSummarizer, raptor
from runner import STRATEGIES, run, sample_file
from semantic import semantic
from shared import load_document, sentence_units
from structure import markdown_sections


LAB = Path(__file__).resolve().parents[1]


class FixedTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document("x", " ".join(f"word{i}" for i in range(12)))

    def test_overlap_and_word_ranges(self):
        result = fixed(self.doc, size=5, overlap=2)
        self.assertEqual(
            [c.metadata["word_range"] for c in result.chunks],
            [[0, 5], [3, 8], [6, 11], [9, 12]],
        )
        for node in result.chunks:
            self.assertEqual(self.doc.text[node.start:node.end], node.text)

    def test_no_empty_terminal_chunk_on_exact_stride(self):
        self.assertEqual(len(fixed(Document("x", "a b c d"), size=2, overlap=0).chunks), 2)

    def test_short_document(self):
        self.assertEqual(len(fixed(Document("x", "Short text"), size=20).chunks), 1)

    def test_invalid_overlap(self):
        for size, overlap in ((0, 0), (5, 5), (5, -1), (True, 0), (5, True)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                fixed(self.doc, size=size, overlap=overlap)


class RecursiveTests(unittest.TestCase):
    def test_prefers_paragraph_boundaries(self):
        doc = Document("p", "Annual leave is twelve days.\n\nSecurity passwords expire promptly.")
        chunks = recursive(doc, size=7, overlap=0).chunks
        self.assertEqual(len(chunks), 2)
        self.assertIn("Annual leave", chunks[0].text)
        self.assertIn("Security passwords", chunks[1].text)

    def test_oversized_paragraph_splits(self):
        doc = Document("p", " ".join(f"word{i}" for i in range(60)))
        chunks = recursive(doc, size=10).chunks
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(len(c.text.split()) for c in chunks), 10)

    def test_overlap_contains_words_from_prior_chunk(self):
        doc = Document("p", " ".join(f"word{i}" for i in range(18)))
        chunks = recursive(doc, size=7, overlap=2).chunks
        self.assertGreater(len(chunks), 1)
        self.assertTrue(set(chunks[0].text.split()).intersection(chunks[1].text.split()))

    def test_source_offsets(self):
        doc = Document("p", "First paragraph contains facts.\n\nSecond paragraph also contains facts.")
        for item in recursive(doc, size=4).chunks:
            self.assertEqual(doc.text[item.start:item.end].strip(), item.text)


class SentenceTests(unittest.TestCase):
    def test_group_complete_sentences(self):
        doc = Document("s", "One fact. Second fact! Third fact?")
        result = sentence_or_paragraph(doc, group_size=2)
        self.assertEqual([c.text for c in result.chunks], [
            "One fact. Second fact!", "Third fact?",
        ])

    def test_paragraph_mode(self):
        doc = Document("s", "First. Second.\n\nThird. Fourth.")
        result = sentence_or_paragraph(doc, mode="paragraph", group_size=1)
        self.assertEqual([c.text.strip() for c in result.chunks], [
            "First. Second.", "Third. Fourth.",
        ])

    def test_invalid_mode_and_group(self):
        with self.assertRaises(ValueError):
            sentence_or_paragraph(Document("s", "Text"), mode="wrong")
        with self.assertRaises(ValueError):
            sentence_or_paragraph(Document("s", "Text"), group_size=0)

    def test_sentence_units_preserve_offsets(self):
        text = "  First sentence.\n Second sentence!"
        for value, start, end in sentence_units(text):
            self.assertEqual(text[start:end], value)


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.doc = load_document(LAB / "data" / "policies.md")

    def test_heading_path_and_original_source(self):
        result = markdown_sections(self.doc, max_words=40)
        self.assertTrue(result.parents)
        self.assertTrue(result.chunks)
        annual = [c for c in result.chunks if "Annual Leave" in c.section_path]
        self.assertTrue(annual)
        self.assertEqual(annual[0].section_path, ("Company Handbook", "Annual Leave"))
        self.assertIn("Full-time employees", annual[0].text)
        self.assertTrue(annual[0].parent_id)
        self.assertTrue(all(
            item.parent_id in {p.id for p in result.parents}
            for item in result.chunks
        ))

    def test_oversized_sections_splitting(self):
        result = markdown_sections(self.doc, max_words=15)
        self.assertGreater(len(result.chunks), len(result.parents))

    def test_no_heading_falls_back_to_recursive(self):
        doc = Document("p", "A short plain document with no Markdown headings.")
        result = markdown_sections(doc, max_words=10)
        self.assertTrue(result.chunks)
        self.assertFalse(result.parents)

    def test_empty_heading_sections_not_indexed(self):
        doc = Document("p", "# Top\n## Empty\n## Content\nA fact is here.")
        result = markdown_sections(doc, max_words=20)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].section_path, ("Top", "Content"))


class SemanticTests(unittest.TestCase):
    def test_topic_transition_splits(self):
        doc = Document(
            "t", "Annual leave policy is important. Leave days need approval. "
                 "Account passwords require two-factor authentication. Security resets expire soon."
        )

        class FakeEmbedder:
            def embed(self, texts):
                self.texts = texts
                return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

        stub = FakeEmbedder()
        result = semantic(doc, stub, distance_threshold=0.5, min_words=5)
        self.assertEqual(len(result.chunks), 2)
        self.assertIn("leave", result.chunks[0].text)
        self.assertIn("passwords", result.chunks[1].text)
        self.assertEqual(len(stub.texts), 4)

    def test_provider_count_mismatch(self):
        class BadEmbedder:
            def embed(self, texts):
                return []
        with self.assertRaises(ValueError):
            semantic(Document("x", "One sentence."), BadEmbedder())

    def test_embedding_dimension_mismatch(self):
        class BadEmbedder:
            def embed(self, texts):
                return [[1, 0], [0, 1, 2]]
        with self.assertRaises(ValueError):
            semantic(Document("x", "One sentence. Two sentences."), BadEmbedder())

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            semantic(Document("x", "One."), TopicLexicalEmbedder(), distance_threshold=-1)

    def test_real_sentence_count_with_offline_provider(self):
        doc = load_document(LAB / "data" / "transcript.txt")
        result = semantic(doc, TopicLexicalEmbedder(), max_words=20)
        self.assertTrue(result.chunks)
        self.assertTrue(all(c.document_id == doc.id for c in result.chunks))


class PropositionTests(unittest.TestCase):
    def test_source_linked_sentence_candidates(self):
        doc = Document("x", "Revenue may grow if demand rises. Activated plans cannot be refunded.")
        result = propositions(doc)
        self.assertEqual(len(result.chunks), 2)
        self.assertIn("may", result.chunks[0].text)
        self.assertIn("cannot", result.chunks[1].text)
        self.assertEqual(result.chunks[0].metadata["extraction"], "sentence_candidate")
        self.assertFalse(result.chunks[0].metadata["requires_source_verification"])

    def test_llm_like_extractor_marked_unverified(self):
        class StubExtractor:
            def extract_propositions(self, text):
                return ["Fact one", "Fact two"]
        result = propositions(Document("x", "One compound sentence."), extractor=StubExtractor())
        self.assertEqual(len(result.chunks), 2)
        self.assertTrue(all(c.start is None for c in result.chunks))
        self.assertTrue(all(c.metadata["requires_source_verification"] for c in result.chunks))
        self.assertEqual(result.chunks[0].metadata["source_sentence"], "One compound sentence.")

    def test_empty_extraction_fails(self):
        class Empty:
            def extract_propositions(self, text):
                return []
        with self.assertRaises(ValueError):
            propositions(Document("x", "One fact."), extractor=Empty())


class HierarchicalTests(unittest.TestCase):
    def setUp(self):
        self.doc = load_document(LAB / "data" / "policies.md")

    def test_children_have_real_parent(self):
        result = parent_child(self.doc, child_words=18, overlap=3)
        ids = {parent.id for parent in result.parents}
        self.assertTrue(result.chunks)
        self.assertTrue(all(c.parent_id in ids for c in result.chunks))
        self.assertGreater(len(result.chunks), len(result.parents))

    def test_parent_expansion_deduplicates(self):
        result = parent_child(self.doc, child_words=10, overlap=2)
        first = result.chunks[0]
        sibling = next(c for c in result.chunks if c.parent_id == first.parent_id and c.id != first.id)
        resolved = expand_to_parents([first, sibling], result)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].id, first.parent_id)

    def test_budget_can_exclude_oversized_parent(self):
        result = parent_child(self.doc, child_words=10)
        self.assertEqual(expand_to_parents(list(result.chunks[:1]), result, max_chars=1), [])

    def test_plain_text_can_have_whole_document_parent(self):
        doc = Document("x", "An ordinary plain document with several separate words.")
        result = parent_child(doc, child_words=3, overlap=0)
        self.assertEqual(len(result.parents), 1)


class ContextualTests(unittest.TestCase):
    def test_retains_original_and_adds_heading(self):
        doc = load_document(LAB / "data" / "policies.md")
        result = contextual(doc, max_words=40)
        annual = next(c for c in result.chunks if "Annual Leave" in c.section_path)
        self.assertIn("Document: policies", annual.text)
        self.assertIn("Section: Company Handbook / Annual Leave", annual.text)
        self.assertIn("Full-time employees", annual.metadata["original_text"])
        self.assertEqual(annual.metadata["context_source"], "deterministic_metadata")

    def test_context_contains_no_generated_claims(self):
        doc = Document("d", "# Heading\nAnnual leave is twelve days.")
        result = contextual(doc)
        self.assertTrue(result.chunks[0].text.endswith("Annual leave is twelve days."))


class TableTests(unittest.TestCase):
    def setUp(self):
        self.doc = load_document(LAB / "data" / "financial_report.md")

    def test_headers_repeated_with_each_row_group(self):
        result = table_aware(self.doc, rows_per_chunk=2)
        self.assertEqual(len(result.parents), 1)
        self.assertEqual(len(result.chunks), 2)
        self.assertTrue(all("Q2 2025" in c.text for c in result.chunks))
        self.assertTrue(all("billion VND" in c.text for c in result.chunks))
        self.assertEqual(result.chunks[0].metadata["row_numbers"], [1, 2])
        self.assertEqual(result.chunks[1].metadata["row_numbers"], [3, 4])

    def test_table_row_units_and_source_mapping(self):
        result = table_aware(self.doc, rows_per_chunk=1)
        revenue = result.chunks[0]
        self.assertIn("Software", revenue.text)
        self.assertIn("Q2 2025=150", revenue.text)
        self.assertEqual(revenue.parent_id, result.parents[0].id)
        self.assertIn("| Software |", self.doc.text[revenue.start:revenue.end])

    def test_bad_table_shape_fails(self):
        doc = Document("t", "| A | B |\n|---|---|\n| only_one |")
        with self.assertRaises(ValueError):
            table_aware(doc)

    def test_no_markdown_table_fails(self):
        with self.assertRaises(ValueError):
            table_aware(Document("x", "No table"))

    def test_bad_rows_per_chunk_fails(self):
        with self.assertRaises(ValueError):
            table_aware(self.doc, rows_per_chunk=0)


class CodeTests(unittest.TestCase):
    def setUp(self):
        self.doc = load_document(LAB / "data" / "order_service.py")

    def test_methods_and_parent_symbol(self):
        result = python_ast(self.doc)
        names = [c.metadata["symbol"] for c in result.chunks]
        self.assertIn("OrderService.calculate_revenue", names)
        self.assertIn("format_currency", names)
        method = next(c for c in result.chunks if c.metadata["symbol"] == "OrderService.calculate_revenue")
        self.assertEqual(method.section_path, ("OrderService", "calculate_revenue"))
        self.assertEqual(method.parent_id, result.parents[0].id)

    def test_source_line_range(self):
        result = python_ast(self.doc)
        for node in result.chunks:
            self.assertEqual(self.doc.text[node.start:node.end].strip(), node.text)

    def test_reject_invalid_python(self):
        with self.assertRaises(ValueError):
            python_ast(Document("x", "def broken(:\n pass"))

    def test_oversized_node_is_split_and_marked(self):
        doc = Document(
            "x", "def long_function():\n"
            + "".join(f"    value_{i} = {i}\n" for i in range(15))
        )
        result = python_ast(doc, max_lines=5)
        self.assertGreater(len(result.chunks), 1)
        self.assertTrue(all(node.metadata["partial"] for node in result.chunks))

    def test_non_code_input_fails(self):
        with self.assertRaises(ValueError):
            python_ast(Document("x", "x = 2"))


class LateTests(unittest.TestCase):
    def test_context_encoder_receives_complete_document_first(self):
        class FakeEncoder:
            def encode_words(self, words):
                self.words = words
                return [[float(i), 1.0] for i in range(len(words))]
        doc = Document("x", "alpha beta gamma delta epsilon")
        encoder = FakeEncoder()
        result = late(doc, encoder, size=3, overlap=1)
        self.assertEqual(encoder.words, ["alpha", "beta", "gamma", "delta", "epsilon"])
        self.assertEqual(result.vectors[0], (1.0, 1.0))
        self.assertEqual(result.vectors[1], (3.0, 1.0))
        self.assertEqual(len(result.chunks), 2)
        self.assertIn("mean_after_contextual_encoding", result.chunks[0].metadata["pooling"])

    def test_toy_word_encoder_produces_per_word_vectors(self):
        vectors = WordContextToyEncoder().encode_words(["hello", "world"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 64)

    def test_encoder_count_mismatch(self):
        class EmptyEncoder:
            def encode_words(self, words):
                return []
        with self.assertRaises(ValueError):
            late(Document("x", "alpha beta"), EmptyEncoder())

    def test_encoder_invalid_vector_dimension(self):
        class BadEncoder:
            def encode_words(self, words):
                return [[1.0], [1.0, 2.0]]
        with self.assertRaises(ValueError):
            late(Document("x", "alpha beta"), BadEncoder())

    def test_late_chunks_keep_source_spans(self):
        doc = Document("x", "One two three four five.")
        result = late(doc, WordContextToyEncoder(), size=3, overlap=1)
        for node in result.chunks:
            self.assertEqual(doc.text[node.start:node.end], node.text)


class RaptorTests(unittest.TestCase):
    def setUp(self):
        self.doc = load_document(LAB / "data" / "policies.md")

    def test_leaf_and_summary_levels(self):
        result = raptor(self.doc, leaf_words=18)
        self.assertTrue(any(c.kind == "text" for c in result.chunks))
        summaries = [c for c in result.chunks if c.kind == "summary"]
        self.assertTrue(summaries)
        ids = {c.id for c in result.chunks}
        self.assertTrue(all(set(s.metadata["child_ids"]).issubset(ids) for s in summaries))
        self.assertTrue(all(set(s.metadata["leaf_ids"]).issubset(ids) for s in summaries))

    def test_extractive_summary_is_not_marked_llm(self):
        result = raptor(self.doc, leaf_words=15)
        summaries = [c for c in result.chunks if c.kind == "summary"]
        self.assertTrue(all(not c.metadata["generated"] for c in summaries))

    def test_llm_summary_is_flagged_unverified(self):
        class FakeSummary:
            def summarize(self, passages):
                return "Synthetic, not verified."
        result = raptor(self.doc, leaf_words=20, summarizer=FakeSummary())
        summaries = [c for c in result.chunks if c.kind == "summary"]
        self.assertTrue(summaries)
        self.assertTrue(all(c.metadata["requires_source_verification"] for c in summaries))

    def test_reject_invalid_summary_and_levels(self):
        class Empty:
            def summarize(self, passages):
                return ""
        with self.assertRaises(ValueError):
            raptor(self.doc, leaf_words=10, summarizer=Empty())
        with self.assertRaises(ValueError):
            raptor(self.doc, max_levels=0)


class IntegrationTests(unittest.TestCase):
    def test_every_strategy_has_sample_and_runs_offline(self):
        for name in STRATEGIES:
            with self.subTest(strategy=name):
                doc = load_document(sample_file(name))
                result = run(name, doc)
                self.assertTrue(result.chunks)
                self.assertEqual(len({c.id for c in result.chunks}), len(result.chunks))

    def test_api_provider_is_restricted_to_supported_strategies(self):
        with self.assertRaises(ValueError):
            run("fixed", Document("x", "Test one two three."), provider="api")

    def test_local_model_only_late(self):
        with self.assertRaises(ValueError):
            run("semantic", Document("x", "One."), local_model_dir=Path("/tmp/model"))

    def test_load_document_rejects_wrong_extension_and_size(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "source.csv"
            p.write_text("col1,col2")
            with self.assertRaises(ValueError):
                load_document(p)
            p = Path(folder) / "large.txt"
            p.write_text("word " * 50000)
            with self.assertRaises(ValueError):
                load_document(p)


if __name__ == "__main__":
    unittest.main()