"""Exercise optional adapter wiring without contacting provider services."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Document
from runner import run


class ToyEmbeddings:
    def embed(self, texts):
        return [[1.0, float(i % 2)] for i, _ in enumerate(texts)]


class StubProcessor:
    def extract_propositions(self, passage):
        return [passage + " [stub]"]

    def summarize(self, passages):
        return " / ".join(passages)[:200]


class FakeContextEncoder:
    def encode_words(self, words):
        return [[float(i + 1), 0.0] for i in range(len(words))]


class ModelWiringTests(unittest.TestCase):
    def test_semantic_real_provider_interface_injected(self):
        with patch("runner.OpenAIEmbeddings", return_value=ToyEmbeddings()) as provider:
            result = run(
                "semantic",
                Document("x", "First policy. Second policy. Third policy."),
                provider="api",
            )
        provider.assert_called_once_with()
        self.assertTrue(result.chunks)

    def test_proposition_model_interface_injected(self):
        with patch("runner.OpenAITextProcessor", return_value=StubProcessor()) as provider:
            result = run("proposition", Document("x", "One fact."), provider="api")
        provider.assert_called_once_with()
        self.assertEqual(result.chunks[0].text, "One fact. [stub]")
        self.assertTrue(result.chunks[0].metadata["requires_source_verification"])

    def test_raptor_summary_model_interface_injected(self):
        with patch("runner.OpenAITextProcessor", return_value=StubProcessor()) as provider:
            result = run(
                "raptor",
                Document("x", " ".join(f"word{i}" for i in range(100))),
                size=20, provider="api",
            )
        provider.assert_called_once_with()
        summary = next(c for c in result.chunks if c.kind == "summary")
        self.assertTrue(summary.metadata["generated"])

    def test_local_transformer_interface_injected(self):
        location = Path("/pretend/local-model")
        with patch("runner.LocalTransformerWordEncoder", return_value=FakeContextEncoder()) as provider:
            result = run(
                "late", Document("x", "Alpha beta gamma delta epsilon"),
                size=3, overlap=1, local_model_dir=location,
            )
        provider.assert_called_once_with(location)
        self.assertEqual(result.vectors[0], (2.0, 0.0))
        self.assertEqual(result.vectors[1], (4.0, 0.0))


class DocumentationTests(unittest.TestCase):
    def test_all_twelve_guides_exist(self):
        lab = Path(__file__).resolve().parents[1]
        guides = sorted((lab / "guides").glob("*/README.md"))
        self.assertEqual(len(guides), 12)
        for guide in guides:
            with self.subTest(path=guide):
                content = guide.read_text(encoding="utf-8")
                self.assertIn("## Algorithm", content) if "## Algorithm" in content else self.assertIn("## Exact mechanism", content)
                self.assertIn("##", content)
                self.assertTrue(len(content) > 1000)

    def test_overview_links_to_every_guide(self):
        lab = Path(__file__).resolve().parents[1]
        overview = (lab / "README.md").read_text(encoding="utf-8")
        for guide in sorted((lab / "guides").glob("*/README.md")):
            self.assertIn(f"guides/{guide.parent.name}/README.md", overview)


if __name__ == "__main__":
    unittest.main()
