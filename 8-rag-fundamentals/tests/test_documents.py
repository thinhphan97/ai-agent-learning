"""Corpus loading and stable overlapping chunk identities."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documents import Document, chunk_documents, load_documents


ROOT = Path(__file__).resolve().parents[1]


class DocumentTests(unittest.TestCase):
    def test_fixture_corpus_loading_is_deterministic(self):
        first = load_documents(ROOT / "data" / "documents")
        second = load_documents(ROOT / "data" / "documents")
        self.assertEqual(first, second)
        self.assertEqual([d.id for d in first], [
            "billing", "returns", "security", "shipping", "support",
        ])
        self.assertEqual(first[1].title, "Returns Policy")

    def test_overlap_between_complete_chunks(self):
        document = Document("demo", "Demo", " ".join(f"w{i}" for i in range(22)))
        chunks = chunk_documents([document], chunk_size=10, overlap=3)
        self.assertEqual([c.id for c in chunks], ["demo:0", "demo:1", "demo:2"])
        self.assertEqual(chunks[0].word_start, 0)
        self.assertEqual(chunks[1].word_start, 7)
        self.assertEqual(chunks[2].word_start, 14)
        self.assertEqual(chunks[0].text.split()[-3:], chunks[1].text.split()[:3])
        self.assertEqual(chunks[1].text.split()[-3:], chunks[2].text.split()[:3])
        self.assertEqual(chunks[-1].word_end, 22)

    def test_short_document_single_chunk(self):
        document = Document("short", "Short", "One two three")
        chunks = chunk_documents([document], chunk_size=5, overlap=4)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "One two three")

    def test_invalid_chunk_parameters(self):
        doc = [Document("x", "Title", "word " * 30)]
        for chunk_size, overlap in ((4, 0), (10, 10), (10, -1), (0, 0), (True, 0), (10, True)):
            with self.subTest(chunk_size=chunk_size, overlap=overlap), self.assertRaises(ValueError):
                chunk_documents(doc, chunk_size=chunk_size, overlap=overlap)

    def test_empty_and_duplicate_documents(self):
        with self.assertRaises(ValueError):
            chunk_documents([])
        with self.assertRaises(ValueError):
            chunk_documents([Document("d", "Title", "text"), Document("d", "Title", "more")])
        with self.assertRaises(ValueError):
            chunk_documents([Document("empty", "Empty", "")])

    def test_empty_document_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with self.assertRaises(ValueError):
                load_documents(folder)
            (folder / "empty.txt").write_text("  ")
            with self.assertRaises(ValueError):
                load_documents(folder)

    def test_bad_document_name_and_oversize(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "bad name.txt").write_text("Title\nText")
            with self.assertRaises(ValueError):
                load_documents(folder)
            (folder / "bad name.txt").unlink()
            (folder / "huge.txt").write_text("A" * 100)
            with self.assertRaises(ValueError):
                load_documents(folder, max_bytes=10)

    def test_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            target = folder / "real.txt"
            target.write_text("Real document content")
            link = folder / "copy.txt"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation unsupported")
            with self.assertRaises(ValueError):
                load_documents(folder)

    def test_file_count_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for number in range(3):
                (folder / f"document_{number}.txt").write_text("Title content")
            with self.assertRaises(ValueError):
                load_documents(folder, max_files=2)

    def test_nonexistent_directory(self):
        with self.assertRaises(ValueError):
            load_documents(Path("/nonexistent-rag-lesson-documents-dir"))


if __name__ == "__main__":
    unittest.main()
