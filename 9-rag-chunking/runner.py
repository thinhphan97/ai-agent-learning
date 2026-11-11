"""Single dispatcher for independently testable standalone chunking modules."""

from __future__ import annotations

from pathlib import Path

from basic import fixed, recursive, sentence_or_paragraph
from code import python_ast
from contextual import contextual
from contracts import ChunkSet, Document
from hierarchy import parent_child
from late import late
from layout import table_aware
from proposition import propositions
from providers import (
    LocalTransformerWordEncoder,
    OpenAIEmbeddings,
    OpenAITextProcessor,
    TopicLexicalEmbedder,
    WordContextToyEncoder,
)
from raptor import raptor
from semantic import semantic
from structure import markdown_sections

STRATEGIES = (
    "fixed",
    "recursive",
    "sentence",
    "paragraph",
    "structure",
    "semantic",
    "proposition",
    "parent-child",
    "contextual",
    "table",
    "code",
    "late",
    "raptor",
)
SAMPLES = {
    "table": "financial_report.md",
    "code": "order_service.py",
    "semantic": "transcript.txt",
}
SAMPLE_DIR = Path(__file__).resolve().parent / "data"


def sample_file(strategy: str) -> Path:
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")
    return SAMPLE_DIR / SAMPLES.get(strategy, "policies.md")


def run(
    strategy: str,
    doc: Document,
    *,
    size: int = 35,
    overlap: int = 5,
    group_size: int = 2,
    threshold: float = 0.55,
    provider: str = "toy",
    local_model_dir: Path | None = None,
) -> ChunkSet:
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")
    if provider not in {"toy", "api"}:
        raise ValueError("provider must be toy or api")
    if provider == "api" and strategy not in {"semantic", "proposition", "raptor"}:
        raise ValueError("API provider only applies to semantic, proposition or raptor")
    if local_model_dir is not None and strategy != "late":
        raise ValueError("Local Transformer model is only for late chunking")
    if strategy == "fixed":
        return fixed(doc, size=size, overlap=overlap)
    if strategy == "recursive":
        return recursive(doc, size=size, overlap=overlap)
    if strategy in {"sentence", "paragraph"}:
        return sentence_or_paragraph(doc, mode=strategy, group_size=group_size)
    if strategy == "structure":
        return markdown_sections(doc, max_words=size)
    if strategy == "semantic":
        emb = OpenAIEmbeddings() if provider == "api" else TopicLexicalEmbedder()
        return semantic(doc, emb, distance_threshold=threshold, max_words=max(size, 12))
    if strategy == "proposition":
        proc = OpenAITextProcessor() if provider == "api" else None
        return propositions(doc, extractor=proc)
    if strategy == "parent-child":
        return parent_child(doc, child_words=size, overlap=overlap)
    if strategy == "contextual":
        return contextual(doc, max_words=size)
    if strategy == "table":
        return table_aware(doc, rows_per_chunk=group_size)
    if strategy == "code":
        return python_ast(doc, max_lines=max(3, size))
    if strategy == "late":
        enc = (
            LocalTransformerWordEncoder(local_model_dir)
            if local_model_dir is not None
            else WordContextToyEncoder()
        )
        return late(doc, enc, size=size, overlap=overlap)
    if strategy == "raptor":
        proc = OpenAITextProcessor() if provider == "api" else None
        return raptor(doc, leaf_words=size, max_levels=4, summarizer=proc)
    raise AssertionError("All strategies must be dispatched")
