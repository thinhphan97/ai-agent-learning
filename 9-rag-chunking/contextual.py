"""Deterministic context-enrichment (document/heading metadata), not LLM invention."""

from __future__ import annotations

from contracts import ChunkSet, Document, make_chunk
from structure import markdown_sections


def contextual(doc: Document, *, max_words: int = 100) -> ChunkSet:
    base = markdown_sections(doc, max_words=max_words)
    chunks = []
    for chunk in base.chunks:
        prefix = (
            f"Document: {doc.id}\n"
            f"Section: {' / '.join(chunk.section_path) or '(none)'}\n"
        )
        chunks.append(
            make_chunk(
                doc,
                "contextual",
                len(chunks),
                prefix + chunk.text,
                kind="enriched",
                start=chunk.start,
                end=chunk.end,
                parent_id=chunk.parent_id,
                section_path=chunk.section_path,
                metadata={
                    "original_text": chunk.text,
                    "context_source": "deterministic_metadata",
                },
            )
        )
    return ChunkSet("contextual", tuple(chunks), base.parents)
