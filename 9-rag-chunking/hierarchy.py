"""Small child index + larger source parents, with deterministic parent expansion."""

from __future__ import annotations

from contracts import Chunk, ChunkSet, Document, make_chunk
from shared import check_size, word_spans
from structure import markdown_sections


def parent_child(
    doc: Document,
    *,
    child_words: int = 30,
    overlap: int = 5,
) -> ChunkSet:
    check_size(child_words, overlap)
    sections = markdown_sections(doc, max_words=4096)
    parents = (
        list(sections.parents)
        if sections.parents
        else [
            make_chunk(
                doc, "parent", 0, doc.text, kind="section", start=0, end=len(doc.text)
            )
        ]
    )
    chunks = []
    for parent in parents:
        spans = word_spans(parent.text)
        if not spans:
            continue
        step = child_words - overlap
        for begin in range(0, len(spans), step):
            finish = min(begin + child_words, len(spans))
            a, b = spans[begin][0], spans[finish - 1][1]
            chunks.append(
                make_chunk(
                    doc,
                    "child",
                    len(chunks),
                    parent.text[a:b],
                    start=parent.start + a if parent.start is not None else None,
                    end=parent.start + b if parent.start is not None else None,
                    parent_id=parent.id,
                    section_path=parent.section_path,
                    metadata={"word_range": [begin, finish]},
                )
            )
            if finish == len(spans):
                break
    return ChunkSet("parent-child", tuple(chunks), tuple(parents))


def expand_to_parents(
    hits: list[Chunk],
    collection: ChunkSet,
    *,
    max_chars: int = 5000,
) -> list[Chunk]:
    """Retrieve children, provide unique parent contexts within a character cap."""
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be positive")
    parent_map = {node.id: node for node in collection.parents}
    resolved, seen, total = [], set(), 0
    for child in hits:
        parent = parent_map.get(child.parent_id)
        if parent is None:
            continue
        if parent.id in seen:
            continue
        seen.add(parent.id)
        if total + len(parent.text) > max_chars:
            continue
        resolved.append(parent)
        total += len(parent.text)
    return resolved
