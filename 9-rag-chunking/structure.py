"""Markdown section splitting with preserved heading paths and source offsets."""

from __future__ import annotations

import re

from basic import recursive
from contracts import ChunkSet, Document, make_chunk
from shared import check_size

HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def markdown_sections(doc: Document, *, max_words: int = 100) -> ChunkSet:
    """Use heading hierarchy; recursively split only oversized sections."""
    check_size(max_words)
    headings = list(HEADING.finditer(doc.text))
    if not headings:
        return recursive(doc, size=max_words)
    chunks = []
    parents = []
    paths: dict[int, str] = {}
    boundaries = (
        [(0, 0, "Preamble", 0)] if doc.text[: headings[0].start()].strip() else []
    )
    for match in headings:
        boundaries.append(
            (match.start(), len(match.group(1)), match.group(2).strip(), match.end())
        )

    for index, (start, depth, title, body_start) in enumerate(boundaries):
        if depth:
            paths = {level: name for level, name in paths.items() if level < depth}
            paths[depth] = title
        path = tuple(paths[k] for k in sorted(paths)) if depth else ()
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(doc.text)
        content = doc.text[body_start:end].strip()
        if not content:
            continue
        parent = make_chunk(
            doc,
            "section",
            index,
            doc.text[start:end],
            kind="section",
            start=start,
            end=end,
            section_path=path,
        )
        parents.append(parent)
        if len(content.split()) <= max_words:
            chunks.append(
                make_chunk(
                    doc,
                    "structure",
                    len(chunks),
                    content,
                    start=body_start,
                    end=end,
                    parent_id=parent.id,
                    section_path=path,
                )
            )
        else:
            # A new local document lets the recursive splitter preserve offsets.
            local = Document(doc.id, doc.text[body_start:end], doc.source)
            for small in recursive(local, size=max_words).chunks:
                chunks.append(
                    make_chunk(
                        doc,
                        "structure",
                        len(chunks),
                        small.text,
                        start=body_start + small.start,
                        end=body_start + small.end,
                        parent_id=parent.id,
                        section_path=path,
                    )
                )
    return ChunkSet("structure", tuple(chunks), tuple(parents))
