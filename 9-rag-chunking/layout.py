"""Markdown table-aware indexing; does not pretend to parse PDF layout/OCR."""

from __future__ import annotations

import re

from contracts import ChunkSet, Document, make_chunk


def _cells(line: str) -> list[str]:
    return [value.strip() for value in line.strip().strip("|").split("|")]


def table_aware(doc: Document, *, rows_per_chunk: int = 2) -> ChunkSet:
    """Keep table title, header, units and row mapping in every table chunk."""
    if type(rows_per_chunk) is not int or not 1 <= rows_per_chunk <= 50:
        raise ValueError("rows_per_chunk must be in [1, 50]")
    lines = doc.text.splitlines(keepends=True)
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    chunks, parents = [], []
    section = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and re.fullmatch(r"\s*\|?[\s|:\-]+\|?\s*", lines[i + 1])
        ):
            header = _cells(lines[i])
            if not header or len(header) < 2:
                i += 1
                continue
            end = i + 2
            while end < len(lines) and lines[end].strip().startswith("|"):
                end += 1
            rows = [(_cells(lines[k]), k) for k in range(i + 2, end)]
            if any(len(row) != len(header) for row, _ in rows):
                raise ValueError("Table row/header column mismatch")
            if not rows:
                i = end
                continue
            table_id = f"{doc.id}:table:{len(parents)}"
            parent = make_chunk(
                doc,
                "table-parent",
                len(parents),
                doc.text[offsets[i] : offsets[end - 1] + len(lines[end - 1])],
                kind="table",
                start=offsets[i],
                end=offsets[end - 1] + len(lines[end - 1]),
                metadata={"title": section, "columns": header, "table_id": table_id},
            )
            parents.append(parent)
            for cursor in range(0, len(rows), rows_per_chunk):
                group = rows[cursor : cursor + rows_per_chunk]
                serialized = "\n".join(
                    f"{' | '.join(f'{name}={cell}' for name, cell in zip(header, values))}"
                    for values, _ in group
                )
                text = f"Table: {section or doc.id}\nColumns: {' | '.join(header)}\n{serialized}"
                chunks.append(
                    make_chunk(
                        doc,
                        "table",
                        len(chunks),
                        text,
                        kind="table_rows",
                        start=offsets[group[0][1]],
                        end=offsets[group[-1][1]] + len(lines[group[-1][1]]),
                        parent_id=parent.id,
                        section_path=(section,) if section else (),
                        metadata={
                            "columns": header,
                            "rows": [values for values, _ in group],
                            "table_id": table_id,
                            "row_numbers": [idx - i - 1 for _, idx in group],
                        },
                    )
                )
            i = end
        else:
            i += 1
    if not chunks:
        raise ValueError("No Markdown tables found; PDF/OCR is not implemented")
    return ChunkSet("table", tuple(chunks), tuple(parents))
