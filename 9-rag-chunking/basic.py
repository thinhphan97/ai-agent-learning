"""Fixed-size, recursive separator, and sentence/paragraph strategies."""

from __future__ import annotations

import re

from contracts import ChunkSet, Document, make_chunk
from shared import check_size, sentence_units, word_spans


def fixed(doc: Document, *, size: int = 40, overlap: int = 8) -> ChunkSet:
    """Exact word-window sliding; keeps original character offsets."""
    check_size(size, overlap)
    positions = word_spans(doc.text)
    chunks = []
    step = size - overlap
    for index, start in enumerate(range(0, len(positions), step)):
        end = min(start + size, len(positions))
        lo, hi = positions[start][0], positions[end - 1][1]
        chunks.append(
            make_chunk(
                doc,
                "fixed",
                index,
                doc.text[lo:hi],
                start=lo,
                end=hi,
                metadata={"word_range": [start, end]},
            )
        )
        if end == len(positions):
            break
    return ChunkSet("fixed", tuple(chunks))


def _split_ranges(
    text: str,
    lo: int,
    hi: int,
    level: int,
    size: int,
) -> list[tuple[int, int]]:
    """Recursive separators: paragraphs -> lines -> sentences -> words."""
    if len(text[lo:hi].split()) <= size:
        return [(lo, hi)] if text[lo:hi].strip() else []
    if level == 4:
        positions = [
            (lo + m.start(), lo + m.end()) for m in re.finditer(r"\S+", text[lo:hi])
        ]
        return [
            (positions[i][0], positions[min(i + size, len(positions)) - 1][1])
            for i in range(0, len(positions), size)
        ]

    patterns = (r"\n[ \t]*\n+", r"\n+", r"(?<=[.!?])\s+", r"\s+")
    matches = list(re.finditer(patterns[level], text[lo:hi]))
    if not matches:
        return _split_ranges(text, lo, hi, level + 1, size)

    results: list[tuple[int, int]] = []
    cursor = lo
    for m in matches:
        cut_a, cut_b = lo + m.start(), lo + m.end()
        results.extend(_split_ranges(text, cursor, cut_a, level + 1, size))
        cursor = cut_b
    results.extend(_split_ranges(text, cursor, hi, level + 1, size))
    return results


def recursive(doc: Document, *, size: int = 40, overlap: int = 0) -> ChunkSet:
    """Merge recursive separator units; optional overlap is in words.

    Overlap is appended as a look-back prefix. The final chunk can exceed
    `size` by up to `overlap` words; this is explicit in this teaching model.
    """
    check_size(size, overlap)
    units = _split_ranges(doc.text, 0, len(doc.text), 0, size)
    grouped: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []
    count = 0
    for a, b in units:
        words = len(doc.text[a:b].split())
        if current and count + words > size:
            grouped.append((current[0][0], current[-1][1]))
            current, count = [], 0
        current.append((a, b))
        count += words
    if current:
        grouped.append((current[0][0], current[-1][1]))
    positions = word_spans(doc.text)
    chunks = []
    for i, (a, b) in enumerate(grouped):
        if i and overlap:
            prior_words = [pos for pos in positions if pos[1] <= a]
            if prior_words:
                a = prior_words[max(0, len(prior_words) - overlap)][0]
        chunks.append(make_chunk(doc, "recursive", i, doc.text[a:b], start=a, end=b))
    return ChunkSet("recursive", tuple(chunks))


def sentence_or_paragraph(
    doc: Document,
    *,
    mode: str = "sentence",
    group_size: int = 2,
) -> ChunkSet:
    """Group complete sentences or complete paragraphs; no mid-unit cuts."""
    if (
        mode not in {"sentence", "paragraph"}
        or type(group_size) is not int
        or not 1 <= group_size <= 50
    ):
        raise ValueError("mode must be sentence/paragraph; group_size in [1,50]")
    if mode == "sentence":
        spans = [(start, end) for _, start, end in sentence_units(doc.text)]
    else:
        spans = []
        cursor = 0
        for m in re.finditer(r"\n[ \t]*\n+", doc.text):
            if doc.text[cursor : m.start()].strip():
                spans.append((cursor, m.start()))
            cursor = m.end()
        if doc.text[cursor:].strip():
            spans.append((cursor, len(doc.text)))
    chunks = [
        make_chunk(
            doc,
            mode,
            i // group_size,
            doc.text[spans[i][0] : spans[min(i + group_size, len(spans)) - 1][1]],
            start=spans[i][0],
            end=spans[min(i + group_size, len(spans)) - 1][1],
        )
        for i in range(0, len(spans), group_size)
    ]
    return ChunkSet(mode, tuple(chunks))
