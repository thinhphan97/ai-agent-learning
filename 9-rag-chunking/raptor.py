"""Small source-traceable RAPTOR-style cluster/summarize teaching implementation.

Offline summaries are EXTRACTIVE, not abstractive; optional LLM summarizer is
available but generated claims must be verified against original leaf chunks.
"""

from __future__ import annotations

from typing import Protocol

from basic import fixed
from contracts import Chunk, ChunkSet, Document, make_chunk
from shared import lexical_vector, similarity


class Summarizer(Protocol):
    def summarize(self, passages: list[str]) -> str: ...


class ExtractiveSummarizer:
    """Take a brief leading sentence from each passage; intentionally lossy."""

    def summarize(self, passages: list[str]) -> str:
        snippets = []
        for passage in passages:
            first = passage.split(". ")[0].strip()
            if first:
                snippets.append(first[:250])
        return " | ".join(snippets)[:1500]


def _clusters(nodes: list[Chunk]) -> list[list[Chunk]]:
    """Deterministic greedy lexical nearest-neighbor pairing (not KMeans)."""
    remaining = list(nodes)
    clusters = []
    while remaining:
        first = remaining.pop(0)
        if not remaining:
            clusters.append([first])
            continue
        first_vec = lexical_vector(first.text)
        index = max(
            range(len(remaining)),
            key=lambda i: (
                similarity(first_vec, lexical_vector(remaining[i].text)),
                -i,
            ),
        )
        second = remaining.pop(index)
        clusters.append([first, second])
    return clusters


def raptor(
    doc: Document,
    *,
    leaf_words: int = 25,
    max_levels: int = 4,
    summarizer: Summarizer | None = None,
) -> ChunkSet:
    if type(max_levels) is not int or not 1 <= max_levels <= 10:
        raise ValueError("max_levels must be in [1, 10]")
    leaves = fixed(doc, size=leaf_words, overlap=0).chunks
    if not leaves:
        raise ValueError("No leaves")
    summaries = []
    current = list(leaves)
    process = summarizer or ExtractiveSummarizer()
    for level in range(1, max_levels + 1):
        if len(current) <= 1:
            break
        next_level = []
        for group in _clusters(current):
            if len(group) == 1:
                next_level.append(group[0])
                continue
            summary = process.summarize([item.text for item in group])
            if (
                not isinstance(summary, str)
                or not summary.strip()
                or len(summary) > 2000
            ):
                raise ValueError("Invalid summary")
            source_ids = list(
                dict.fromkeys(
                    source
                    for item in group
                    for source in item.metadata.get("leaf_ids", [item.id])
                )
            )
            node = make_chunk(
                doc,
                f"raptor-l{level}",
                len(summaries),
                summary,
                kind="summary",
                metadata={
                    "level": level,
                    "child_ids": [item.id for item in group],
                    "leaf_ids": source_ids,
                    "generated": summarizer is not None,
                    "requires_source_verification": summarizer is not None,
                },
            )
            summaries.append(node)
            next_level.append(node)
        current = next_level
    # All levels are indexable, as in multi-level retrieval.
    return ChunkSet("raptor", tuple(leaves) + tuple(summaries))
