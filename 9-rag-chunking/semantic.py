"""Embedding-distance sentence splitting (real via injected provider; toy otherwise)."""

from __future__ import annotations

from typing import Protocol

from contracts import ChunkSet, Document, make_chunk
from shared import sentence_units, vector_cosine


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def semantic(
    doc: Document,
    embeddings: Embedder,
    *,
    distance_threshold: float = 0.55,
    min_words: int = 6,
    max_words: int = 100,
) -> ChunkSet:
    if not 0 <= distance_threshold <= 2:
        raise ValueError("distance_threshold must be in [0, 2]")
    if (
        type(min_words) is not int
        or type(max_words) is not int
        or not 1 <= min_words <= max_words
    ):
        raise ValueError("Invalid semantic word limits")
    sentences = sentence_units(doc.text)
    if not sentences:
        raise ValueError("No sentences to chunk")
    vectors = embeddings.embed([s for s, _, _ in sentences])
    if len(vectors) != len(sentences):
        raise ValueError("Embedding count mismatch")
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("Embedding dimensions differ")

    chunks = []
    current = [sentences[0]]
    previous_boundary_distance = 0.0

    def flush():
        nonlocal current
        if current:
            a, b = current[0][1], current[-1][2]
            chunks.append(
                make_chunk(
                    doc,
                    "semantic",
                    len(chunks),
                    doc.text[a:b],
                    start=a,
                    end=b,
                    metadata={
                        "sentences": len(current),
                        "boundary_distance": round(previous_boundary_distance, 4),
                    },
                )
            )
            current = []

    for i in range(1, len(sentences)):
        distance = 1 - vector_cosine(vectors[i - 1], vectors[i])
        words_so_far = sum(len(text.split()) for text, _, _ in current)
        incoming = len(sentences[i][0].split())
        if (words_so_far >= min_words and distance >= distance_threshold) or (
            words_so_far + incoming > max_words
        ):
            previous_boundary_distance = distance
            flush()
        current.append(sentences[i])
    flush()
    return ChunkSet("semantic", tuple(chunks))
