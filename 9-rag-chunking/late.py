"""Late chunking mechanics: contextual encode full span -> pool word ranges.

Real mode requires an explicitly supplied local Transformer word encoder.
Offline mode uses a clearly marked pedagogical context-mixing toy.
"""

from __future__ import annotations

import math
from typing import Protocol

from contracts import ChunkSet, Document, make_chunk
from shared import check_size, word_spans


class WordEncoder(Protocol):
    def encode_words(self, words: list[str]) -> list[list[float]]: ...


def late(
    doc: Document,
    encoder: WordEncoder,
    *,
    size: int = 40,
    overlap: int = 5,
) -> ChunkSet:
    check_size(size, overlap)
    positions = word_spans(doc.text)
    if not positions:
        raise ValueError("No words to encode")
    words = [doc.text[a:b] for a, b in positions]
    # The entire document is encoded before chunk-level vector pooling.
    token_vectors = encoder.encode_words(words)
    if len(token_vectors) != len(words) or not token_vectors:
        raise ValueError("Encoder word count mismatch")
    dim = len(token_vectors[0])
    if dim < 1 or any(
        len(v) != dim
        or any(type(x) not in (float, int) or not math.isfinite(x) for x in v)
        for v in token_vectors
    ):
        raise ValueError("Invalid contextual vector dimensions/values")
    chunks, pooled_vectors = [], []
    stride = size - overlap
    for start in range(0, len(words), stride):
        finish = min(start + size, len(words))
        lo, hi = positions[start][0], positions[finish - 1][1]
        chunks.append(
            make_chunk(
                doc,
                "late",
                len(chunks),
                doc.text[lo:hi],
                start=lo,
                end=hi,
                metadata={
                    "word_range": [start, finish],
                    "pooling": "mean_after_contextual_encoding",
                },
            )
        )
        width = finish - start
        pooled = tuple(
            sum(token_vectors[i][column] for i in range(start, finish)) / width
            for column in range(dim)
        )
        pooled_vectors.append(pooled)
        if finish == len(words):
            break
    return ChunkSet("late", tuple(chunks), vectors=tuple(pooled_vectors))
