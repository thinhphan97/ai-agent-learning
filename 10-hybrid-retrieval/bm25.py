"""BM25 Okapi with a real in-memory inverted index and corpus statistics.

Uses a nonnegative Robertson-style IDF variant and no external dependencies.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Mapping, Sequence

from analyzer import tokenize, validate_query
from contracts import Chunk, Hit
from metadata import matches, normalize_filters


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.2, b: float = 0.75):
        if not chunks or not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Need chunks and valid BM25 k1/b")
        self.chunks = list(chunks)
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk ID")
        self.k1, self.b = k1, b
        tokens = [tokenize(c.contextual_text) for c in self.chunks]
        self.avgdl = sum(map(len, tokens)) / len(tokens)
        if self.avgdl == 0:
            raise ValueError("Empty index after tokenization")
        self.lengths = [len(ts) for ts in tokens]
        # A posting stores document index and term frequency.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_index, words in enumerate(tokens):
            for term, count in Counter(words).items():
                self.postings[term].append((doc_index, count))
        n = len(tokens)
        self.idf = {
            term: math.log1p((n - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def search(
        self,
        question: str,
        *,
        tenant_id: str,
        filters: Mapping[str, str] | None = None,
        top_k: int = 5,
    ) -> list[Hit]:
        validate_query(question, top_k)
        canonical = normalize_filters(filters)
        if not tenant_id:
            raise ValueError("Trusted tenant_id is required")
        scores: dict[int, float] = defaultdict(float)
        # Deduplicate query terms (the common simple BM25 variant used here).
        for term in dict.fromkeys(tokenize(question)):
            for doc_index, tf in self.postings.get(term, ()):
                chunk = self.chunks[doc_index]
                if not matches(chunk, tenant_id, canonical):
                    continue
                norm = 1 - self.b + self.b * self.lengths[doc_index] / self.avgdl
                scores[doc_index] += (
                    self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
                )
        ranked = sorted(
            (
                Hit(self.chunks[i], score, "bm25")
                for i, score in scores.items()
                if score > 0
            ),
            key=lambda hit: (-hit.score, hit.chunk.id),
        )
        return ranked[:top_k]
