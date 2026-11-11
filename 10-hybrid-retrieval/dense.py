"""Dense retrieval protocol and honest offline lexical-hashing surrogate.

The default hash vectors are deterministic but NOT neural semantic embeddings.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from analyzer import tokenize, validate_query
from contracts import Chunk, Hit
from metadata import matches, normalize_filters


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def unit(values: Sequence[float]) -> tuple[float, ...]:
    if not values or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in values
    ):
        raise ValueError("Invalid embedding vector")
    norm = math.sqrt(sum(float(v) ** 2 for v in values))
    if norm <= 0:
        raise ValueError("Zero-length embedding vector")
    return tuple(float(v) / norm for v in values)


class HashingEmbedder:
    """Offline deterministic feature hashing; measures lexical similarity, not semantics."""

    def __init__(self, dimensions: int = 256):
        if type(dimensions) is not int or dimensions < 32:
            raise ValueError("dimensions must be >=32")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            arr = [0.0] * self.dimensions
            for term, frequency in Counter(tokenize(text)).items():
                index = (
                    int.from_bytes(
                        hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(),
                        "big",
                    )
                    % self.dimensions
                )
                arr[index] += float(frequency)
            vectors.append(arr)
        return vectors


class OpenAIEmbedder:
    """Opt-in real API embeddings; root .env only, no network in default mode."""

    def __init__(self):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        self.model = os.getenv("EMBEDDING_MODEL")
        if not self.model:
            raise ValueError("Configure EMBEDDING_MODEL in the shared root .env")
        self.client = OpenAI(
            base_url=os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or None,
            api_key=os.getenv("EMBEDDING_API_KEY")
            or os.getenv("LLM_API_KEY")
            or "EMPTY",
            timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        entries = sorted(response.data, key=lambda e: e.index)
        if [e.index for e in entries] != list(range(len(texts))):
            raise ValueError("Embedding provider returned wrong batch indices")
        return [e.embedding for e in entries]


class DenseIndex:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder):
        if not chunks or len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("Need unique, nonempty chunks")
        self.chunks = list(chunks)
        self.embedder = embedder
        data = embedder.embed([chunk.contextual_text for chunk in chunks])
        if len(data) != len(chunks):
            raise ValueError("Embedding batch count mismatch")
        self.vectors = [unit(vector) for vector in data]
        self.dimensions = len(self.vectors[0])
        if any(len(vector) != self.dimensions for vector in self.vectors):
            raise ValueError("Embedding dimensions mismatch")

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
        batch = self.embedder.embed([question])
        if len(batch) != 1:
            raise ValueError("Expected one query vector")
        query = unit(batch[0])
        if len(query) != self.dimensions:
            raise ValueError("Query vector dimensions mismatch")
        ranked = [
            Hit(chunk, sum(a * b for a, b in zip(query, vector)), "dense")
            for chunk, vector in zip(self.chunks, self.vectors)
            if matches(chunk, tenant_id, canonical)
        ]
        ranked = [hit for hit in ranked if hit.score > 0]
        return sorted(ranked, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]
