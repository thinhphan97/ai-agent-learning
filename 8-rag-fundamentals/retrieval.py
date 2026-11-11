"""Retrieval backends: transparent sparse TF-IDF and optional dense embeddings.

Sparse TF-IDF is a lexical vector baseline, NOT a semantic embedding model.
The optional dense backend uses a real provider embedding API.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

from documents import Chunk

# Removing a few English function words helps make a lexical baseline readable.
STOPWORDS = frozenset(
    "a an and are at be by can do does for from how i in is it of on or our "
    "the their there these this to was were what when where which who why will "
    "with within you your after through".split()
)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"\w+", text.casefold())
        if token not in STOPWORDS and len(token) > 1
    ]


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    magnitude = math.sqrt(sum(value * value for value in weights.values()))
    return (
        {key: value / magnitude for key, value in weights.items()} if magnitude else {}
    )


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def retrieve(self, question: str, *, top_k: int, min_score: float) -> list[Hit]: ...


class TfidfRetriever:
    """Simple in-memory cosine search over sparse lexical vectors."""

    def __init__(self, chunks: Sequence[Chunk]):
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        self.chunks = list(chunks)
        tokens = [tokenize(chunk.text) for chunk in chunks]
        frequencies = Counter(term for doc in tokens for term in set(doc))
        n = len(chunks)
        self.idf = {
            term: math.log((n + 1) / (count + 1)) + 1
            for term, count in frequencies.items()
        }
        self.vectors = [
            _normalize(
                {
                    term: (1 + math.log(count)) * self.idf[term]
                    for term, count in Counter(words).items()
                }
            )
            for words in tokens
        ]

    def retrieve(
        self, question: str, *, top_k: int = 3, min_score: float = 0.1
    ) -> list[Hit]:
        validate_search(question, top_k, min_score)
        query_counts = Counter(tokenize(question))
        query = _normalize(
            {
                term: (1 + math.log(count)) * self.idf[term]
                for term, count in query_counts.items()
                if term in self.idf
            }
        )
        if not query:
            return []
        matches = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = sum(value * vector.get(term, 0.0) for term, value in query.items())
            if score >= min_score and score > 0:
                matches.append(Hit(chunk, score))
        return sorted(matches, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]


def validate_search(question: str, top_k: int, min_score: float) -> None:
    if not isinstance(question, str) or not question.strip() or len(question) > 600:
        raise ValueError("Question must be nonempty and no longer than 600 characters")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1 to 10")
    if (
        isinstance(min_score, bool)
        or not isinstance(min_score, (int, float))
        or not math.isfinite(min_score)
        or not 0 <= min_score <= 1
    ):
        raise ValueError("min_score must be finite and in [0, 1]")


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _dense_unit(values: Sequence[float]) -> tuple[float, ...]:
    if not values or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in values
    ):
        raise ValueError("Embedding must contain finite numeric values")
    norm = math.sqrt(sum(float(x) ** 2 for x in values))
    if norm == 0:
        raise ValueError("Embedding vector cannot have zero norm")
    return tuple(float(x) / norm for x in values)


class DenseRetriever:
    """In-memory dense vector search using an injected embedding provider."""

    def __init__(self, chunks: Sequence[Chunk], embeddings: EmbeddingClient):
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        self.chunks = list(chunks)
        self.embeddings = embeddings
        batch = embeddings.embed([chunk.text for chunk in chunks])
        if len(batch) != len(chunks):
            raise ValueError("Embedding provider response count mismatch")
        self.vectors = [_dense_unit(vector) for vector in batch]
        dimensions = len(self.vectors[0])
        if any(len(vector) != dimensions for vector in self.vectors):
            raise ValueError("Inconsistent embedding dimensions")

    def retrieve(
        self, question: str, *, top_k: int = 3, min_score: float = 0.1
    ) -> list[Hit]:
        validate_search(question, top_k, min_score)
        batch = self.embeddings.embed([question])
        if len(batch) != 1:
            raise ValueError("Expected one query embedding")
        query = _dense_unit(batch[0])
        if len(query) != len(self.vectors[0]):
            raise ValueError("Query embedding dimensions do not match index")
        ranked = [
            Hit(chunk, sum(x * y for x, y in zip(query, vector)))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        return sorted(
            (hit for hit in ranked if hit.score >= min_score),
            key=lambda hit: (-hit.score, hit.chunk.id),
        )[:top_k]


def openai_embedding_client() -> EmbeddingClient:
    """Read only the existing workspace-root .env; require a real embedding model."""
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    model = os.getenv("EMBEDDING_MODEL")
    if not model:
        raise ValueError("Set EMBEDDING_MODEL in root .env to use --retriever dense")
    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL") or None,
        api_key=os.getenv("LLM_API_KEY") or "EMPTY",
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
    )

    class OpenAIEmbeddingClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(model=model, input=texts)
            results = sorted(response.data, key=lambda entry: entry.index)
            if len(results) != len(texts):
                raise ValueError("Provider embedding count mismatch")
            if [entry.index for entry in results] != list(range(len(texts))):
                raise ValueError("Provider embedding indices mismatch")
            return [entry.embedding for entry in results]

    return OpenAIEmbeddingClient()
