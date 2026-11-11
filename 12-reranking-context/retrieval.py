"""Pure Python independent BM25 + hashed pseudo-dense retrieval and RRF.

Offline dense hashing is lexical feature hashing, NOT a neural semantic model.
Both retrieval paths are tenant-scoped before scoring or fusion.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from contracts import Child, Hit
from corpus import Corpus


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:_[^\W_]+)*", text.casefold())


def _embedding(text: str, dims: int = 128) -> tuple[float, ...]:
    counts = Counter(tokenize(text))
    vector = [0.0] * dims
    for word, count in counts.items():
        idx = int.from_bytes(
            hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "big"
        ) % dims
        vector[idx] += float(count)
    norm = math.sqrt(sum(v * v for v in vector))
    return tuple(v / norm for v in vector) if norm else tuple(vector)


def _bm25(query: str, children: list[Child]) -> list[tuple[Child, float]]:
    terms = tokenize(query)
    if not terms:
        return []
    docs = [Counter(tokenize(c.indexed_text)) for c in children]
    length = [sum(doc.values()) for doc in docs]
    avgdl = sum(length) / len(length) if length else 1.0
    df = Counter(term for doc in docs for term in doc)
    k1, b, n = 1.5, 0.75, len(docs)
    scores = []
    for child, document, dl in zip(children, docs, length):
        score = 0.0
        for term in set(terms):
            tf = document.get(term, 0)
            if not tf:
                continue
            idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf * (k1 + 1) / (
                tf + k1 * (1 - b + b * dl / max(avgdl, 1))
            )
        if score > 0:
            scores.append((child, score))
    return sorted(scores, key=lambda t: (-t[1], t[0].id))


def _dense(query: str, children: list[Child]) -> list[tuple[Child, float]]:
    q = _embedding(query)
    values = []
    for child in children:
        d = _embedding(child.indexed_text)
        score = sum(a * b for a, b in zip(q, d))
        if score > 0:
            values.append((child, score))
    return sorted(values, key=lambda t: (-t[1], t[0].id))


class OpenAIEmbedder:
    """Opt-in genuine embedding API, using only the workspace-root .env."""

    def __init__(self):
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        model = os.getenv("EMBEDDING_MODEL")
        if not model:
            raise ValueError("EMBEDDING_MODEL required for actual dense embeddings")
        self.model_id = model
        self.client = OpenAI(
            base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL") or None,
            api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or "EMPTY",
            timeout=float(os.getenv("LLM_TIMEOUT", "30")),
            max_retries=0,
        )

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model_id, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        if [x.index for x in ordered] != list(range(len(texts))):
            raise ValueError("Embedding provider returned invalid batch indices")
        dimensions = {len(x.embedding) for x in ordered}
        if len(dimensions) != 1 or not next(iter(dimensions)):
            raise ValueError("Inconsistent embedding dimensions")
        vectors = []
        for item in ordered:
            vector = tuple(float(v) for v in item.embedding)
            if not all(math.isfinite(x) for x in vector):
                raise ValueError("Embedding provider returned non-finite vector")
            norm = math.sqrt(sum(x * x for x in vector))
            if norm == 0:
                raise ValueError("Embedding provider returned zero vector")
            vectors.append(tuple(x / norm for x in vector))
        return vectors


def _dense_real(
    query: str, children: list[Child], embedder: OpenAIEmbedder
) -> list[tuple[Child, float]]:
    # Retrieval filters tenant scope BEFORE external provider call.
    vectors = embedder.embed([query] + [child.indexed_text for child in children])
    if len(vectors) != len(children) + 1:
        raise ValueError("Embedding provider returned wrong vector count")
    q, docs = vectors[0], vectors[1:]
    if any(len(q) != len(v) for v in docs):
        raise ValueError("Query and index dimensions disagree")
    return sorted(
        [(child, sum(a * b for a, b in zip(q, doc)))
         for child, doc in zip(children, docs)],
        key=lambda p: (-p[1], p[0].id),
    )


class Retriever:
    def __init__(self, corpus: Corpus, *, dense_provider: str = "toy"):
        if dense_provider not in {"toy", "api"}:
            raise ValueError("Unsupported dense provider")
        self.corpus = corpus
        self.dense_provider = dense_provider
        self.embedder = OpenAIEmbedder() if dense_provider == "api" else None

    def search(
        self, question: str, *, tenant_id: str, top_k: int = 50,
        mode: str = "hybrid",
    ) -> list[Hit]:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unsupported retrieval mode")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Nonempty question required")
        if type(top_k) is not int or not 1 <= top_k <= 200:
            raise ValueError("Retrieval top_k must be within 1..200")
        children = list(self.corpus.authorized(tenant_id))
        if not children:
            return []
        bm25 = _bm25(question, children)[:top_k] if mode != "dense" else []
        if mode == "bm25":
            dense = []
        elif self.embedder is not None:
            dense = _dense_real(question, children, self.embedder)[:top_k]
        else:
            dense = _dense(question, children)[:top_k]
        if mode == "bm25":
            ranked = bm25
        elif mode == "dense":
            ranked = dense
        else:
            # Reciprocal Rank Fusion avoids combining incomparable raw scores.
            fusion: dict[str, float] = {}
            lookup = {c.id: c for c in children}
            for group in (bm25, dense):
                for rank, (child, _) in enumerate(group, 1):
                    fusion[child.id] = fusion.get(child.id, 0.0) + 1.0 / (60 + rank)
            ranked = sorted(
                ((lookup[id], score) for id, score in fusion.items()),
                key=lambda t: (-t[1], t[0].id),
            )[:top_k]
        return [
            Hit(child, float(score), rank, mode)
            for rank, (child, score) in enumerate(ranked, 1)
        ]