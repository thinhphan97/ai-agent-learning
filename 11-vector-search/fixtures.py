"""Synthetic, small document corpus and reproducible numerical benchmark vectors."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from contracts import VectorRecord
from embeddings import Embedder
from metadata import normalize_metadata
from metrics import normalize

SOURCE = Path(__file__).parent / "data" / "documents.json"


def load_documents(embedder: Embedder) -> list[VectorRecord]:
    documents = json.loads(SOURCE.read_text(encoding="utf-8"))
    texts = [
        (
            f"Company: {doc['metadata']['company_id']}. "
            f"Period: {doc['metadata']['period']}. "
            f"Section: {doc['metadata']['section']}. {doc['text']}"
        )
        for doc in documents
    ]
    vectors = embedder.embed(texts)
    if len(vectors) != len(documents):
        raise ValueError("Embedding provider returned wrong batch size")
    return [
        VectorRecord(
            id=item["id"],
            vector=tuple(vector),
            text=item["text"],
            metadata=normalize_metadata(item["metadata"]),
            embedding_model=embedder.model_id,
        )
        for item, vector in zip(documents, vectors)
    ]


def synthetic_vectors(
    count: int = 240, dimensions: int = 12, seed: int = 2026
) -> tuple[list[VectorRecord], list[tuple[float, ...]]]:
    """Dense numeric clusters for ANN-vs-exact, not natural-language semantics."""
    if count < 20 or dimensions < 4:
        raise ValueError("Need at least 20 points and 4 dimensions")
    rng = random.Random(seed)
    centers = [
        normalize([rng.gauss(0, 1) for _ in range(dimensions)]) for _ in range(8)
    ]
    records = []
    for i in range(count):
        center = centers[i % len(centers)]
        vector = normalize([v + rng.gauss(0, 0.14) for v in center])
        records.append(
            VectorRecord(
                id=f"point_{i:05d}",
                vector=vector,
                text=f"Synthetic vector point {i}",
                metadata=normalize_metadata(
                    {
                        "tenant_id": "tenant_a",
                        "company_id": "FPT" if i % 2 == 0 else "HPG",
                        "period": "2025-Q2",
                        "section": "bench",
                        "source": "synthetic_numerical_fixture",
                    }
                ),
                embedding_model="synthetic-cluster-v1",
            )
        )
    queries = [
        normalize([v + rng.gauss(0, 0.09) for v in centers[i % len(centers)]])
        for i in range(16)
    ]
    return records, queries
