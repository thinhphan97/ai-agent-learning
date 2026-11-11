"""Offline deterministic lexical feature hashing and explicit real API embedding.

Hashing is NOT an AI semantic embedding model. Do not use it to claim semantic
paraphrase retrieval quality. It exists to make vector indexing reproducible.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from metrics import normalize


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    model_id = "toy-hashing-v1"

    def __init__(self, dimensions: int = 128) -> None:
        if type(dimensions) is not int or dimensions < 16:
            raise ValueError("Hash embedding dimensions must be >= 16")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Embedding text must be nonempty")
            weights = [0.0] * self.dimensions
            terms = re.findall(r"[^\W_]+(?:_[^\W_]+)*", text.casefold())
            for term, count in Counter(terms).items():
                idx = (
                    int.from_bytes(
                        hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(),
                        "big",
                    )
                    % self.dimensions
                )
                weights[idx] += float(count)
            result.append(list(normalize(weights)))
        return result


class OpenAIEmbedder:
    """Explicitly opt in to a real OpenAI-compatible embedding API."""

    def __init__(self) -> None:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        model = os.getenv("EMBEDDING_MODEL")
        if not model:
            raise ValueError("EMBEDDING_MODEL must be configured in the root .env")
        self.model_id = model
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
        if not texts:
            return []
        result = self.client.embeddings.create(model=self.model_id, input=texts)
        entries = sorted(result.data, key=lambda x: x.index)
        if [item.index for item in entries] != list(range(len(texts))):
            raise ValueError("Provider embedding batch indices mismatch")
        return [list(item.embedding) for item in entries]
