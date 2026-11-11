"""Provider-neutral records for vectors, metadata, search and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: tuple[float, ...]
    text: str
    metadata: Mapping[str, str]
    embedding_model: str

    def __post_init__(self) -> None:
        if not self.id or not self.text.strip() or not self.embedding_model:
            raise ValueError("Record ID, text and embedding model are required")


@dataclass(frozen=True)
class SearchHit:
    record: VectorRecord
    score: float
    rank: int = 0


@dataclass(frozen=True)
class SearchStats:
    distance_evaluations: int
    visited_nodes: int
    candidates_considered: int
