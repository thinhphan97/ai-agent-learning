"""Shared, provider-neutral contracts for Lab 10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    contextual_text: str
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.id or not self.text.strip() or not self.contextual_text.strip():
            raise ValueError("Chunk id and texts must be nonempty")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    method: str
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_ids: tuple[str, ...]
    filters: Mapping[str, str]
    tenant_id: str = "tenant_a"
