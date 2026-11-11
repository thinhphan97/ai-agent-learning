"""Provider-neutral immutable contracts for Lab 12.

Aspect labels are a curated synthetic teaching fixture, NOT model-inferred facts.
In production they need trusted metadata, a classifier, or verified annotations.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parent:
    id: str
    source: str
    section: str
    title: str
    version: str
    tenant_id: str
    aspect: str
    text: str


@dataclass(frozen=True)
class Child:
    id: str
    parent_id: str
    source: str
    section: str
    version: str
    tenant_id: str
    aspect: str
    text: str
    contextual_note: str = ""

    @property
    def indexed_text(self) -> str:
        return f"{self.contextual_note}\n{self.text}" if self.contextual_note else self.text


@dataclass(frozen=True)
class Hit:
    child: Child
    score: float
    rank: int
    stage: str


@dataclass(frozen=True)
class RerankedHit:
    child: Child
    retrieval_rank: int
    retrieval_score: float
    score: float
    rank: int


@dataclass(frozen=True)
class Evidence:
    id: str
    child_id: str
    parent_id: str
    source: str
    section: str
    version: str
    aspect: str
    content: str
    tokens: int


@dataclass(frozen=True)
class ContextResult:
    text: str
    evidence: tuple[Evidence, ...]
    tokens_used: int
    token_budget: int
    dropped: tuple[tuple[str, str], ...]  # (child_id, reason)
    missing_aspects: tuple[str, ...]
    token_counter: str
