"""Lab 14 provider-independent contracts. Runtime never consumes gold labels."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source_type: str
    tenant_id: str
    valid_from: str
    valid_until: str | None = None


@dataclass(frozen=True)
class Request:
    question: str
    tenant_id: str
    as_of_date: str
    allowed_sources: tuple[str, ...] = ("docs", "code", "sql")


@dataclass(frozen=True)
class Analysis:
    intent: str
    aspects: tuple[str, ...]
    anchors: tuple[str, ...]


@dataclass(frozen=True)
class SearchStep:
    id: str
    aspect: str
    query: str
    source: str
    depends_on: tuple[str, ...] = ()
    action: str = "retrieve"


@dataclass(frozen=True)
class SearchPlan:
    steps: tuple[SearchStep, ...]
    strategy: str


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    rank: int
    query: str
    source: str
    aspect: str


@dataclass(frozen=True)
class Evidence:
    id: str
    doc_id: str
    source_type: str
    aspect: str
    content: str


@dataclass(frozen=True)
class Budgets:
    max_subqueries: int = 5
    max_rounds: int = 2
    max_search_calls: int = 12
    max_hits_per_search: int = 5
    max_evidence: int = 6
    max_llm_calls: int = 3

    def validate(self) -> None:
        for name in (
            "max_subqueries", "max_rounds", "max_search_calls",
            "max_hits_per_search", "max_evidence", "max_llm_calls",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1 or value > 100:
                raise ValueError(f"{name} must be an integer within 1..100")