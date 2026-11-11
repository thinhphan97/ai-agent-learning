"""Independent evaluation contracts for Lab 13.

Evaluation gold and observations are separate. None of the evaluator labels
are passed to a retrieval / generation runtime to influence its output.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    tenant_id: str
    text: str
    valid_from: str
    valid_until: str | None


@dataclass(frozen=True)
class RequiredFact:
    id: str
    description: str
    supporting_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    tenant_id: str
    as_of_date: str
    relevance_grades: dict[str, int]
    required_facts: tuple[RequiredFact, ...]
    reference_answer: str
    answerable: bool


@dataclass(frozen=True)
class ContextRef:
    evidence_id: str
    doc_id: str


@dataclass(frozen=True)
class Trace:
    case_id: str
    tenant_id: str
    retrieved: tuple[str, ...]
    candidates: tuple[str, ...]
    reranked: tuple[str, ...]
    context: tuple[ContextRef, ...]
    answer: str
    abstained: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    # fixture = synthetic profile, external = imported, live = real generation
    provenance: str = "fixture"


@dataclass(frozen=True)
class ClaimLabel:
    span: str
    supporting_doc_ids: tuple[str, ...]
    correct: bool
    covers_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class AnswerLabels:
    answer: str
    claims: tuple[ClaimLabel, ...]
    injection_followed: bool = False
    answer_relevancy: float | None = None
