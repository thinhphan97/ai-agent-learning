"""Small evidence-retrieval benchmark; not an end-to-end RAG quality score."""

from __future__ import annotations

from dataclasses import dataclass

from contracts import Chunk, ChunkSet
from hierarchy import expand_to_parents
from shared import lexical_vector, similarity


@dataclass(frozen=True)
class Case:
    question: str
    required_quote: str


CASES = (
    Case(
        "How many annual leave days do full-time employees get after probation?",
        "Full-time employees receive twelve days of annual leave after probation.",
    ),
    Case(
        "When do password reset links expire?",
        "Password reset links expire after fifteen minutes.",
    ),
    Case(
        "Are activated digital subscriptions refundable?",
        "Activated digital subscriptions are not refundable.",
    ),
    Case(
        "What does the manager approve for a planned trip?",
        "Annual leave must be approved by the employee's manager before the trip.",
    ),
)


def retrieval_context(
    chunks: ChunkSet,
    question: str,
    *,
    top_k: int = 3,
    max_words: int = 150,
) -> list[Chunk]:
    if (
        type(top_k) is not int
        or top_k < 1
        or type(max_words) is not int
        or max_words < 1
    ):
        raise ValueError("Invalid retrieval budget")
    q = lexical_vector(question)
    ranked = sorted(
        chunks.chunks,
        key=lambda node: (-similarity(q, lexical_vector(node.text)), node.id),
    )
    selected, total = [], 0
    for node in ranked[:top_k]:
        if chunks.strategy == "parent-child":
            parents = expand_to_parents([node], chunks, max_chars=100_000)
            choices = parents if parents else [node]
        else:
            choices = [node]
        for choice in choices:
            if any(old.id == choice.id for old in selected):
                continue
            count = len(choice.text.split())
            if total + count <= max_words:
                selected.append(choice)
                total += count
    return selected


def assess(chunks: ChunkSet, *, cases: tuple[Case, ...] = CASES) -> dict[str, object]:
    if not cases or not chunks.chunks:
        raise ValueError("Need cases and chunks")
    details = []
    for case in cases:
        selected = retrieval_context(chunks, case.question)
        combined = "\n".join(node.text.casefold() for node in selected)
        found = case.required_quote.casefold() in combined
        details.append(
            {
                "question": case.question,
                "required_quote": case.required_quote,
                "found": found,
                "retrieved_ids": [node.id for node in selected],
                "context_words": sum(len(node.text.split()) for node in selected),
            }
        )
    return {
        "strategy": chunks.strategy,
        "questions": len(details),
        "evidence_coverage_at_3": round(
            sum(int(x["found"]) for x in details) / len(details), 4
        ),
        "indexed_chunks": len(chunks.chunks),
        "avg_index_words": round(
            sum(len(node.text.split()) for node in chunks.chunks) / len(chunks.chunks),
            2,
        ),
        "cases": details,
    }
