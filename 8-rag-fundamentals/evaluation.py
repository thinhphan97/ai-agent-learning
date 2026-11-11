"""Small independently specified retrieval/answer fixture set.

The retriever and generator never import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pipeline import RAGAnswer


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_document: str | None
    required_phrase: str | None


CASES = (
    EvalCase("How many days can I return unopened hardware?", "returns", "30 days"),
    EvalCase("How much does expedited shipping cost?", "shipping", "50000 VND"),
    EvalCase("When is the service desk open?", "support", "09:00 to 18:00"),
    EvalCase("When do password reset links expire?", "security", "15 minutes"),
    EvalCase("Do listed VND prices include VAT?", "billing", "include VAT"),
    EvalCase(
        "Can I refund an activated digital subscription?", "returns", "not refundable"
    ),
    EvalCase("Is international shipping available?", "shipping", "unavailable"),
    EvalCase("What is the employee cafeteria menu?", None, None),
)


@dataclass(frozen=True)
class EvalSummary:
    questions: int
    source_recall_at_k: float
    grounded_answer_accuracy: float
    negative_abstention_accuracy: float


def evaluate(
    answer: Callable[[str], RAGAnswer], cases: tuple[EvalCase, ...] = CASES
) -> tuple[EvalSummary, list[dict[str, object]]]:
    if not cases:
        raise ValueError("Cannot evaluate an empty suite")
    outcomes: list[dict[str, object]] = []
    positives, found, correct, negatives, abstained = 0, 0, 0, 0, 0
    for case in cases:
        result = answer(case.question)
        if case.expected_document is not None:
            positives += 1
            retrieved = case.expected_document in result.source_documents
            found += int(retrieved)
            supported = (
                result.status == "answered"
                and any(
                    citation.startswith(case.expected_document + ":")
                    for citation in result.citations
                )
                and case.required_phrase is not None
                and case.required_phrase.casefold() in result.text.casefold()
            )
            correct += int(supported)
        else:
            negatives += 1
            retrieved = False
            supported = result.status == "abstain"
            abstained += int(supported)
        outcomes.append(
            {
                "question": case.question,
                "expected_document": case.expected_document,
                "retrieved": retrieved,
                "answer_check": supported,
                "status": result.status,
                "reason": result.reason,
            }
        )

    return (
        EvalSummary(
            questions=len(cases),
            source_recall_at_k=round(found / positives, 4) if positives else 1.0,
            grounded_answer_accuracy=(
                round(correct / positives, 4) if positives else 1.0
            ),
            negative_abstention_accuracy=(
                round(abstained / negatives, 4) if negatives else 1.0
            ),
        ),
        outcomes,
    )
