"""Private-to-evaluator answer keys for an independent, deterministic fixture set.

Do not import this module from model.py, runtime.py, or tools.py.
The harness passes case.question ONLY to the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts import Answer

DATASET_VERSION = "lab6-eval-v1"


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    question: str
    target_customer_id: int | None
    expected: Answer


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        "revenue_lan",
        "revenue",
        "What is the paid order revenue for customer_id=1?",
        1,
        Answer("answered", 200000, "VND"),
    ),
    EvalCase(
        "count_lan",
        "count",
        "How many paid orders does customer_id=1 have?",
        1,
        Answer("answered", 2, "orders"),
    ),
    EvalCase(
        "revenue_minh",
        "revenue",
        "Calculate total paid order revenue for customer_id=2.",
        2,
        Answer("answered", 45000, "VND"),
    ),
    EvalCase(
        "count_minh",
        "count",
        "Count paid orders for customer_id=2.",
        2,
        Answer("answered", 1, "orders"),
    ),
    EvalCase(
        "revenue_nhi_zero",
        "revenue",
        "Calculate paid order revenue for customer_id=3.",
        3,
        Answer("answered", 0, "VND"),
    ),
    EvalCase(
        "count_nhi_zero",
        "count",
        "How many paid orders does customer_id=3 have?",
        3,
        Answer("answered", 0, "orders"),
    ),
    EvalCase(
        "unknown_customer",
        "no_data",
        "Calculate paid order revenue for customer_id=999.",
        999,
        Answer("no_data", None, None),
    ),
    EvalCase(
        "unsupported_forecast",
        "unsupported",
        "Predict tomorrow's paid order revenue for customer_id=1.",
        None,
        Answer("abstain", None, None),
    ),
    EvalCase(
        "missing_identifier",
        "unsupported",
        "How much paid order revenue did the customer generate?",
        None,
        Answer("abstain", None, None),
    ),
)


def select_cases(case_id: str | None = None) -> tuple[EvalCase, ...]:
    if case_id is None:
        return CASES
    selected = tuple(case for case in CASES if case.id == case_id)
    if not selected:
        raise ValueError(f"Unknown case ID: {case_id}")
    return selected
