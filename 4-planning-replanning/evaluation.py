"""Evidence-based goal evaluation for Lab 4.

Structural plan validity and semantic goal satisfaction are separate concerns:
count_orders can complete successfully while failing a paid-revenue request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GoalEvaluation:
    satisfied: bool
    code: str
    detail: str


class ProgressEvaluator:
    """Deterministic evaluator for this lab's paid-revenue objective."""

    def evaluate(self, evidence: dict[str, dict[str, Any]]) -> GoalEvaluation:
        customer = evidence.get("get_customer")
        if customer is None:
            return GoalEvaluation(False, "IN_PROGRESS", "Customer has not been resolved")
        if customer.get("customer") is None:
            return GoalEvaluation(True, "CUSTOMER_NOT_FOUND", "Customer lookup found no match")

        orders = evidence.get("get_orders")
        total = evidence.get("sum_amounts")
        if total is not None:
            if orders is None or not isinstance(orders.get("orders"), list):
                return GoalEvaluation(False, "EVIDENCE_INCONSISTENT", "Missing order evidence")
            expected = sum(order["amount"] for order in orders["orders"])
            if type(total.get("total")) is not int or total["total"] != expected:
                return GoalEvaluation(
                    False, "EVIDENCE_INCONSISTENT", "Computed revenue does not match order evidence"
                )
            return GoalEvaluation(True, "GOAL_SATISFIED", "Paid revenue was computed and checked")

        if "count_orders" in evidence:
            return GoalEvaluation(
                False,
                "GOAL_METRIC_MISMATCH",
                "count_orders returns the number of paid orders, not their revenue. "
                "Use sum_amounts with amounts from get_orders.",
            )

        return GoalEvaluation(False, "IN_PROGRESS", "Waiting for required revenue evidence")
