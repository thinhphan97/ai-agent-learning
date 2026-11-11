"""Independent oracle: separate run completion, answer accuracy, evidence, policy.

The AgentRunner sees only the question, never EvalCase.expected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cases import EvalCase
from contracts import Answer, Episode, Limits


@dataclass(frozen=True)
class CaseScore:
    id: str
    category: str
    question: str
    expected: Answer
    observed: Answer | None
    status: str
    completed: bool
    answer_exact: bool
    evidence_grounded: bool
    tool_policy_ok: bool
    budget_ok: bool
    task_success: bool
    failure_reasons: tuple[str, ...]
    model_calls: int
    tool_attempts: int
    elapsed_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Evaluator:
    """Check an episode AFTER execution, with labels held out from the agent."""

    def score(self, case: EvalCase, episode: Episode, limits: Limits) -> CaseScore:
        completed = episode.status == "completed" and episode.answer is not None
        answer_exact = completed and episode.answer == case.expected
        grounded = self._grounded(case, episode)
        tool_policy_ok = episode.status != "tool_error" and not any(
            event.get("kind") == "tool_observation"
            and not event.get("observation", {}).get("ok", False)
            for event in episode.events
        )
        budget_ok = (
            episode.model_calls <= limits.max_model_calls
            and episode.tool_attempts <= limits.max_tool_attempts
        )
        success = all((completed, answer_exact, grounded, tool_policy_ok, budget_ok))
        reasons = []
        if not completed:
            reasons.append("RUNTIME_INCOMPLETE")
        if not answer_exact:
            reasons.append("ANSWER_MISMATCH")
        if not grounded:
            reasons.append("UNGROUNDED_ANSWER")
        if not tool_policy_ok:
            reasons.append("TOOL_POLICY_VIOLATION")
        if not budget_ok:
            reasons.append("BUDGET_VIOLATION")
        return CaseScore(
            id=case.id,
            category=case.category,
            question=case.question,
            expected=case.expected,
            observed=episode.answer,
            status=episode.status,
            completed=completed,
            answer_exact=answer_exact,
            evidence_grounded=grounded,
            tool_policy_ok=tool_policy_ok,
            budget_ok=budget_ok,
            task_success=success,
            failure_reasons=tuple(reasons),
            model_calls=episode.model_calls,
            tool_attempts=episode.tool_attempts,
            elapsed_ms=episode.elapsed_ms,
        )

    def _grounded(self, case: EvalCase, episode: Episode) -> bool:
        answer = episode.answer
        if answer is None or episode.status != "completed":
            return False
        if answer.status == "abstain":
            return case.category == "unsupported" and not episode.evidence

        customer_data = episode.evidence.get("get_customer")
        if not isinstance(customer_data, dict):
            return False
        customer = customer_data.get("customer")
        if answer.status == "no_data":
            return (
                case.category == "no_data"
                and customer_data.get("found") is False
                and customer is None
                and set(episode.evidence) == {"get_customer"}
            )
        if (
            answer.status != "answered"
            or not isinstance(customer, dict)
            or customer.get("id") != case.target_customer_id
            or customer_data.get("found") is not True
        ):
            return False

        order_data = episode.evidence.get("get_orders")
        if not isinstance(order_data, dict):
            return False
        orders = order_data.get("orders")
        if not isinstance(orders, list) or any(
            not isinstance(order, dict)
            or order.get("customer") != customer.get("name")
            or order.get("status") != "paid"
            or type(order.get("amount")) is not int
            or order["amount"] < 0
            or type(order.get("id")) is not int
            for order in orders
        ):
            return False

        if answer.unit == "VND":
            result = episode.evidence.get("sum_amounts")
            expected_total = sum(order["amount"] for order in orders)
            return (
                isinstance(result, dict)
                and type(result.get("total")) is int
                and result["total"] == expected_total == answer.value
            )
        if answer.unit == "orders":
            result = episode.evidence.get("count_orders")
            return (
                isinstance(result, dict)
                and type(result.get("count")) is int
                and result["count"] == len(orders) == answer.value
            )
        return False
