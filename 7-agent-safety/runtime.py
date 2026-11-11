"""Lab 7 safe agent loop: enforce tool policy and verify final actions.

The model is NOT the security authority. No real external side effects exist.
"""

from __future__ import annotations

import json
from typing import Any

from audit import AuditTrail
from contracts import FinalAnswer, Limits, ModelTurn, RunResult
from model import Model, SYSTEM_PROMPT
from policy import SafetyPolicy
from tools import ToolRegistry


class SafeAgentRunner:
    def __init__(
        self,
        model: Model,
        registry: ToolRegistry,
        policy: SafetyPolicy,
        *,
        limits: Limits | None = None,
    ):
        self.model = model
        self.registry = registry
        self.policy = policy
        self.limits = limits or Limits()

    def run(self, request: str) -> RunResult:
        result = RunResult(status="running")
        audit = AuditTrail()
        scope = self.policy.check_scope()

        def stop(
            status: str, code: str, answer: FinalAnswer | None = None
        ) -> RunResult:
            result.status = status
            result.code = code
            result.answer = answer
            audit.record("run_stopped", code=code)
            result.events = audit.events
            return result

        if not scope.allowed:
            audit.record("scope_denied", code=scope.code)
            return stop("denied", scope.code)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ]
        successful: dict[str, dict[str, Any]] = {}
        used_ids: set[str] = set()

        for _ in range(self.limits.max_model_calls):
            result.model_calls += 1
            audit.record("model_requested")
            try:
                turn: ModelTurn = self.model.complete(messages, self.registry.schemas())
            except Exception:
                return stop("model_error", "MODEL_REQUEST_FAILED")

            if turn.tool_calls:
                if len(turn.tool_calls) != 1:
                    return stop("denied", "MULTIPLE_TOOL_CALLS")
                call = turn.tool_calls[0]
                if not call.id or call.id in used_ids:
                    return stop("denied", "INVALID_TOOL_CALL_ID")
                if result.tool_attempts >= self.limits.max_tool_attempts:
                    return stop("budget_exceeded", "TOOL_ATTEMPT_LIMIT")
                result.tool_attempts += 1
                used_ids.add(call.id)
                outcome = self.registry.execute(call, successful)
                audit.record(
                    "tool_approved" if outcome["ok"] else "tool_denied",
                    code="ALLOWED" if outcome["ok"] else outcome["code"],
                    tool=call.name,
                )
                if not outcome["ok"]:
                    return stop("denied", outcome["code"])
                successful[call.name] = outcome["data"]
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": turn.content,
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments,
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": json.dumps(outcome),
                        },
                    ]
                )
                continue

            try:
                final = FinalAnswer.parse(turn.content)
            except ValueError:
                return stop("denied", "INVALID_FINAL")

            customer = successful.get("get_customer")
            if (
                customer is None
                or final.customer_id != self.policy.intent.customer_id
                or final.name != customer["name"]
            ):
                return stop("denied", "UNGROUNDED_FINAL")
            expected_status = (
                "updated" if self.policy.intent.kind == "write" else "read"
            )
            if final.status != expected_status:
                return stop("denied", "FINAL_ACTION_MISMATCH")
            if self.policy.intent.kind == "write":
                saved = successful.get("set_customer_note")
                current = self.registry.store.get(self.policy.intent.customer_id)
                if (
                    saved is None
                    or not saved.get("updated")
                    or current is None
                    or current["note"] != self.policy.intent.note
                    or self.registry.store.write_count != 1
                ):
                    return stop("denied", "UNVERIFIED_WRITE")
            elif self.registry.store.write_count != 0:
                return stop("denied", "READ_ONLY_VIOLATION")
            audit.record("final_verified")
            return stop("completed", "VERIFIED", final)

        return stop("budget_exceeded", "MODEL_CALL_LIMIT")
