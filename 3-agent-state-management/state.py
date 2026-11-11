"""Lab 3: explicit agent state, lifecycle events and JSON checkpoints.

Checkpoint boundaries are settled iterations (after a tool result or terminal
decision). This is a teaching implementation, not exactly-once persistence.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from uuid import uuid4
from typing import Any


class Status(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_MODEL = "waiting_for_model"
    WAITING_FOR_TOOL = "waiting_for_tool"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"


ALLOWED: dict[Status, frozenset[Status]] = {
    Status.CREATED: frozenset({Status.RUNNING, Status.CANCELLED}),
    Status.RUNNING: frozenset(
        {
            Status.WAITING_FOR_MODEL,
            Status.WAITING_FOR_TOOL,
            Status.PAUSED,
            Status.COMPLETED,
            Status.FAILED,
            Status.BUDGET_EXCEEDED,
            Status.CANCELLED,
        }
    ),
    Status.WAITING_FOR_MODEL: frozenset(
        {
            Status.RUNNING,
            Status.FAILED,
            Status.BUDGET_EXCEEDED,
            Status.CANCELLED,
        }
    ),
    Status.WAITING_FOR_TOOL: frozenset(
        {Status.RUNNING, Status.FAILED, Status.CANCELLED}
    ),
    Status.PAUSED: frozenset({Status.RUNNING, Status.CANCELLED}),
    Status.COMPLETED: frozenset(),
    Status.FAILED: frozenset(),
    Status.BUDGET_EXCEEDED: frozenset(),
    Status.CANCELLED: frozenset(),
}
TERMINAL = {Status.COMPLETED, Status.FAILED, Status.BUDGET_EXCEEDED, Status.CANCELLED}


@dataclass
class AgentState:
    run_id: str
    request: str
    status: Status
    messages: list[dict[str, Any]]
    successful: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    used_call_ids: list[str] = field(default_factory=list)
    model_calls: int = 0
    tool_attempts: int = 0
    errors: int = 0
    answer: str = ""
    failure_reason: str = ""

    @classmethod
    def new(cls, request: str) -> "AgentState":
        if not request.strip():
            raise ValueError("request cannot be empty")
        state = cls(
            run_id=uuid4().hex,
            request=request,
            status=Status.CREATED,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a revenue assistant. Use one tool call per model turn. "
                        "Resolve customer_id with get_customer, then use get_orders and "
                        "sum_amounts to compute paid revenue. Use tool observations, "
                        "do not guess results. Return a final answer only when you have "
                        "a computed total or have confirmed the customer does not exist."
                    ),
                },
                {"role": "user", "content": request},
            ],
        )
        state.emit("run_created")
        return state

    def emit(self, kind: str, **data: Any) -> None:
        self.events.append({"seq": len(self.events) + 1, "kind": kind, **data})

    def transition(self, target: Status, *, reason: str = "") -> None:
        if target not in ALLOWED[self.status]:
            raise ValueError(
                f"Invalid transition: {self.status.value} -> {target.value}"
            )
        previous = self.status
        self.status = target
        self.emit(
            "state_changed",
            previous=previous.value,
            current=target.value,
            reason=reason,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "request": self.request,
            "status": self.status.value,
            "messages": self.messages,
            "successful": self.successful,
            "events": self.events,
            "used_call_ids": self.used_call_ids,
            "model_calls": self.model_calls,
            "tool_attempts": self.tool_attempts,
            "errors": self.errors,
            "answer": self.answer,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_snapshot(cls, raw: dict[str, Any]) -> "AgentState":
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("Unsupported or malformed checkpoint")
        try:
            state = cls(
                run_id=raw["run_id"],
                request=raw["request"],
                status=Status(raw["status"]),
                messages=raw["messages"],
                successful=raw["successful"],
                events=raw["events"],
                used_call_ids=raw["used_call_ids"],
                model_calls=raw["model_calls"],
                tool_attempts=raw["tool_attempts"],
                errors=raw["errors"],
                answer=raw["answer"],
                failure_reason=raw["failure_reason"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed checkpoint fields") from exc

        if (
            not isinstance(state.run_id, str)
            or not isinstance(state.request, str)
            or not isinstance(state.messages, list)
            or not isinstance(state.events, list)
            or not isinstance(state.successful, dict)
            or not isinstance(state.used_call_ids, list)
            or any(
                type(n) is not int or n < 0
                for n in (state.model_calls, state.tool_attempts, state.errors)
            )
        ):
            raise ValueError("Malformed checkpoint field types")
        if [
            event.get("seq") for event in state.events if isinstance(event, dict)
        ] != list(range(1, len(state.events) + 1)):
            raise ValueError("Malformed checkpoint event sequence")
        return state


class CheckpointStore:
    """Atomic per-file JSON replacement, with no credentials or SDK objects."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def save(self, state: AgentState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as fp:
                temporary = fp.name
                os.chmod(temporary, 0o600)
                json.dump(state.snapshot(), fp, ensure_ascii=False, indent=2)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def load(self) -> AgentState:
        try:
            with self.path.open(encoding="utf-8") as fp:
                return AgentState.from_snapshot(json.load(fp))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError("Unreadable checkpoint JSON") from exc
