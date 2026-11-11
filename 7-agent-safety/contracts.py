"""Typed contracts shared between the Lab 7 modules."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelTurn:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Intent:
    kind: str  # "read" or "write"
    customer_id: int
    note: str | None = None


@dataclass(frozen=True)
class ApprovalGrant:
    """Issued by trusted CLI/application code, NOT by the model."""

    principal: str
    customer_id: int
    note: str


@dataclass(frozen=True)
class Limits:
    max_model_calls: int = 5
    max_tool_attempts: int = 3

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or n < 1
            for n in (self.max_model_calls, self.max_tool_attempts)
        ):
            raise ValueError("Budgets must be positive integers")


@dataclass(frozen=True)
class FinalAnswer:
    status: str
    customer_id: int
    name: str

    @classmethod
    def parse(cls, content: str | None) -> "FinalAnswer":
        if not isinstance(content, str):
            raise ValueError("Final response must be JSON")
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Final response must be JSON") from exc
        if (
            not isinstance(obj, dict)
            or set(obj) != {"status", "customer_id", "name"}
            or obj["status"] not in ("read", "updated")
            or type(obj["customer_id"]) is not int
            or not isinstance(obj["name"], str)
        ):
            raise ValueError("Invalid final answer contract")
        return cls(**obj)


@dataclass
class RunResult:
    status: str = "created"
    answer: FinalAnswer | None = None
    code: str | None = None
    model_calls: int = 0
    tool_attempts: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
