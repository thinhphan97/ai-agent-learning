"""Shared runtime contracts. No oracle labels or evaluation fixtures belong here."""

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
class Answer:
    status: str
    value: int | None
    unit: str | None

    @classmethod
    def parse(cls, content: str | None) -> "Answer":
        if not isinstance(content, str):
            raise ValueError("Answer is missing")
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Answer must be JSON") from exc
        if not isinstance(obj, dict) or set(obj) != {"status", "value", "unit"}:
            raise ValueError("Expected exactly status, value and unit")
        status, value, unit = obj["status"], obj["value"], obj["unit"]
        if status == "answered":
            if type(value) is not int or unit not in {"VND", "orders"}:
                raise ValueError("Answered result needs an integer and supported unit")
        elif status in {"no_data", "abstain"}:
            if value is not None or unit is not None:
                raise ValueError("No-data/abstain cannot include a numeric claim")
        else:
            raise ValueError("Unknown answer status")
        return cls(status, value, unit)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Limits:
    max_model_calls: int = 8
    max_tool_attempts: int = 4

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or n < 1
            for n in (self.max_model_calls, self.max_tool_attempts)
        ):
            raise ValueError("Limits must be positive integers")


@dataclass
class Episode:
    question: str
    status: str = "created"
    answer: Answer | None = None
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    tool_attempts: int = 0
    error_code: str | None = None
    elapsed_ms: float = 0.0

    def log(self, kind: str, **data: Any) -> None:
        self.events.append({"seq": len(self.events) + 1, "kind": kind, **data})

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["elapsed_ms"] = round(self.elapsed_ms, 2)
        return result
