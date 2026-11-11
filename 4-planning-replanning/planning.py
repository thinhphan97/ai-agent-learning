"""Contracts and independent validation for structured plans in Lab 4."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlanStep:
    id: str
    tool: str
    objective: str
    depends_on: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["depends_on"] = list(self.depends_on)
        return result


@dataclass(frozen=True)
class Plan:
    steps: tuple[PlanStep, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"steps": [step.as_dict() for step in self.steps]}


class PlanRejected(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


class PlanValidator:
    """Validate plan syntax, dependencies and end-to-end reachability.

    Rules are configurable; the revenue-demo rules are passed by the caller.
    The planner may propose a different plan, but cannot bypass tool dependencies.
    """

    def __init__(
        self,
        dependencies: dict[str, tuple[str, ...]],
        final_tool: str,
        *,
        alternative_final_tools: tuple[str, ...] = (),
    ):
        self.dependencies = dependencies
        self.final_tool = final_tool
        self.terminal_tools = (final_tool, *alternative_final_tools)
        if any(tool not in dependencies for tool in self.terminal_tools):
            raise ValueError("Terminal tool is not registered")

    def parse(self, content: str) -> Plan:
        if not isinstance(content, str) or not content.strip():
            raise PlanRejected("INVALID_PLAN_JSON", "Model returned an empty plan")
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", text).strip()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanRejected("INVALID_PLAN_JSON", "Expected one JSON object") from exc
        if (
            not isinstance(raw, dict)
            or set(raw) != {"steps"}
            or not isinstance(raw["steps"], list)
        ):
            raise PlanRejected("INVALID_PLAN_SHAPE", 'Expected {"steps": [...]} object')

        if not 1 <= len(raw["steps"]) <= 6:
            raise PlanRejected("INVALID_PLAN_LENGTH", "Plan must contain 1–6 steps")
        steps = []
        for data in raw["steps"]:
            if not isinstance(data, dict) or set(data) != {
                "id",
                "tool",
                "objective",
                "depends_on",
            }:
                raise PlanRejected(
                    "INVALID_STEP_SHAPE",
                    "Each step requires id, tool, objective and depends_on",
                )
            step_id, tool, objective, deps = (
                data["id"],
                data["tool"],
                data["objective"],
                data["depends_on"],
            )
            if (
                not isinstance(step_id, str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,39}", step_id) is None
                or not isinstance(tool, str)
                or not isinstance(objective, str)
                or not objective.strip()
                or len(objective) > 200
                or not isinstance(deps, list)
                or any(not isinstance(d, str) for d in deps)
            ):
                raise PlanRejected(
                    "INVALID_STEP_FIELDS", "Invalid id, tool, objective or dependencies"
                )
            steps.append(PlanStep(step_id, tool, objective.strip(), tuple(deps)))
        return Plan(tuple(steps))

    def validate(self, plan: Plan, evidence: dict[str, dict[str, Any]]) -> None:
        seen_ids: set[str] = set()
        seen_tools: set[str] = set()
        available: set[str] = set(evidence)
        for step in plan.steps:
            if step.id in seen_ids:
                raise PlanRejected("DUPLICATE_STEP_ID", "Step IDs must be unique")
            seen_ids.add(step.id)
            if step.tool not in self.dependencies:
                raise PlanRejected("UNKNOWN_TOOL", f"Unsupported tool: {step.tool}")
            if step.tool in seen_tools:
                raise PlanRejected("DUPLICATE_TOOL", "A tool may appear only once in each plan revision")
            if step.tool in evidence:
                raise PlanRejected("REDUNDANT_STEP", f"Evidence already exists for {step.tool}")
            seen_tools.add(step.tool)

            required = self.dependencies[step.tool]
            if tuple(step.depends_on) != tuple(required):
                raise PlanRejected(
                    "INVALID_DEPENDENCIES",
                    f"{step.tool} requires depends_on={list(required)}",
                )
            missing = [name for name in required if name not in available]
            if missing:
                raise PlanRejected(
                    "UNREACHABLE_STEP",
                    f"Missing prerequisites for {step.tool}: {missing}",
                )
            available.add(step.tool)

        if not any(tool in available for tool in self.terminal_tools):
            raise PlanRejected(
                "INCOMPLETE_PLAN",
                f"Plan must reach a terminal tool: {list(self.terminal_tools)}",
            )