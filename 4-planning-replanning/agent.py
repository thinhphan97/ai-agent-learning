"""Lab 4: plan -> validate -> act -> observe -> replan -> verify.

Standalone teaching project. Both models share the workspace-root .env and
requirements.txt; prior lessons are not imported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from planning import Plan, PlanRejected, PlanStep, PlanValidator
from evaluation import ProgressEvaluator

RULES = {
    "get_customer": (),
    "get_orders": ("get_customer",),
    "sum_amounts": ("get_orders",),
    "count_orders": ("get_orders",),
}
DEFAULT_QUESTION = "Calculate paid order revenue for customer_id=1."


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Action:
    tool_calls: tuple[ToolCall, ...] = ()
    content: str | None = None


class PlanningModel(Protocol):
    def propose_plan(self, context: dict[str, Any]) -> str: ...
    def propose_action(
        self, context: dict[str, Any], tools: list[dict[str, Any]]
    ) -> Action: ...


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool], *, fail_orders_once: bool = False):
        if len(set(t.name for t in tools)) != len(tools):
            raise ValueError("Duplicate tool names")
        self.tools = {tool.name: tool for tool in tools}
        self.fail_orders_once = fail_orders_once
        self._orders_attempts = 0

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def execute(
        self,
        call: ToolCall,
        evidence: dict[str, dict[str, Any]],
        target_customer_id: int,
    ) -> dict[str, Any]:
        if call.name not in self.tools:
            return {"ok": False, "code": "UNKNOWN_TOOL"}
        try:
            arguments = json.loads(call.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Arguments must be a JSON object")
        except (ValueError, TypeError):
            return {"ok": False, "code": "INVALID_ARGUMENTS"}

        if call.name == "get_customer":
            if (
                set(arguments) != {"customer_id"}
                or type(arguments["customer_id"]) is not int
                or arguments["customer_id"] < 1
            ):
                return {"ok": False, "code": "INVALID_ARGUMENTS"}
            if arguments["customer_id"] != target_customer_id:
                return {"ok": False, "code": "DATA_MISMATCH"}
        if call.name == "get_orders":
            customer = evidence.get("get_customer", {}).get("customer")
            if customer is None:
                return {"ok": False, "code": "PREREQUISITE_MISSING"}
            if arguments.get("customer") != customer["name"]:
                return {"ok": False, "code": "DATA_MISMATCH"}
        if call.name in {"sum_amounts", "count_orders"}:
            orders = evidence.get("get_orders", {}).get("orders")
            if orders is None:
                return {"ok": False, "code": "PREREQUISITE_MISSING"}
            if call.name == "sum_amounts" and arguments.get("amounts") != [
                order["amount"] for order in orders
            ]:
                return {"ok": False, "code": "DATA_MISMATCH"}
            if call.name == "count_orders" and arguments.get("order_ids") != [
                order["id"] for order in orders
            ]:
                return {"ok": False, "code": "DATA_MISMATCH"}

        try:
            if call.name == "get_orders":
                self._orders_attempts += 1
                if self.fail_orders_once and self._orders_attempts == 1:
                    raise RuntimeError("Simulated transient tool failure")
            data = self.tools[call.name].handler(arguments)
        except (ValueError, TypeError, KeyError):
            return {"ok": False, "code": "INVALID_ARGUMENTS"}
        except Exception:
            # Never propagate internal exception messages/secrets into LLM context.
            return {"ok": False, "code": "TOOL_EXECUTION_FAILED"}
        return {"ok": True, "data": data}


@dataclass(frozen=True)
class Limits:
    max_model_calls: int = 14
    max_plan_attempts: int = 3
    max_tool_attempts: int = 8
    max_errors: int = 3
    max_tool_retries: int = 1

    def __post_init__(self) -> None:
        if (
            min(
                self.max_model_calls,
                self.max_plan_attempts,
                self.max_tool_attempts,
                self.max_errors,
            )
            < 1
        ):
            raise ValueError("All limits must be positive")
        if self.max_tool_retries < 0:
            raise ValueError("max_tool_retries cannot be negative")


@dataclass
class RunState:
    request: str
    status: str = "created"
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    plan_history: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    plan_attempts: int = 0
    tool_attempts: int = 0
    errors: int = 0
    answer: str = ""
    reason: str = ""

    def log(self, event: str, **fields: Any) -> None:
        self.events.append({"seq": len(self.events) + 1, "event": event, **fields})

    @property
    def replans(self) -> int:
        return max(0, self.plan_attempts - 1)


class AgentRunner:
    def __init__(
        self,
        model: PlanningModel,
        registry: ToolRegistry,
        *,
        limits: Limits | None = None,
    ):
        self.model = model
        self.registry = registry
        self.limits = limits or Limits()
        self.validator = PlanValidator(
            RULES, final_tool="sum_amounts",
            alternative_final_tools=("count_orders",),
        )
        self.evaluator = ProgressEvaluator()

    def stop(self, state: RunState, status: str, reason: str) -> RunState:
        state.status, state.reason = status, reason
        state.log("run_stopped", status=status, reason=reason)
        return state

    def completed(self, state: RunState) -> bool:
        result = self.evaluator.evaluate(state.evidence)
        if not result.satisfied:
            return False
        customer = state.evidence["get_customer"]["customer"]
        if customer is None:
            state.answer = "Customer was not found."
        else:
            state.answer = (
                f"Paid revenue for {customer['name']}: "
                f"{state.evidence['sum_amounts']['total']:,} VND"
            )
        return True

    def run(self, question: str) -> RunState:
        state = RunState(request=question)
        match = re.search(r"\bcustomer_id\s*=\s*(\d+)\b", question)
        if not match:
            return self.stop(state, "failed", "MISSING_CUSTOMER_ID")
        customer_id = int(match.group(1))
        if customer_id < 1:
            return self.stop(state, "failed", "INVALID_CUSTOMER_ID")
        state.log("run_started", customer_id=customer_id)
        last_error: dict[str, str] | None = None
        accepted_plan: Plan | None = None
        used_call_ids: set[str] = set()

        while True:
            if self.completed(state):
                return self.stop(state, "completed", "EVIDENCE_VERIFIED")

            if accepted_plan is None:
                if state.plan_attempts >= self.limits.max_plan_attempts:
                    return self.stop(state, "replan_exhausted", "PLAN_ATTEMPT_LIMIT")
                if state.model_calls >= self.limits.max_model_calls:
                    return self.stop(state, "budget_exceeded", "MODEL_CALL_LIMIT")

                state.status = "replanning" if state.plan_attempts else "planning"
                context = {
                    "request": question,
                    "target_customer_id": customer_id,
                    "successful_evidence": state.evidence,
                    "last_error": last_error,
                    "tool_dependencies": {k: list(v) for k, v in RULES.items()},
                    "goal_metric": "paid_revenue_amount_vnd",
                    "available_terminal_tools": ["sum_amounts", "count_orders"],
                    "required_final_tool": "sum_amounts",
                    "plan_attempt": state.plan_attempts + 1,
                }
                state.plan_attempts += 1
                state.model_calls += 1
                state.log(
                    "plan_requested", attempt=state.plan_attempts, last_error=last_error
                )
                try:
                    content = self.model.propose_plan(context)
                except Exception:
                    return self.stop(state, "failed", "PLANNER_PROVIDER_ERROR")
                try:
                    plan = self.validator.parse(content)
                    self.validator.validate(plan, state.evidence)
                except PlanRejected as exc:
                    state.errors += 1
                    last_error = {
                        "code": exc.code,
                        "detail": exc.detail,
                        "phase": "planning",
                    }
                    state.log("plan_rejected", code=exc.code, detail=exc.detail)
                    if state.errors >= self.limits.max_errors:
                        return self.stop(state, "recovery_exhausted", "ERROR_LIMIT")
                    continue
                accepted_plan = plan
                state.plan_history.append(plan.as_dict())
                state.log(
                    "plan_accepted",
                    revision=len(state.plan_history),
                    steps=[
                        {"id": s.id, "tool": s.tool, "depends_on": list(s.depends_on)}
                        for s in plan.steps
                    ],
                )
                last_error = None
                state.status = "executing"

            failure: dict[str, str] | None = None
            for step in accepted_plan.steps:
                if self.completed(state):
                    break
                if state.model_calls >= self.limits.max_model_calls:
                    return self.stop(state, "budget_exceeded", "MODEL_CALL_LIMIT")
                if state.tool_attempts >= self.limits.max_tool_attempts:
                    return self.stop(state, "budget_exceeded", "TOOL_ATTEMPT_LIMIT")
                state.model_calls += 1
                context = {
                    "request": question,
                    "target_customer_id": customer_id,
                    "plan_revision": len(state.plan_history),
                    "current_step": step.as_dict(),
                    "successful_evidence": state.evidence,
                }
                state.log("action_requested", step=step.id, expected_tool=step.tool)
                try:
                    action = self.model.propose_action(context, self.registry.schemas())
                except Exception:
                    return self.stop(state, "failed", "EXECUTOR_PROVIDER_ERROR")

                state.tool_attempts += 1
                if len(action.tool_calls) != 1:
                    failure = {"code": "ACTION_PROTOCOL_ERROR", "phase": "execution"}
                else:
                    call = action.tool_calls[0]
                    if not call.id or call.id in used_call_ids:
                        failure = {
                            "code": "DUPLICATE_TOOL_CALL_ID",
                            "phase": "execution",
                        }
                    elif call.name != step.tool:
                        failure = {"code": "ACTION_TOOL_MISMATCH", "phase": "execution"}
                    else:
                        used_call_ids.add(call.id)
                        observed = self.registry.execute(
                            call, state.evidence, customer_id
                        )
                        state.log(
                            "tool_observation", step=step.id, tool=call.name,
                            call_id=call.id, observation=observed,
                        )
                        retry_count = 0
                        while (
                            not observed["ok"]
                            and observed["code"] == "TOOL_EXECUTION_FAILED"
                            and retry_count < self.limits.max_tool_retries
                        ):
                            state.errors += 1
                            state.log(
                                "tool_retry", step=step.id, tool=call.name,
                                reason=observed["code"], retry=retry_count + 1,
                            )
                            if state.errors >= self.limits.max_errors:
                                return self.stop(state, "recovery_exhausted", "ERROR_LIMIT")
                            if state.tool_attempts >= self.limits.max_tool_attempts:
                                return self.stop(state, "budget_exceeded", "TOOL_ATTEMPT_LIMIT")
                            retry_count += 1
                            state.tool_attempts += 1
                            observed = self.registry.execute(
                                call, state.evidence, customer_id
                            )
                            state.log(
                                "tool_observation", step=step.id, tool=call.name,
                                call_id=call.id, observation=observed, retry=retry_count,
                            )
                        if observed["ok"]:
                            state.evidence[step.tool] = observed["data"]
                            state.log("step_completed", step=step.id, tool=step.tool)
                        else:
                            failure = {"code": observed["code"], "phase": "execution"}

                if failure:
                    state.errors += 1
                    state.log(
                        "step_failed",
                        step=step.id,
                        tool=step.tool,
                        code=failure["code"],
                    )
                    if state.errors >= self.limits.max_errors:
                        return self.stop(state, "recovery_exhausted", "ERROR_LIMIT")
                    if failure["code"] == "TOOL_EXECUTION_FAILED":
                        return self.stop(state, "failed", "TOOL_EXECUTION_FAILED")
                    break

            evaluation = self.evaluator.evaluate(state.evidence)
            state.log(
                "progress_evaluated", code=evaluation.code,
                satisfied=evaluation.satisfied, detail=evaluation.detail,
            )
            if self.completed(state):
                return self.stop(state, "completed", "EVIDENCE_VERIFIED")
            if evaluation.code == "EVIDENCE_INCONSISTENT":
                return self.stop(state, "failed", evaluation.code)
            if failure is None:
                if evaluation.code != "GOAL_METRIC_MISMATCH":
                    return self.stop(state, "failed", "PLAN_DID_NOT_REACH_GOAL")
                last_error = {
                    "code": evaluation.code,
                    "detail": evaluation.detail,
                    "phase": "goal_evaluation",
                }
                state.log("goal_mismatch", code=evaluation.code, detail=evaluation.detail)
            else:
                last_error = failure

            accepted_plan = None
            state.log(
                "replan_triggered",
                last_error=last_error,
                successful_tools=list(state.evidence),
            )


CUSTOMERS = {1: {"id": 1, "name": "Lan"}, 2: {"id": 2, "name": "Minh"}}
ORDERS = [
    {"id": 101, "customer": "Lan", "status": "paid", "amount": 120000},
    {"id": 102, "customer": "Lan", "status": "paid", "amount": 80000},
    {"id": 103, "customer": "Lan", "status": "pending", "amount": 50000},
    {"id": 104, "customer": "Minh", "status": "paid", "amount": 45000},
]


def get_customer(args: dict[str, Any]) -> dict[str, Any]:
    if (
        set(args) != {"customer_id"}
        or type(args["customer_id"]) is not int
        or args["customer_id"] < 1
    ):
        raise ValueError("Expected positive customer_id integer")
    customer = CUSTOMERS.get(args["customer_id"])
    return {"found": customer is not None, "customer": customer}


def get_orders(args: dict[str, Any]) -> dict[str, Any]:
    if (
        set(args) != {"customer", "status"}
        or not isinstance(args["customer"], str)
        or args["status"] != "paid"
    ):
        raise ValueError("Expected customer and status='paid'")
    return {
        "orders": [
            order
            for order in ORDERS
            if order["customer"] == args["customer"] and order["status"] == "paid"
        ]
    }


def sum_amounts(args: dict[str, Any]) -> dict[str, Any]:
    values = args.get("amounts")
    if (
        set(args) != {"amounts"}
        or not isinstance(values, list)
        or any(type(n) is not int or n < 0 for n in values)
    ):
        raise ValueError("Expected nonnegative integer list")
    return {"total": sum(values)}


def count_orders(args: dict[str, Any]) -> dict[str, Any]:
    """Count orders. Successful execution does NOT prove revenue."""
    ids = args.get("order_ids")
    if (
        set(args) != {"order_ids"}
        or not isinstance(ids, list)
        or any(type(value) is not int or value < 1 for value in ids)
    ):
        raise ValueError("Expected a list of positive order IDs")
    return {"count": len(ids)}


def build_registry(*, fail_orders_once: bool = False) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                "get_customer",
                "Find customer by numeric ID",
                {
                    "type": "object",
                    "properties": {"customer_id": {"type": "integer", "minimum": 1}},
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
                get_customer,
            ),
            Tool(
                "get_orders",
                "Get PAID orders for customer name returned by get_customer",
                {
                    "type": "object",
                    "properties": {
                        "customer": {"type": "string"},
                        "status": {"type": "string", "enum": ["paid"]},
                    },
                    "required": ["customer", "status"],
                    "additionalProperties": False,
                },
                get_orders,
            ),
            Tool(
                "count_orders",
                "Count the number of previously returned paid orders (NOT revenue)",
                {
                    "type": "object",
                    "properties": {
                        "order_ids": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                        }
                    },
                    "required": ["order_ids"],
                    "additionalProperties": False,
                },
                count_orders,
            ),
            Tool(
                "sum_amounts",
                "Sum amounts from previously returned paid orders",
                {
                    "type": "object",
                    "properties": {
                        "amounts": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        }
                    },
                    "required": ["amounts"],
                    "additionalProperties": False,
                },
                sum_amounts,
            ),
        ],
        fail_orders_once=fail_orders_once,
    )


class MockModel:
    """Deterministic plan/action provider, useful for unit tests without API calls."""

    def __init__(self, scenario: str = "normal"):
        if scenario not in {"normal", "invalid-plan", "wrong-action", "tool-failure", "semantic-mismatch"}:
            raise ValueError("Unknown mock scenario")
        self.scenario = scenario
        self.plan_calls = 0
        self.action_calls = 0

    def propose_plan(self, context: dict[str, Any]) -> str:
        self.plan_calls += 1
        if self.scenario == "invalid-plan" and self.plan_calls == 1:
            return json.dumps(
                {
                    "steps": [
                        {
                            "id": "bad",
                            "tool": "get_orders",
                            "objective": "Query before lookup",
                            "depends_on": ["get_customer"],
                        }
                    ]
                }
            )
        if self.scenario == "semantic-mismatch" and self.plan_calls == 1:
            chosen = ("get_customer", "get_orders", "count_orders")
        else:
            chosen = tuple(
                tool for tool in ("get_customer", "get_orders", "sum_amounts")
                if tool not in context["successful_evidence"]
            )
        tools = chosen
        return json.dumps(
            {
                "steps": [
                    {
                        "id": f"step_{i}",
                        "tool": tool,
                        "objective": f"Execute {tool} with grounded inputs",
                        "depends_on": list(RULES[tool]),
                    }
                    for i, tool in enumerate(tools, 1)
                ]
            }
        )

    def propose_action(
        self, context: dict[str, Any], tools: list[dict[str, Any]]
    ) -> Action:
        self.action_calls += 1
        target = context["current_step"]["tool"]
        if self.scenario == "wrong-action" and self.action_calls == 1:
            target = "get_orders"
        evidence = context["successful_evidence"]
        if target == "get_customer":
            args = {"customer_id": context["target_customer_id"]}
        elif target == "get_orders":
            customer = evidence.get("get_customer", {}).get("customer")
            args = {
                "customer": customer["name"] if customer else "Lan",
                "status": "paid",
            }
        elif target == "count_orders":
            orders = evidence.get("get_orders", {}).get("orders", [])
            args = {"order_ids": [o["id"] for o in orders]}
        else:
            orders = evidence.get("get_orders", {}).get("orders", [])
            args = {"amounts": [o["amount"] for o in orders]}
        return Action(
            tool_calls=(
                ToolCall(f"mock_call_{self.action_calls}", target, json.dumps(args)),
            )
        )


class PlanFaultInjector:
    """Replace the first real planner response with a valid but wrong-metric plan.

    The first provider call is made; only its response is discarded.
    Every executor decision and every later plan comes from the real provider.
    """

    def __init__(self, inner: PlanningModel):
        self.inner = inner
        self.injected = False

    def propose_plan(self, context: dict[str, Any]) -> str:
        actual = self.inner.propose_plan(context)
        if self.injected:
            return actual
        self.injected = True
        faulty_tools = ("get_customer", "get_orders", "count_orders")
        plan = {
            "steps": [
                {
                    "id": f"step_{i}", "tool": tool,
                    "objective": f"Use {tool} to answer the user request",
                    "depends_on": list(RULES[tool]),
                }
                for i, tool in enumerate(faulty_tools, 1)
            ]
        }
        print("Injected first LLM plan: wrong-metric (original discarded).")
        return json.dumps(plan)

    def propose_action(
        self, context: dict[str, Any], tools: list[dict[str, Any]]
    ) -> Action:
        return self.inner.propose_action(context, tools)


class ResponseFaultInjector:
    """Replace ONE real model action but delegate planning and all subsequent actions."""

    def __init__(self, inner: PlanningModel, fault: str):
        if fault not in {"wrong-tool", "invalid-arguments"}:
            raise ValueError("Unsupported fault")
        self.inner, self.fault, self.injected = inner, fault, False

    def propose_plan(self, context: dict[str, Any]) -> str:
        return self.inner.propose_plan(context)

    def propose_action(
        self, context: dict[str, Any], tools: list[dict[str, Any]]
    ) -> Action:
        original = self.inner.propose_action(context, tools)
        if self.injected:
            return original
        self.injected = True
        if self.fault == "wrong-tool":
            forged = ToolCall(
                "injected_action_1", "get_orders", '{"customer":"Lan","status":"paid"}'
            )
        else:
            forged = ToolCall(
                "injected_action_1",
                context["current_step"]["tool"],
                '{"customer_id":"1"}',
            )
        print(f"Injected first LLM action: {self.fault} (original discarded).")
        return Action(tool_calls=(forged,))


def openai_compatible_model() -> PlanningModel:
    """OpenAI SDK adapter for OpenAI and OpenAI-compatible vLLM endpoints."""
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install workspace-root requirements.txt before live mode"
        ) from exc
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        raise ValueError("Set LLM_MODEL in the workspace-root .env")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY") or "EMPTY",
        base_url=os.getenv("LLM_BASE_URL") or None,
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
    )

    class OpenAICompatibleModel:
        def propose_plan(self, context: dict[str, Any]) -> str:
            result = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the PLANNER. Return ONE JSON object, no explanation. "
                            "Every step MUST have exactly four fields: "
                            "id (STRING, e.g. 'resolve_customer', NEVER a number), "
                            "tool (STRING), objective (nonempty STRING), "
                            "depends_on (ARRAY of tool-name STRINGS). "
                            "Available tools: get_customer, get_orders, "
                            "sum_amounts (SUM monetary amounts), count_orders "
                            "(COUNT orders; never revenue). "
                            "The full initial plan example is: "
                            "{\"steps\":[{\"id\":\"lookup\",\"tool\":\"get_customer\","
                            "\"objective\":\"Resolve customer ID\",\"depends_on\":[]},"
                            "{\"id\":\"orders\",\"tool\":\"get_orders\","
                            "\"objective\":\"Read paid orders\",\"depends_on\":[\"get_customer\"]},"
                            "{\"id\":\"total\",\"tool\":\"sum_amounts\","
                            "\"objective\":\"Sum amounts\",\"depends_on\":[\"get_orders\"]}]}. "
                            "Use the exact tool_dependencies in the request; prerequisites "
                            "must appear earlier in the plan or already exist as evidence. "
                            "For a paid REVENUE request, the goal is a monetary total: "
                            "sum_amounts, not count_orders. A count alone does not meet "
                            "the user goal. Return only remaining steps to reach the goal; "
                            "never rerun a tool with successful evidence. "
                            "If last_error.code is GOAL_METRIC_MISMATCH, replace the "
                            "wrong metric with sum_amounts based on saved get_orders. "
                            "On other plan or execution errors, revise remaining steps "
                            "using last_error. No tool arguments in the plan."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(context, ensure_ascii=False),
                    },
                ],
            )
            return result.choices[0].message.content or ""

        def propose_action(
            self, context: dict[str, Any], tools: list[dict[str, Any]]
        ) -> Action:
            result = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the EXECUTOR. Call exactly ONE tool for the given "
                            "current_step.tool. Use request + successful_evidence to compute "
                            "arguments. Do not guess customer names or order amounts. "
                            "Do not return a final text response."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(context, ensure_ascii=False),
                    },
                ],
                tools=tools,
                tool_choice="auto",
            )
            msg = result.choices[0].message
            return Action(
                tool_calls=tuple(
                    ToolCall(t.id, t.function.name, t.function.arguments)
                    for t in (msg.tool_calls or ())
                ),
                content=msg.content,
            )

    return OpenAICompatibleModel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 4: Planning and Replanning")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="Force offline mode")
    mode.add_argument("--live", action="store_true", help="Use real LLM from root .env")
    parser.add_argument(
        "--scenario",
        choices=["normal", "invalid-plan", "wrong-action", "tool-failure", "semantic-mismatch"],
        help="Force offline mock scenario, even when LIVE_MODE=1 is exported",
    )
    parser.add_argument(
        "--inject-llm-action",
        choices=["wrong-tool", "invalid-arguments"],
        help="Live: replace only the first real executor response",
    )
    parser.add_argument(
        "--inject-tool-failure",
        action="store_true",
        help="Live: make the first get_orders execution fail",
    )
    parser.add_argument(
        "--inject-plan-fault",
        action="store_true",
        help="Live: replace the first real planner response with a structurally valid wrong-goal plan",
    )
    args = parser.parse_args()
    if args.scenario is not None and args.live:
        parser.error("--scenario and --live are mutually exclusive")
    live = args.live or (
        not args.mock and args.scenario is None and os.getenv("LIVE_MODE") == "1"
    )
    if (args.inject_llm_action or args.inject_tool_failure or args.inject_plan_fault) and not live:
        parser.error("Fault injection flags require live mode")
    if sum(bool(x) for x in (
        args.inject_llm_action, args.inject_tool_failure, args.inject_plan_fault
    )) > 1:
        parser.error("Choose one fault injection type at a time")

    scenario = args.scenario or "normal"
    model = openai_compatible_model() if live else MockModel(scenario)
    if args.inject_llm_action:
        model = ResponseFaultInjector(model, args.inject_llm_action)
    if args.inject_plan_fault:
        model = PlanFaultInjector(model)
    registry = build_registry(
        fail_orders_once=(
            args.inject_tool_failure if live else scenario == "tool-failure"
        )
    )
    print(
        f"Mode: {'live' if live else 'mock'}"
        + (f" | scenario={scenario}" if not live else "")
    )
    state = AgentRunner(model, registry).run(args.question)
    for event in state.events:
        if event["event"] == "plan_accepted":
            print(
                "Plan revision",
                event["revision"],
                ":",
                " -> ".join(s["tool"] for s in event["steps"]),
            )
        elif event["event"] in {"plan_rejected", "step_failed"}:
            print(event["event"], ":", event["code"])
        elif event["event"] == "step_completed":
            print("Step completed:", event["tool"])
        elif event["event"] == "goal_mismatch":
            print("Goal mismatch:", event["code"], "-", event["detail"])
        elif event["event"] == "tool_retry":
            print("Tool retry:", event["tool"], "-", event["reason"])
        elif event["event"] == "replan_triggered":
            print("Replan triggered:", event["last_error"]["code"])
    print(f"Status: {state.status}")
    print(
        f"Model calls: {state.model_calls}; plan attempts: {state.plan_attempts}; "
        f"tool attempts: {state.tool_attempts}; errors: {state.errors}"
    )
    if state.reason and state.status != "completed":
        print("Reason:", state.reason)
    if state.answer:
        print("Answer:", state.answer)


if __name__ == "__main__":
    main()