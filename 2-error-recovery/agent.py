"""Lab 2: feedback-driven tool error recovery in an independent ReAct runtime.

Run offline: python agent.py --scenario wrong-order
Run live:    LIVE_MODE=1 python agent.py --question "Calculate paid order revenue for customer_id=1."
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelTurn:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class Model(Protocol):
    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn: ...


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


def error(tool: str, code: str, detail: str) -> dict[str, Any]:
    return {"ok": False, "tool": tool, "error": {"code": code, "message": detail}}


class DependencyPolicy:
    """Gate capabilities on *successful observations*, not fixed step numbers."""

    def check(
        self, name: str, args: dict[str, Any], successful: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        if name == "get_orders":
            resolved = successful.get("get_customer")
            if resolved is None:
                return error(
                    name,
                    "PREREQUISITE_MISSING",
                    "Call get_customer successfully before get_orders.",
                )
            customer = resolved.get("customer")
            if customer is None:
                return error(
                    name,
                    "CUSTOMER_NOT_FOUND",
                    "Cannot query orders for an unknown customer.",
                )
            if args.get("customer") != customer["name"]:
                return error(
                    name,
                    "DATA_MISMATCH",
                    "Use the customer name from get_customer observation.",
                )
        if name == "sum_amounts":
            orders = successful.get("get_orders")
            if orders is None:
                return error(
                    name,
                    "PREREQUISITE_MISSING",
                    "Call get_orders successfully before sum_amounts.",
                )
            expected = [order["amount"] for order in orders["orders"]]
            if args.get("amounts") != expected:
                return error(
                    name,
                    "DATA_MISMATCH",
                    "Use the amounts from get_orders observation.",
                )
        return None

    def can_finish(self, successful: dict[str, dict[str, Any]]) -> bool:
        customer = successful.get("get_customer")
        if customer is not None and customer["customer"] is None:
            return True  # No customer exists; a factual not-found response is valid.
        return "sum_amounts" in successful


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Duplicate tool name")
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def execute(
        self,
        call: ToolCall,
        policy: DependencyPolicy,
        successful: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        tool = self.tools.get(call.name)
        if tool is None:
            return error(
                call.name, "UNKNOWN_TOOL", "Use a tool advertised in the registry."
            )
        try:
            args = json.loads(call.arguments)
            if not isinstance(args, dict):
                raise ValueError("Arguments must be a JSON object")
        except (ValueError, TypeError):
            return error(
                call.name,
                "INVALID_ARGUMENTS",
                "Provide a valid JSON object for arguments.",
            )

        rejected = policy.check(call.name, args, successful)
        if rejected is not None:
            return rejected

        try:
            data = tool.handler(args)
        except (ValueError, TypeError, KeyError):
            return error(
                call.name,
                "INVALID_ARGUMENTS",
                "Arguments do not match this tool's contract.",
            )
        except Exception:
            # Do not leak arbitrary exception details, credentials or stack traces to the model.
            return error(
                call.name,
                "TOOL_EXECUTION_FAILED",
                "Tool failed. You may retry or choose another action.",
            )
        return {"ok": True, "tool": call.name, "data": data}


@dataclass(frozen=True)
class RecoveryPolicy:
    max_model_calls: int = 10
    max_tool_attempts: int = 10
    max_errors: int = 3

    def __post_init__(self) -> None:
        if min(self.max_model_calls, self.max_tool_attempts, self.max_errors) < 1:
            raise ValueError("All budgets must be positive")


@dataclass
class AgentResult:
    status: str
    answer: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    tool_attempts: int = 0
    errors: int = 0


class AgentRunner:
    def __init__(
        self,
        model: Model,
        registry: ToolRegistry,
        *,
        policy: DependencyPolicy | None = None,
        recovery: RecoveryPolicy | None = None,
    ):
        self.model, self.registry = model, registry
        self.policy = policy or DependencyPolicy()
        self.recovery = recovery or RecoveryPolicy()

    def run(self, request: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a revenue assistant. Use tools rather than guessing. "
                    "Resolve the customer with get_customer before get_orders; derive amounts "
                    "from get_orders before sum_amounts. Use one tool call per model turn. "
                    "If a tool result has ok=false, correct the action using its error code. "
                    "Only finish when you have computed the total or the customer does not exist."
                ),
            },
            {"role": "user", "content": request},
        ]
        successful: dict[str, dict[str, Any]] = {}
        result = AgentResult(status="budget_exceeded")
        used_call_ids: set[str] = set()

        for iteration in range(1, self.recovery.max_model_calls + 1):
            try:
                turn = self.model.respond(messages, self.registry.schemas())
            except Exception:
                result.status = "provider_error"
                result.model_calls = iteration
                return result
            result.model_calls = iteration

            if turn.tool_calls:
                calls = turn.tool_calls
                if result.tool_attempts + len(calls) > self.recovery.max_tool_attempts:
                    result.status = "budget_exceeded"
                    return result
                ids = [c.id for c in calls]
                if any(not i or i in used_call_ids for i in ids) or len(
                    set(ids)
                ) != len(ids):
                    result.status = "protocol_error"
                    return result
                used_call_ids.update(ids)
                messages.append(
                    {
                        "role": "assistant",
                        "content": turn.content,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments},
                            }
                            for c in calls
                        ],
                    }
                )
                # A rejected multi-tool batch still receives one response per tool_call_id.
                # No tool in a batch executes; subsequent LLM turns may recover.
                for call in calls:
                    result.tool_attempts += 1
                    observation = (
                        error(
                            call.name,
                            "MULTIPLE_TOOL_CALLS_UNSUPPORTED",
                            "Call one tool at a time.",
                        )
                        if len(calls) != 1
                        else self.registry.execute(call, self.policy, successful)
                    )
                    result.trace.append(
                        {
                            "event": "tool",
                            "iteration": iteration,
                            "action": call.name,
                            "tool_call_id": call.id,
                            "observation": observation,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(observation, ensure_ascii=False),
                        }
                    )
                    if observation["ok"]:
                        successful[call.name] = observation["data"]
                    else:
                        result.errors += 1

            elif turn.content:
                if self.policy.can_finish(successful):
                    result.status = "completed"
                    result.answer = turn.content
                    return result
                result.trace.append(
                    {
                        "event": "final_rejected",
                        "iteration": iteration,
                        "reason": "FINAL_BEFORE_COMPLETION",
                    }
                )
                result.errors += 1
                # A plain-text final has no tool_call_id; send internal feedback as a
                # new user message (teaching shortcut; production needs stronger roles).
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Runtime feedback: FINAL_BEFORE_COMPLETION. "
                            "You must obtain the required tool observations before finishing."
                        ),
                    }
                )
            else:
                # Some OpenAI-compatible models return an empty turn after tools.
                # Treat it as recoverable feedback, not an immediate silent failure.
                result.trace.append(
                    {
                        "event": "empty_turn",
                        "iteration": iteration,
                        "reason": "EMPTY_MODEL_TURN",
                    }
                )
                result.errors += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Runtime feedback: EMPTY_MODEL_TURN. Your last response contained "
                            "neither a tool call nor an answer. Please make one tool call or "
                            "provide a short, non-empty answer grounded in tool observations."
                        ),
                    }
                )

            if result.errors >= self.recovery.max_errors:
                result.status = "recovery_exhausted"
                return result

        return result


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
        raise ValueError("Expected a positive integer customer_id")
    customer = CUSTOMERS.get(args["customer_id"])
    return {"found": customer is not None, "customer": customer}


def get_orders(args: dict[str, Any]) -> dict[str, Any]:
    if (
        set(args) != {"customer", "status"}
        or not isinstance(args["customer"], str)
        or args["status"] != "paid"
    ):
        raise ValueError("Expected customer (string), status='paid'")
    return {
        "orders": [
            o
            for o in ORDERS
            if o["customer"] == args["customer"] and o["status"] == "paid"
        ]
    }


def sum_amounts(args: dict[str, Any]) -> dict[str, Any]:
    amounts = args.get("amounts")
    if (
        set(args) != {"amounts"}
        or not isinstance(amounts, list)
        or any(type(amount) is not int or amount < 0 for amount in amounts)
    ):
        raise ValueError("Expected list of nonnegative integer amounts")
    return {"total": sum(amounts)}


def build_registry(*, fail_orders_once: bool = False) -> ToolRegistry:
    attempts = 0

    def maybe_fail_get_orders(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if fail_orders_once and attempts == 1:
            raise RuntimeError("Simulated transient database failure")
        return get_orders(args)

    return ToolRegistry(
        [
            Tool(
                "get_customer",
                "Find a customer by numeric ID.",
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
                "Fetch PAID orders for the customer name returned by get_customer.",
                {
                    "type": "object",
                    "properties": {
                        "customer": {"type": "string"},
                        "status": {"type": "string", "enum": ["paid"]},
                    },
                    "required": ["customer", "status"],
                    "additionalProperties": False,
                },
                maybe_fail_get_orders,
            ),
            Tool(
                "sum_amounts",
                "Sum exact amounts obtained from paid orders.",
                {
                    "type": "object",
                    "properties": {
                        "amounts": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                    },
                    "required": ["amounts"],
                    "additionalProperties": False,
                },
                sum_amounts,
            ),
        ]
    )


class MockModel:
    """A history-driven simulator for reproducible teaching; NOT an LLM."""

    def __init__(self, scenario: str = "normal"):
        if scenario not in {
            "normal",
            "wrong-order",
            "invalid-arguments",
            "early-final",
            "tool-failure",
        }:
            raise ValueError("Unknown mock scenario")
        self.scenario = scenario
        self.calls = 0

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.calls += 1
        question = next(m["content"] for m in messages if m["role"] == "user")
        match = re.search(r"customer_id\s*=\s*(\d+)", question)
        if not match:
            return ModelTurn(content="The request needs a customer_id.")
        customer_id = int(match.group(1))
        observations = [
            json.loads(message["content"])
            for message in messages
            if message["role"] == "tool"
        ]
        successful = {item["tool"]: item["data"] for item in observations if item["ok"]}

        if self.calls == 1:
            if self.scenario == "wrong-order":
                return self._call("get_orders", {"customer": "Lan", "status": "paid"})
            if self.scenario == "invalid-arguments":
                return self._call("get_customer", {"customer_id": str(customer_id)})
            if self.scenario == "early-final":
                return ModelTurn(content="I think the total is 200,000 VND")

        if "get_customer" not in successful:
            return self._call("get_customer", {"customer_id": customer_id})
        customer = successful["get_customer"]["customer"]
        if customer is None:
            return ModelTurn(content=f"Customer {customer_id} was not found.")
        if "get_orders" not in successful:
            return self._call(
                "get_orders", {"customer": customer["name"], "status": "paid"}
            )
        if "sum_amounts" not in successful:
            orders = successful["get_orders"]["orders"]
            return self._call("sum_amounts", {"amounts": [o["amount"] for o in orders]})
        total = successful["sum_amounts"]["total"]
        return ModelTurn(content=f"Paid revenue for {customer['name']}: {total:,} VND")

    def _call(self, name: str, args: dict[str, Any]) -> ModelTurn:
        return ModelTurn(
            tool_calls=(
                ToolCall(
                    id=f"call_{self.calls}", name=name, arguments=json.dumps(args)
                ),
            )
        )


class ResponseFaultInjector:
    """Replace one normalized LLM response, then delegate every later turn.

    The real provider is STILL called on the injected turn. Only its returned
    ModelTurn is replaced; the tool protocol and runtime stay unchanged.
    """

    def __init__(self, model: Model, scenario: str):
        if scenario not in {"wrong-order", "invalid-arguments", "early-final"}:
            raise ValueError("Unknown LLM response injection scenario")
        self.model = model
        self.scenario = scenario
        self.injected = False

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        actual = self.model.respond(messages, tools)
        if self.injected:
            return actual

        self.injected = True
        if self.scenario == "wrong-order":
            injected = ModelTurn(tool_calls=(
                ToolCall("injected_call_1", "get_orders", json.dumps({
                    "customer": "Lan", "status": "paid"
                })),
            ))
        elif self.scenario == "invalid-arguments":
            question = next(
                message["content"] for message in messages if message["role"] == "user"
            )
            match = re.search(r"customer_id\\s*=\\s*(\\d+)", question)
            customer_id = match.group(1) if match else "1"
            injected = ModelTurn(tool_calls=(
                ToolCall("injected_call_1", "get_customer", json.dumps({
                    "customer_id": customer_id  # intentionally a string, not an integer
                })),
            ))
        else:
            injected = ModelTurn(content="The total is 200,000 VND (unverified).")

        print(
            f"Injected LLM response: {self.scenario}; "
            f"replaced first provider turn (tools={len(actual.tool_calls)})."
        )
        return injected


def openai_compatible_model() -> Model:
    """Read the shared root .env. Never print its credentials."""
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install root requirements.txt before LIVE_MODE=1") from exc

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        raise ValueError("LLM_MODEL is required in the root .env")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY") or "EMPTY",
        base_url=os.getenv("LLM_BASE_URL") or None,
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
    )

    class OpenAICompatibleModel:
        def respond(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> ModelTurn:
            response = client.chat.completions.create(
                model=model_name, messages=messages, tools=tools, tool_choice="auto"
            )
            message = response.choices[0].message
            calls = tuple(
                ToolCall(call.id, call.function.name, call.function.arguments)
                for call in (message.tool_calls or ())
            )
            return ModelTurn(content=message.content, tool_calls=calls)

    return OpenAICompatibleModel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 2: ReAct error recovery")
    parser.add_argument(
        "--question", default="Calculate paid order revenue for customer_id=1."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--scenario",
        choices=["normal", "wrong-order", "invalid-arguments", "early-final", "tool-failure"],
        default=None,
        help="Force a deterministic offline scenario, even if LIVE_MODE=1 is exported",
    )
    mode_group.add_argument(
        "--live", action="store_true",
        help="Use the real LLM configured in the shared root .env",
    )
    parser.add_argument(
        "--inject-tool-failure", action="store_true",
        help="Live mode only: fail the first get_orders call to exercise real LLM recovery",
    )
    parser.add_argument(
        "--inject-llm-response",
        choices=["wrong-order", "invalid-arguments", "early-final"],
        help="Live mode only: replace the first real LLM response; later turns remain real",
    )
    args = parser.parse_args()

    # Explicit --scenario always overrides the exported LIVE_MODE setting.
    # Without a mode flag, preserve backward compatibility with LIVE_MODE=1.
    live = args.live or (args.scenario is None and os.getenv("LIVE_MODE") == "1")
    if args.inject_tool_failure and not live:
        parser.error("--inject-tool-failure requires live mode")
    if args.inject_llm_response and not live:
        parser.error("--inject-llm-response requires live mode")
    if args.inject_tool_failure and args.inject_llm_response:
        parser.error("Choose only one injection type for a reproducible experiment")
    scenario = args.scenario or "wrong-order"

    if live:
        mode = "Mode: live"
        if args.inject_tool_failure:
            mode += " (injected get_orders failure)"
        elif args.inject_llm_response:
            mode += f" (injected LLM response: {args.inject_llm_response})"
        print(mode)
    else:
        print(f"Mode: mock (scenario: {scenario})")
    model = openai_compatible_model() if live else MockModel(scenario)
    if args.inject_llm_response:
        model = ResponseFaultInjector(model, args.inject_llm_response)
    registry = build_registry(
        fail_orders_once=args.inject_tool_failure if live else scenario == "tool-failure"
    )
    result = AgentRunner(model, registry).run(args.question)
    for step in result.trace:
        if step["event"] == "tool":
            observation = step["observation"]
            outcome = "OK" if observation["ok"] else observation["error"]["code"]
            print(f"Iteration {step['iteration']}: {step['action']} -> {outcome}")
        else:
            print(f"Iteration {step['iteration']}: {step['event']} -> {step['reason']}")
    print(f"Status: {result.status}")
    print(
        f"Model calls: {result.model_calls}, Tool attempts: {result.tool_attempts}, "
        f"Recoverable errors: {result.errors}"
    )
    print(f"Answer: {result.answer}")


if __name__ == "__main__":
    main()