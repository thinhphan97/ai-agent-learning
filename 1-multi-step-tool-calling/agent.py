"""Independent teaching implementation: ReAct loop, tool contracts and model adapter."""

from __future__ import annotations

import json
import os
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
    handler: Callable[[dict[str, Any]], Any]

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
    def __init__(self, tools: list[Tool]):
        if len({t.name for t in tools}) != len(tools):
            raise ValueError("Duplicate tool name")
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, call: ToolCall) -> dict[str, Any]:
        tool = self._tools.get(call.name)
        if tool is None:
            return {"ok": False, "error": "UNKNOWN_TOOL"}
        try:
            args = json.loads(call.arguments)
            if not isinstance(args, dict):
                raise ValueError("Arguments must be an object")
            # Tool handlers validate their inputs independently; schema alone does not enforce safety.
            return {"ok": True, "data": tool.handler(args)}
        except (ValueError, TypeError, KeyError) as exc:
            return {"ok": False, "error": "INVALID_ARGUMENTS", "detail": str(exc)}


@dataclass
class AgentResult:
    status: str
    answer: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0


class AgentRunner:
    def __init__(
        self,
        model: Model,
        registry: ToolRegistry,
        *,
        max_model_calls: int = 8,
        max_tool_calls: int = 8,
        required_order: tuple[str, ...] = (),
    ):
        if max_model_calls < 1 or max_tool_calls < 1:
            raise ValueError("Budgets must be positive")
        self.model, self.registry = model, registry
        self.max_model_calls, self.max_tool_calls = max_model_calls, max_tool_calls
        self.required_order = required_order

    def run(self, request: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": request}]
        trace: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for iteration in range(1, self.max_model_calls + 1):
            turn = self.model.respond(messages, self.registry.schemas())
            if turn.tool_calls:
                if len(trace) + len(turn.tool_calls) > self.max_tool_calls:
                    return AgentResult(
                        "budget_exceeded", trace=trace, model_calls=iteration
                    )
                if self.required_order:
                    expected = self.required_order[
                        len(trace) : len(trace) + len(turn.tool_calls)
                    ]
                    if (
                        len(turn.tool_calls) != 1
                        or tuple(c.name for c in turn.tool_calls) != expected
                    ):
                        return AgentResult(
                            "policy_violation", trace=trace, model_calls=iteration
                        )
                if any(not c.id or c.id in used_ids for c in turn.tool_calls) or len(
                    {c.id for c in turn.tool_calls}
                ) != len(turn.tool_calls):
                    return AgentResult(
                        "protocol_error", trace=trace, model_calls=iteration
                    )
                for call in turn.tool_calls:
                    used_ids.add(call.id)
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
                            for c in turn.tool_calls
                        ],
                    }
                )
                for call in turn.tool_calls:
                    observation = self.registry.execute(call)
                    trace.append(
                        {
                            "iteration": iteration,
                            "tool_call_id": call.id,
                            "action": call.name,
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
                continue
            if turn.content:
                if self.required_order and len(trace) != len(self.required_order):
                    return AgentResult(
                        "policy_violation", trace=trace, model_calls=iteration
                    )
                return AgentResult("completed", turn.content, trace, iteration)
            return AgentResult("protocol_error", trace=trace, model_calls=iteration)
        return AgentResult(
            "budget_exceeded", trace=trace, model_calls=self.max_model_calls
        )


def openai_compatible_model() -> Model:
    """Load ONE .env at workspace root; works for OpenAI and OpenAI-compatible vLLM."""
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements.txt before running LIVE_MODE=1"
        ) from exc

    root_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=root_env, override=False)
    api_key = os.getenv("LLM_API_KEY") or "EMPTY"
    model_name = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL") or None
    if not model_name:
        raise ValueError("Set LLM_MODEL in workspace root .env")

    class OpenAICompatibleModel:
        def __init__(self):
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(os.getenv("LLM_TIMEOUT", "30")),
            )
            self.model = model_name

        def respond(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> ModelTurn:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto"
            )
            message = response.choices[0].message
            calls = tuple(
                ToolCall(call.id, call.function.name, call.function.arguments)
                for call in (message.tool_calls or [])
            )
            return ModelTurn(message.content, calls)

    return OpenAICompatibleModel()


def last_observation(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message["role"] == "tool":
            return json.loads(message["content"])
    return None


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
        raise ValueError("Expected positive integer customer_id")
    customer = CUSTOMERS.get(args["customer_id"])
    return {"found": customer is not None, "customer": customer}


def get_orders(args: dict[str, Any]) -> dict[str, Any]:
    if (
        set(args) != {"customer", "status"}
        or not isinstance(args["customer"], str)
        or args["status"] != "paid"
    ):
        raise ValueError("Expected customer (string) and status='paid'")
    return {
        "orders": [
            o
            for o in ORDERS
            if o["customer"] == args["customer"] and o["status"] == args["status"]
        ]
    }


def sum_amounts(args: dict[str, Any]) -> dict[str, Any]:
    if (
        set(args) != {"amounts"}
        or not isinstance(args["amounts"], list)
        or any(type(a) is not int or a < 0 for a in args["amounts"])
    ):
        raise ValueError("Expected non-negative integer amounts")
    return {"total": sum(args["amounts"])}


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                "get_customer",
                "Find customer by integer ID",
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
                "Fetch paid customer orders",
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
                "sum_amounts",
                "Sum integer amounts",
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
        ]
    )


class MockModel:
    """Deterministic simulator; does not prove actual LLM reasoning."""

    def __init__(self, customer_id: int = 1):
        self.customer_id = customer_id
        self.calls = 0

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.calls += 1
        observation = last_observation(messages)
        if observation is None:
            return ModelTurn(
                tool_calls=(
                    ToolCall(
                        "call_1",
                        "get_customer",
                        json.dumps({"customer_id": self.customer_id}),
                    ),
                )
            )
        if not observation["ok"]:
            return ModelTurn(content="Tool failed")
        if self.calls == 2:
            customer = observation["data"]["customer"]
            if customer is None:
                return ModelTurn(content="Customer not found")
            return ModelTurn(
                tool_calls=(
                    ToolCall(
                        "call_2",
                        "get_orders",
                        json.dumps({"customer": customer["name"], "status": "paid"}),
                    ),
                )
            )
        if self.calls == 3:
            amounts = [o["amount"] for o in observation["data"]["orders"]]
            return ModelTurn(
                tool_calls=(
                    ToolCall("call_3", "sum_amounts", json.dumps({"amounts": amounts})),
                )
            )
        customer = CUSTOMERS[self.customer_id]["name"]
        return ModelTurn(
            content=f"Paid revenue for {customer}: {observation['data']['total']:,} VND"
        )


def make_runner(model: Model) -> AgentRunner:
    return AgentRunner(
        model,
        build_registry(),
        required_order=("get_customer", "get_orders", "sum_amounts"),
    )


if __name__ == "__main__":
    model = openai_compatible_model() if os.getenv("LIVE_MODE") == "1" else MockModel()
    result = make_runner(model).run(
        "Calculate paid order revenue for customer_id=1. First resolve customer using get_customer, "
        "then get_orders, then sum_amounts. Use one tool per model turn."
    )
    for event in result.trace:
        print(f"Step {event['iteration']}: {event['action']} -> {event['observation']}")
    print(f"Status: {result.status}\nAnswer: {result.answer}")
