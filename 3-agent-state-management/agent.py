"""Lab 3: resumable agent lifecycle, state transitions, events, checkpoints.

Independent of earlier labs; reads the common root .env for optional live LLMs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from state import AgentState, CheckpointStore, Status


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


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        if len({t.name for t in tools}) != len(tools):
            raise ValueError("Duplicate tool name")
        self.tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def execute(
        self, call: ToolCall, successful: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        if call.name not in self.tools:
            return {"ok": False, "error": "UNKNOWN_TOOL"}
        try:
            args = json.loads(call.arguments)
            if not isinstance(args, dict):
                raise ValueError("Arguments must be an object")

            # A narrow dependency policy retained only for this revenue example.
            if call.name == "get_orders":
                customer = successful.get("get_customer", {}).get("customer")
                if customer is None:
                    return {"ok": False, "error": "PREREQUISITE_MISSING"}
                if args.get("customer") != customer["name"]:
                    return {"ok": False, "error": "DATA_MISMATCH"}
            if call.name == "sum_amounts":
                orders = successful.get("get_orders", {}).get("orders")
                if orders is None:
                    return {"ok": False, "error": "PREREQUISITE_MISSING"}
                if args.get("amounts") != [o["amount"] for o in orders]:
                    return {"ok": False, "error": "DATA_MISMATCH"}

            data = self.tools[call.name].handler(args)
        except (ValueError, TypeError, KeyError):
            return {"ok": False, "error": "INVALID_ARGUMENTS"}
        except Exception:
            return {"ok": False, "error": "TOOL_EXECUTION_FAILED"}
        return {"ok": True, "data": data}


@dataclass(frozen=True)
class Limits:
    max_model_calls: int = 8
    max_tool_attempts: int = 8
    max_errors: int = 3

    def __post_init__(self) -> None:
        if min(self.max_model_calls, self.max_tool_attempts, self.max_errors) < 1:
            raise ValueError("Limits must be positive")


class AgentRunner:
    def __init__(
        self, model: Model, registry: ToolRegistry, limits: Limits | None = None
    ):
        self.model = model
        self.registry = registry
        self.limits = limits or Limits()

    @staticmethod
    def can_finish(state: AgentState) -> bool:
        customer = state.successful.get("get_customer")
        if customer is not None and customer["customer"] is None:
            return True
        return "sum_amounts" in state.successful

    @staticmethod
    def persist(store: CheckpointStore | None, state: AgentState) -> None:
        if store is not None:
            store.save(state)

    def finish(
        self,
        state: AgentState,
        status: Status,
        reason: str,
        store: CheckpointStore | None,
    ) -> AgentState:
        state.failure_reason = "" if status == Status.COMPLETED else reason
        state.transition(status, reason=reason)
        self.persist(store, state)
        return state

    def execute(
        self,
        state: AgentState,
        *,
        store: CheckpointStore | None = None,
        pause_after_tools: int | None = None,
    ) -> AgentState:
        if pause_after_tools is not None and pause_after_tools < 1:
            raise ValueError("pause_after_tools must be positive")
        if state.status not in {Status.CREATED, Status.PAUSED, Status.RUNNING}:
            raise ValueError(f"Run cannot continue from {state.status.value}")

        if state.status == Status.PAUSED:
            state.transition(Status.RUNNING, reason="resume")
            state.emit(
                "run_resumed",
                model_calls=state.model_calls,
                tool_attempts=state.tool_attempts,
            )
        elif state.status == Status.CREATED:
            state.transition(Status.RUNNING, reason="start")

        attempts_at_entry = state.tool_attempts
        while True:
            if state.model_calls >= self.limits.max_model_calls:
                return self.finish(
                    state, Status.BUDGET_EXCEEDED, "MODEL_CALL_LIMIT", store
                )
            state.transition(Status.WAITING_FOR_MODEL)
            # Count provider attempts even when a timeout/error occurs.
            state.model_calls += 1
            try:
                turn = self.model.respond(state.messages, self.registry.schemas())
            except Exception:
                return self.finish(state, Status.FAILED, "MODEL_PROVIDER_ERROR", store)
            state.transition(Status.RUNNING)
            state.emit(
                "model_decision",
                model_call=state.model_calls,
                tool_names=[c.name for c in turn.tool_calls],
                has_answer=bool(turn.content),
            )

            if turn.tool_calls:
                if len(turn.tool_calls) != 1:
                    return self.finish(
                        state, Status.FAILED, "MULTI_TOOL_UNSUPPORTED", store
                    )
                call = turn.tool_calls[0]
                if not call.id or call.id in state.used_call_ids:
                    return self.finish(
                        state, Status.FAILED, "DUPLICATE_TOOL_CALL_ID", store
                    )
                if state.tool_attempts >= self.limits.max_tool_attempts:
                    return self.finish(
                        state, Status.BUDGET_EXCEEDED, "TOOL_CALL_LIMIT", store
                    )

                state.used_call_ids.append(call.id)
                state.messages.append(
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
                    }
                )
                state.transition(Status.WAITING_FOR_TOOL)
                observation = self.registry.execute(call, state.successful)
                state.tool_attempts += 1
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(observation, ensure_ascii=False),
                    }
                )
                if observation["ok"]:
                    state.successful[call.name] = observation["data"]
                else:
                    state.errors += 1

                state.emit(
                    "tool_result",
                    model_call=state.model_calls,
                    tool_call_id=call.id,
                    tool=call.name,
                    ok=observation["ok"],
                    error=observation.get("error"),
                )
                state.transition(Status.RUNNING)
                if state.errors >= self.limits.max_errors:
                    return self.finish(state, Status.FAILED, "TOOL_ERROR_LIMIT", store)

                if pause_after_tools is not None and (
                    state.tool_attempts - attempts_at_entry >= pause_after_tools
                ):
                    state.transition(Status.PAUSED, reason="manual_pause_after_tool")
                    self.persist(store, state)
                    return state

                # A settled tool boundary: resume will never replay this call.
                self.persist(store, state)
                continue

            if turn.content:
                if not self.can_finish(state):
                    return self.finish(
                        state, Status.FAILED, "FINAL_BEFORE_EVIDENCE", store
                    )
                state.answer = turn.content
                return self.finish(state, Status.COMPLETED, "final_answer", store)

            return self.finish(state, Status.FAILED, "EMPTY_MODEL_TURN", store)

    @staticmethod
    def cancel(state: AgentState, store: CheckpointStore | None = None) -> AgentState:
        if state.status not in {Status.CREATED, Status.RUNNING, Status.PAUSED}:
            raise ValueError(f"Cannot cancel from {state.status.value}")
        state.transition(Status.CANCELLED, reason="user_cancelled")
        if store is not None:
            store.save(state)
        return state


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
            if o["customer"] == args["customer"] and o["status"] == "paid"
        ]
    }


def sum_amounts(args: dict[str, Any]) -> dict[str, Any]:
    amounts = args.get("amounts")
    if (
        set(args) != {"amounts"}
        or not isinstance(amounts, list)
        or any(type(v) is not int or v < 0 for v in amounts)
    ):
        raise ValueError("Expected non-negative integer amounts")
    return {"total": sum(amounts)}


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                "get_customer",
                "Find a customer by integer ID.",
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
                "Read paid orders for a resolved customer.",
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
                "Sum the order amounts from get_orders.",
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
    """Stateless model that derives its next decision from restored messages."""

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        question = next(msg["content"] for msg in messages if msg["role"] == "user")
        match = re.search(r"customer_id\s*=\s*(\d+)", question)
        if not match:
            return ModelTurn(content="Please provide a customer_id.")
        customer_id = int(match.group(1))
        tool_messages = [msg for msg in messages if msg["role"] == "tool"]
        successes: dict[str, dict[str, Any]] = {}
        for index, message in enumerate(messages):
            if message["role"] != "tool":
                continue
            observation = json.loads(message["content"])
            if observation["ok"]:
                previous = messages[index - 1]
                tool_call = previous["tool_calls"][0]
                if tool_call["id"] != message["tool_call_id"]:
                    raise ValueError("Tool result correlation mismatch")
                successes[tool_call["function"]["name"]] = observation["data"]
        # ID depends on message history, so it stays unique across process restarts.
        next_id = f"mock_call_{len(tool_messages) + 1}"
        if "get_customer" not in successes:
            return ModelTurn(
                tool_calls=(
                    ToolCall(
                        next_id,
                        "get_customer",
                        json.dumps({"customer_id": customer_id}),
                    ),
                )
            )
        customer = successes["get_customer"]["customer"]
        if customer is None:
            return ModelTurn(content=f"Customer {customer_id} was not found.")
        if "get_orders" not in successes:
            return ModelTurn(
                tool_calls=(
                    ToolCall(
                        next_id,
                        "get_orders",
                        json.dumps({"customer": customer["name"], "status": "paid"}),
                    ),
                )
            )
        if "sum_amounts" not in successes:
            orders = successes["get_orders"]["orders"]
            return ModelTurn(
                tool_calls=(
                    ToolCall(
                        next_id,
                        "sum_amounts",
                        json.dumps({"amounts": [order["amount"] for order in orders]}),
                    ),
                )
            )
        return ModelTurn(
            content=(
                f"Paid revenue for {customer['name']}: {successes['sum_amounts']['total']:,} VND"
            )
        )


def openai_compatible_model() -> Model:
    """Load ONLY the root workspace .env, for OpenAI or OpenAI-compatible vLLM."""
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install root requirements.txt first") from exc
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        raise ValueError("Configure LLM_MODEL in root .env")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY") or "EMPTY",
        base_url=os.getenv("LLM_BASE_URL") or None,
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
    )

    class OpenAICompatibleModel:
        def respond(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> ModelTurn:
            answer = (
                client.chat.completions.create(
                    model=model_name, messages=messages, tools=tools, tool_choice="auto"
                )
                .choices[0]
                .message
            )
            return ModelTurn(
                answer.content,
                tuple(
                    ToolCall(t.id, t.function.name, t.function.arguments)
                    for t in (answer.tool_calls or [])
                ),
            )

    return OpenAICompatibleModel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 3: Agent State Management")
    parser.add_argument(
        "--question", default="Calculate paid order revenue for customer_id=1."
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="JSON file for checkpoint persistence"
    )
    control = parser.add_mutually_exclusive_group()
    control.add_argument(
        "--resume", action="store_true", help="Resume a paused/settled run"
    )
    control.add_argument(
        "--cancel", action="store_true", help="Cancel a saved paused/settled run"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock", action="store_true", help="Force offline model, overriding LIVE_MODE"
    )
    mode.add_argument("--live", action="store_true", help="Use shared-root real LLM")
    parser.add_argument(
        "--pause-after-tools", type=int, help="Pause after N settled tool attempts"
    )
    args = parser.parse_args()

    if args.resume or args.cancel or args.pause_after_tools is not None:
        if args.checkpoint is None:
            parser.error(
                "--resume, --cancel and --pause-after-tools require --checkpoint"
            )

    store = CheckpointStore(args.checkpoint) if args.checkpoint else None
    if args.resume or args.cancel:
        try:
            state = store.load()
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    else:
        if store is not None and store.path.exists():
            parser.error("Checkpoint exists; use --resume or a fresh checkpoint path")
        state = AgentState.new(args.question)

    if args.cancel:
        try:
            state = AgentRunner.cancel(state, store)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        live = args.live or (not args.mock and os.getenv("LIVE_MODE") == "1")
        print(f"Mode: {'live' if live else 'mock'}")
        model = openai_compatible_model() if live else MockModel()
        runner = AgentRunner(model, build_registry())
        try:
            state = runner.execute(
                state, store=store, pause_after_tools=args.pause_after_tools
            )
        except ValueError as exc:
            parser.error(str(exc))

    print(f"Run ID: {state.run_id}")
    for event in state.events:
        if event["kind"] == "state_changed":
            print(f"Event {event['seq']}: {event['previous']} -> {event['current']}")
        elif event["kind"] == "tool_result":
            print(
                f"Event {event['seq']}: {event['tool']} -> "
                f"{'OK' if event['ok'] else event['error']}"
            )
    print(f"Status: {state.status.value}")
    print(f"Model calls: {state.model_calls}, Tool attempts: {state.tool_attempts}")
    if state.failure_reason and state.status != Status.COMPLETED:
        print(f"Reason: {state.failure_reason}")
    if store:
        print(f"Checkpoint: {store.path}")
    if state.answer:
        print(f"Answer: {state.answer}")


if __name__ == "__main__":
    main()
