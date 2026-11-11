"""Lab 5 CLI and orchestration: short-term context + durable explicit memory.

Run: python 5-agent-memory/agent.py --mock --demo
     python 5-agent-memory/agent.py --mock --ask "Remember that my preferred programming language is Python."
     python 5-agent-memory/agent.py --mock --ask "What programming language do I prefer?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context import ConversationWindow
from memory import MemoryStore
from model import Model, MockModel, SYSTEM_PROMPT, openai_compatible_model
from tools import MemoryTools


DEFAULT_DB = Path(__file__).resolve().parent / "data" / "memory.sqlite3"


@dataclass(frozen=True)
class Limits:
    max_model_calls: int = 4
    max_tool_calls: int = 2
    max_user_chars: int = 1000

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or n < 1
            for n in (self.max_model_calls, self.max_tool_calls, self.max_user_chars)
        ):
            raise ValueError("All limits must be positive integers")


@dataclass
class TurnResult:
    status: str
    answer: str
    model_calls: int
    tool_calls: int
    events: list[dict[str, Any]] = field(default_factory=list)


class MemoryAgent:
    """Owns one session's volatile context; persistent facts belong to MemoryStore."""

    def __init__(
        self,
        model: Model,
        store: MemoryStore,
        user_id: str,
        *,
        context: ConversationWindow | None = None,
        limits: Limits | None = None,
    ):
        self.model = model
        self.tools = MemoryTools(store, user_id)
        self.context = context if context is not None else ConversationWindow(max_turns=3)
        self.limits = limits or Limits()

    def ask(self, question: str) -> TurnResult:
        if not isinstance(question, str) or not question.strip():
            return TurnResult("invalid_request", "A nonempty question is required.", 0, 0)
        if len(question) > self.limits.max_user_chars:
            return TurnResult("invalid_request", "Question exceeds the input limit.", 0, 0)

        question = question.strip()
        current: list[dict[str, Any]] = [{"role": "user", "content": question}]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.context.messages(),
            *current,
        ]
        events: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        tool_calls = 0
        required_action = None
        if re.match(r"^(?:please\s+)?remember\b", question, re.IGNORECASE):
            required_action = "save_memory"
        if re.match(r"^(?:please\s+)?forget\b", question, re.IGNORECASE):
            required_action = "forget_memory"
        completed_actions: set[str] = set()

        def finish(status: str, answer: str, model_calls: int) -> TurnResult:
            current.append({"role": "assistant", "content": answer})
            self.context.add_turn(current)
            return TurnResult(status, answer, model_calls, tool_calls, events)

        for attempt in range(1, self.limits.max_model_calls + 1):
            events.append({"event": "model_requested", "call": attempt})
            try:
                turn = self.model.complete(messages, self.tools.schemas())
            except Exception:
                return finish("model_error", "The model request failed.", attempt)
            if turn.tool_calls:
                if len(turn.tool_calls) != 1:
                    return finish("protocol_error", "Expected exactly one tool call.", attempt)
                call = turn.tool_calls[0]
                if (
                    not call.id or call.id in seen_ids or tool_calls >= self.limits.max_tool_calls
                ):
                    return finish("protocol_error", "Tool call ID or budget was invalid.", attempt)
                seen_ids.add(call.id)
                tool_calls += 1
                assistant_message = {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                    ],
                }
                observed = self.tools.execute(call, question)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(observed, ensure_ascii=False),
                }
                current.extend([assistant_message, tool_message])
                messages.extend([assistant_message, tool_message])
                events.append({
                    "event": "memory_tool_result",
                    "tool": call.name, "ok": observed["ok"],
                    **({"code": observed["code"]} if not observed["ok"] else {}),
                })
                if not observed["ok"]:
                    return finish(
                        "tool_rejected", f"Memory operation rejected: {observed['code']}.",
                        attempt,
                    )
                completed_actions.add(call.name)
                continue

            if required_action and required_action not in completed_actions:
                return finish(
                    "protocol_error",
                    "The model did not perform the explicitly requested memory operation.",
                    attempt,
                )
            if not turn.content or not turn.content.strip():
                return finish("protocol_error", "Model returned no answer.", attempt)
            return finish("completed", turn.content.strip(), attempt)

        return finish(
            "budget_exceeded", "Model-call budget exceeded before a final answer.",
            self.limits.max_model_calls,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 5: Short- and Long-Term Agent Memory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock", action="store_true", help="Force deterministic offline model")
    group.add_argument("--live", action="store_true", help="Use real LLM configured in root .env")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--ask", help="Ask one question; starts a fresh short-term session")
    action.add_argument("--demo", action="store_true", help="Run a multi-turn learning demonstration")
    parser.add_argument("--user", default="student", help="Memory namespace (do not use a secret)")
    parser.add_argument(
        "--memory-db", type=Path, default=DEFAULT_DB,
        help="SQLite file for persistent memory (default: this lab's data/memory.sqlite3)",
    )
    parser.add_argument("--window-turns", type=int, default=3)
    args = parser.parse_args()

    live = args.live or (not args.mock and os.getenv("LIVE_MODE") == "1")
    model = openai_compatible_model() if live else MockModel()
    store = MemoryStore(args.memory_db)
    agent = MemoryAgent(
        model, store, args.user,
        context=ConversationWindow(args.window_turns),
    )
    print(f"Mode: {'live' if live else 'mock'}; user scope: {args.user}")
    if args.ask:
        result = agent.ask(args.ask)
        print("Question:", args.ask)
        print("Answer:", result.answer)
        print(f"Status: {result.status}; model calls: {result.model_calls}; tool calls: {result.tool_calls}")
        for event in result.events:
            if event["event"] == "memory_tool_result":
                print("Memory tool:", event["tool"], "OK" if event["ok"] else event["code"])
        return

    questions = [
        "Remember that my preferred programming language is Python.",
        "What programming language do I prefer?",
        "For this chat, my project codename is Orion.",
        "What is the project codename?",
    ]
    for question in questions:
        result = agent.ask(question)
        print(f"Q: {question}\nA: {result.answer} [{result.status}]")
    print("--- New session, same user and same SQLite database ---")
    fresh = MemoryAgent(
        model, MemoryStore(args.memory_db), args.user,
        context=ConversationWindow(args.window_turns),
    )
    for question in (
        "What programming language do I prefer?",
        "What is the project codename?",
    ):
        result = fresh.ask(question)
        print(f"Q: {question}\nA: {result.answer} [{result.status}]")


if __name__ == "__main__":
    main()
