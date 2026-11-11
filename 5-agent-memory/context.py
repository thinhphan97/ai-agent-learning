"""Bounded short-term model context: complete recent conversation turns."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class ConversationWindow:
    """Keep the last N complete user->assistant turns, not a durable memory.

    A turn may include assistant tool calls and corresponding tool responses.
    Trimming by complete turns avoids orphaned OpenAI tool messages.
    """

    def __init__(self, max_turns: int = 3):
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        self.max_turns = max_turns
        self._turns: list[list[dict[str, Any]]] = []

    def add_turn(self, messages: list[dict[str, Any]]) -> None:
        if (
            not messages
            or messages[0].get("role") != "user"
            or messages[-1].get("role") != "assistant"
        ):
            raise ValueError("A complete turn must start with user and end with assistant")
        self._turns.append(deepcopy(messages))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    def messages(self) -> list[dict[str, Any]]:
        return [msg for turn in deepcopy(self._turns) for msg in turn]

    @property
    def turn_count(self) -> int:
        return len(self._turns)
