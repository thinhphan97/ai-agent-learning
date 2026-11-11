"""Explicit memory tools with a stable allowlist and runtime-controlled user scope."""

from __future__ import annotations

import json
import re
from typing import Any

from memory import MemoryStore
from model import ToolCall


def _schema(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


MEMORY_TOOL_SCHEMAS = [
    _schema(
        "save_memory",
        "Persist a user preference only when the user explicitly asked to remember it",
        {
            "key": {"type": "string", "description": "Lowercase snake_case fact key; use programming_language for preferred programming language"},
            "value": {"type": "string", "description": "Exact value from the current user request"},
        },
    ),
    _schema(
        "recall_memory",
        "Retrieve a previously saved preference from the current user's long-term memory",
        {"key": {"type": "string", "description": "Lowercase snake_case fact key; use programming_language for preferred programming language"}},
    ),
    _schema(
        "forget_memory",
        "Delete a previously saved preference after the user explicitly asks to forget it",
        {"key": {"type": "string", "description": "Lowercase snake_case fact key; use programming_language for preferred programming language"}},
    ),
]


# Recognized aliases for the teaching goal. This is deliberately NOT an
# open-ended semantic key mapper: unrestricted aliases could conflate facts.
KEY_ALIASES = {
    "preferred_programming_language": "programming_language",
    "programming_language_preference": "programming_language",
}


class MemoryTools:
    """The model never receives or chooses user_id: it is injected by the runtime."""

    def __init__(self, store: MemoryStore, user_id: str):
        self.store = store
        self.user_id = store.validate_user(user_id)

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return MEMORY_TOOL_SCHEMAS

    def execute(self, call: ToolCall, user_request: str) -> dict[str, Any]:
        if call.name not in {"save_memory", "recall_memory", "forget_memory"}:
            return {"ok": False, "code": "UNKNOWN_TOOL"}
        try:
            args = json.loads(call.arguments)
        except (ValueError, TypeError):
            return {"ok": False, "code": "INVALID_ARGUMENTS"}
        if not isinstance(args, dict):
            return {"ok": False, "code": "INVALID_ARGUMENTS"}
        expected = {"key", "value"} if call.name == "save_memory" else {"key"}
        if set(args) != expected:
            return {"ok": False, "code": "INVALID_ARGUMENTS"}
        try:
            proposed_key = self.store.validate_key(args["key"])
            key = KEY_ALIASES.get(proposed_key, proposed_key)
            if call.name == "save_memory":
                if not re.match(
                    r"^\s*(?:please\s+)?remember\b", user_request, re.IGNORECASE
                ):
                    return {"ok": False, "code": "EXPLICIT_CONSENT_REQUIRED"}
                value = self.store.validate_value(args["value"])
                if value.casefold() not in user_request.casefold():
                    return {"ok": False, "code": "UNGROUNDED_MEMORY_VALUE"}
                fact = self.store.save(self.user_id, key, value)
                return {"ok": True, "key": fact.key, "value": fact.value}
            if call.name == "recall_memory":
                fact = self.store.get(self.user_id, key)
                if fact is None:
                    # Read-only compatibility for earlier files where a model
                    # saved a noncanonical alias; never mutate on retrieval.
                    for alias, canonical in KEY_ALIASES.items():
                        if canonical == key:
                            fact = self.store.get(self.user_id, alias)
                            if fact is not None:
                                break
                if fact is None:
                    return {"ok": True, "key": key, "found": False}
                return {"ok": True, "key": key, "found": True, "value": fact.value}
            if not re.match(
                r"^\s*(?:please\s+)?forget\b", user_request, re.IGNORECASE
            ):
                return {"ok": False, "code": "EXPLICIT_CONSENT_REQUIRED"}
            deleted = self.store.forget(self.user_id, key)
            for alias, canonical in KEY_ALIASES.items():
                if canonical == key and self.store.forget(self.user_id, alias):
                    deleted = True
            return {"ok": True, "key": key, "deleted": deleted}
        except ValueError:
            return {"ok": False, "code": "INVALID_ARGUMENTS"}