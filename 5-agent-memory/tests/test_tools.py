"""Explicit memory tools: permission, provenance, and argument validation."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory import MemoryStore
from model import ToolCall
from tools import MemoryTools


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MemoryStore(Path(self.temp.name) / "test.sqlite3")
        self.tools = MemoryTools(self.store, "alice")

    def call(self, name, args, request):
        return self.tools.execute(ToolCall("id1", name, json.dumps(args)), request)

    def test_save_and_recall(self):
        result = self.call(
            "save_memory",
            {"key": "programming_language", "value": "Python"},
            "Remember that my preferred programming language is Python.",
        )
        self.assertTrue(result["ok"])
        recalled = self.call("recall_memory", {"key": "programming_language"}, "Recall it")
        self.assertEqual(recalled["value"], "Python")
        self.assertTrue(recalled["found"])

    def test_no_write_without_explicit_request(self):
        observed = self.call(
            "save_memory", {"key": "programming_language", "value": "Python"},
            "My preferred programming language is Python.",
        )
        self.assertEqual(observed["code"], "EXPLICIT_CONSENT_REQUIRED")
        self.assertIsNone(self.store.get("alice", "programming_language"))

    def test_reject_value_not_grounded_in_request(self):
        result = self.call(
            "save_memory",
            {"key": "programming_language", "value": "Java"},
            "Remember that my preferred programming language is Python.",
        )
        self.assertEqual(result["code"], "UNGROUNDED_MEMORY_VALUE")
        self.assertIsNone(self.store.get("alice", "programming_language"))

    def test_volatile_chat_note_not_persisted(self):
        result = self.call(
            "save_memory", {"key": "project_codename", "value": "Orion"},
            "For this chat, my project codename is Orion.",
        )
        self.assertEqual(result["code"], "EXPLICIT_CONSENT_REQUIRED")

    def test_canonical_key_is_stable_when_model_uses_alias(self):
        saved = self.call(
            "save_memory",
            {"key": "preferred_programming_language", "value": "Python"},
            "Remember that my preferred programming language is Python.",
        )
        self.assertTrue(saved["ok"])
        self.assertEqual(saved["key"], "programming_language")
        self.assertEqual(
            self.store.get("alice", "programming_language").value, "Python"
        )
        self.assertIsNone(self.store.get("alice", "preferred_programming_language"))
        recalled = self.call(
            "recall_memory", {"key": "programming_language"}, "What is my preference?"
        )
        self.assertEqual(recalled["value"], "Python")

    def test_legacy_key_alias_is_readable_and_forgettable(self):
        self.store.save("alice", "preferred_programming_language", "Python")
        recalled = self.call(
            "recall_memory", {"key": "programming_language"}, "What do I prefer?"
        )
        self.assertTrue(recalled["found"])
        self.assertEqual(recalled["value"], "Python")
        self.assertIsNone(self.store.get("alice", "programming_language"))
        forgotten = self.call(
            "forget_memory", {"key": "programming_language"},
            "Forget my preferred programming language."
        )
        self.assertTrue(forgotten["deleted"])
        self.assertIsNone(self.store.get("alice", "preferred_programming_language"))

    def test_recall_miss(self):
        result = self.call("recall_memory", {"key": "programming_language"}, "What do I prefer?")
        self.assertEqual(result, {
            "ok": True, "key": "programming_language", "found": False,
        })

    def test_forget_needs_permission(self):
        self.store.save("alice", "language", "Python")
        refused = self.call("forget_memory", {"key": "language"}, "Tell me my language.")
        self.assertEqual(refused["code"], "EXPLICIT_CONSENT_REQUIRED")
        self.assertIsNotNone(self.store.get("alice", "language"))
        deleted = self.call("forget_memory", {"key": "language"}, "Forget my language.")
        self.assertTrue(deleted["deleted"])
        self.assertIsNone(self.store.get("alice", "language"))

    def test_reject_unknown_tool(self):
        result = self.call("delete_database", {"key": "anything"}, "Forget all")
        self.assertEqual(result["code"], "UNKNOWN_TOOL")

    def test_reject_non_json(self):
        result = self.tools.execute(ToolCall("1", "save_memory", "{bad"), "Remember Python")
        self.assertEqual(result["code"], "INVALID_ARGUMENTS")

    def test_reject_wrong_shape_and_scope_injection(self):
        for args in (
            ["key", "value"],
            {"key": "a"},
            {"key": "a", "value": "Python", "user_id": "bob"},
        ):
            with self.subTest(args=args):
                result = self.call("save_memory", args, "Remember Python")
                self.assertEqual(result["code"], "INVALID_ARGUMENTS")

    def test_user_scope_not_supplied_by_llm(self):
        self.store.save("bob", "language", "Rust")
        observed = self.call("recall_memory", {"key": "language"}, "What do I prefer?")
        self.assertFalse(observed["found"])
        self.assertEqual(self.store.get("bob", "language").value, "Rust")

    def test_schema_has_no_user_id_and_all_params_required(self):
        schemas = self.tools.schemas()
        self.assertEqual({s["function"]["name"] for s in schemas},
                         {"save_memory", "recall_memory", "forget_memory"})
        for schema in schemas:
            params = schema["function"]["parameters"]
            self.assertNotIn("user_id", params["properties"])
            self.assertEqual(set(params["required"]), set(params["properties"]))
            self.assertFalse(params["additionalProperties"])


if __name__ == "__main__":
    unittest.main()