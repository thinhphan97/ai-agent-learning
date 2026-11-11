"""End-to-end agent memory behavior, budgets and isolated sessions."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import Limits, MemoryAgent
from context import ConversationWindow
from memory import MemoryStore
from model import ModelTurn, MockModel, ToolCall


SAVE_QUESTION = "Remember that my preferred programming language is Python."
RECALL_QUESTION = "What programming language do I prefer?"
TEMP_QUESTION = "For this chat, my project codename is Orion."
TEMP_RECALL = "What is the project codename?"


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.store = MemoryStore(self.path)

    def agent(self, user="alice", model=None, **kwargs):
        return MemoryAgent(model or MockModel(), self.store, user, **kwargs)

    def test_short_term_and_long_term_separation(self):
        agent = self.agent()
        self.assertEqual(agent.ask(SAVE_QUESTION).status, "completed")
        self.assertEqual(agent.ask(RECALL_QUESTION).answer,
                         "Your saved programming language preference is Python.")
        self.assertIn("Orion", agent.ask(TEMP_QUESTION).answer)
        self.assertIn("Orion", agent.ask(TEMP_RECALL).answer)

        new_session = self.agent()
        self.assertIn("Python", new_session.ask(RECALL_QUESTION).answer)
        self.assertEqual(
            new_session.ask(TEMP_RECALL).answer,
            "There is no project codename in this conversation.",
        )

    def test_cross_user_cannot_recall(self):
        self.agent("alice").ask(SAVE_QUESTION)
        bob = self.agent("bob")
        self.assertIn("no saved", bob.ask(RECALL_QUESTION).answer)
        self.assertIsNone(self.store.get("bob", "programming_language"))

    def test_restart_store_instance(self):
        first = self.agent()
        first.ask(SAVE_QUESTION)
        restored = MemoryAgent(MockModel(), MemoryStore(self.path), "alice")
        self.assertIn("Python", restored.ask(RECALL_QUESTION).answer)

    def test_forget_and_recall_miss(self):
        a = self.agent()
        self.assertEqual(a.ask(SAVE_QUESTION).status, "completed")
        result = a.ask("Forget my preferred programming language.")
        self.assertEqual(result.status, "completed")
        self.assertIn("Forgot", result.answer)
        self.assertIn("no saved", self.agent().ask(RECALL_QUESTION).answer)

    def test_no_auto_persistence_for_temporary_fact(self):
        a = self.agent()
        a.ask(TEMP_QUESTION)
        self.assertIsNone(self.store.get("alice", "project_codename"))
        self.assertEqual(a.context.turn_count, 1)

    def test_short_term_context_eviction(self):
        a = self.agent(context=ConversationWindow(max_turns=1))
        a.ask(TEMP_QUESTION)
        self.assertIn("Orion", a.ask(TEMP_RECALL).answer)
        a.ask("Unrelated message")
        self.assertNotIn("Orion", a.ask(TEMP_RECALL).answer)

    def test_tool_history_keeps_matching_tool_messages(self):
        a = self.agent()
        r = a.ask(SAVE_QUESTION)
        self.assertEqual(r.tool_calls, 1)
        self.assertEqual(r.model_calls, 2)
        messages = a.context.messages()
        self.assertEqual(
            [m["role"] for m in messages], ["user", "assistant", "tool", "assistant"]
        )
        self.assertEqual(
            messages[1]["tool_calls"][0]["id"], messages[2]["tool_call_id"]
        )

    def test_save_requires_actual_tool_not_claim(self):
        class Fabricator:
            def complete(self, messages, tools):
                return ModelTurn(content="I saved that preference.")

        a = self.agent(model=Fabricator())
        result = a.ask(SAVE_QUESTION)
        self.assertEqual(result.status, "protocol_error")
        self.assertIsNone(self.store.get("alice", "programming_language"))

    def test_forget_requires_actual_tool_not_claim(self):
        self.store.save("alice", "programming_language", "Python")
        class Fabricator:
            def complete(self, messages, tools):
                return ModelTurn(content="I forgot it.")

        result = self.agent(model=Fabricator()).ask(
            "Forget my preferred programming language."
        )
        self.assertEqual(result.status, "protocol_error")
        self.assertIsNotNone(self.store.get("alice", "programming_language"))

    def test_model_error_masked(self):
        class BadModel:
            def complete(self, messages, tools):
                raise RuntimeError("secret internal message")

        result = self.agent(model=BadModel()).ask(RECALL_QUESTION)
        self.assertEqual(result.status, "model_error")
        self.assertNotIn("secret", result.answer)

    def test_empty_final_rejected(self):
        class EmptyModel:
            def complete(self, messages, tools):
                return ModelTurn(content=None)

        result = self.agent(model=EmptyModel()).ask("Hello")
        self.assertEqual(result.status, "protocol_error")

    def test_unknown_tool_rejected(self):
        class UnknownTool:
            def complete(self, messages, tools):
                return ModelTurn(tool_calls=(ToolCall("bad", "drop_table", "{}"),))

        result = self.agent(model=UnknownTool()).ask(SAVE_QUESTION)
        self.assertEqual(result.status, "tool_rejected")
        self.assertIn("UNKNOWN_TOOL", result.answer)
        self.assertIsNone(self.store.get("alice", "programming_language"))

    def test_duplicate_call_id_rejected(self):
        class Duplicate:
            def complete(self, messages, tools):
                return ModelTurn(tool_calls=(ToolCall(
                    "duplicate", "recall_memory", '{"key":"programming_language"}'
                ),))

        a = self.agent(model=Duplicate())
        result = a.ask(RECALL_QUESTION)
        self.assertEqual(result.status, "protocol_error")
        self.assertEqual(result.tool_calls, 1)

    def test_multi_tool_call_rejected(self):
        class Multi:
            def complete(self, messages, tools):
                call = ToolCall("id", "recall_memory", '{"key":"programming_language"}')
                return ModelTurn(tool_calls=(call, call))

        self.assertEqual(self.agent(model=Multi()).ask(RECALL_QUESTION).status, "protocol_error")

    def test_tool_budget_rejected(self):
        class Repeated:
            def __init__(self):
                self.count = 0

            def complete(self, messages, tools):
                self.count += 1
                return ModelTurn(tool_calls=(ToolCall(
                    f"id{self.count}", "recall_memory",
                    '{"key":"programming_language"}'
                ),))

        agent = self.agent(model=Repeated(), limits=Limits(max_tool_calls=1))
        result = agent.ask(RECALL_QUESTION)
        self.assertEqual(result.status, "protocol_error")
        self.assertEqual(result.tool_calls, 1)

    def test_model_budget_enforced(self):
        class NeverFinished:
            def __init__(self):
                self.count = 0

            def complete(self, messages, tools):
                self.count += 1
                return ModelTurn(tool_calls=(ToolCall(
                    f"id{self.count}", "recall_memory",
                    '{"key":"programming_language"}'
                ),))

        result = self.agent(
            model=NeverFinished(), limits=Limits(max_model_calls=2, max_tool_calls=3)
        ).ask(RECALL_QUESTION)
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.model_calls, 2)

    def test_invalid_questions_do_not_consume_provider(self):
        mock = MockModel()
        a = self.agent(model=mock, limits=Limits(max_user_chars=10))
        self.assertEqual(a.ask("").status, "invalid_request")
        self.assertEqual(a.ask("x" * 11).status, "invalid_request")
        self.assertEqual(mock.calls, 0)

    def test_invalid_limits_rejected(self):
        for args in (
            {"max_model_calls": 0},
            {"max_tool_calls": -1},
            {"max_user_chars": True},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Limits(**args)

    def test_cross_process_memory_persistence(self):
        command = [sys.executable, str(ROOT / "agent.py"), "--mock",
                   "--memory-db", str(self.path), "--user", "alice"]
        write = subprocess.run(
            [*command, "--ask", SAVE_QUESTION],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(write.returncode, 0, write.stderr)
        self.assertIn("Saved programming_language = Python", write.stdout)
        read = subprocess.run(
            [*command, "--ask", RECALL_QUESTION],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(read.returncode, 0, read.stderr)
        self.assertIn("saved programming language preference is Python", read.stdout)
        bob = subprocess.run(
            [*command[:], "--ask", RECALL_QUESTION, "--user", "bob"],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(bob.returncode, 0, bob.stderr)
        self.assertIn("no saved", bob.stdout)


if __name__ == "__main__":
    unittest.main()
