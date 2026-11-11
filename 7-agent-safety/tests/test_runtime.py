"""Agent-level safety checks: actions can fail safely and never secretly mutate."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import ApprovalGrant, FinalAnswer, Intent, Limits, ModelTurn, ToolCall
from model import MockModel
from policy import SafetyPolicy
from runtime import SafeAgentRunner
from tools import CustomerStore, ToolRegistry


READ = "Show customer_id=1."
WRITE = "Set note for customer_id=1 to 'Call Friday'."


class RuntimeTests(unittest.TestCase):
    def run_agent(self, scenario="normal", *, intent=None, approve=False,
                  malicious_note=False, principal="alice", limits=None, model=None):
        intent = intent or Intent("read", 1)
        store = CustomerStore(malicious_note=malicious_note)
        grant = ApprovalGrant(principal, intent.customer_id, intent.note) if approve else None
        policy = SafetyPolicy(principal, intent, grant)
        agent = SafeAgentRunner(
            model or MockModel(intent, scenario),
            ToolRegistry(store, policy), policy,
            limits=limits,
        )
        outcome = agent.run(WRITE if intent.kind == "write" else READ)
        return outcome, store

    def test_normal_read_uses_tool_and_verifies_final(self):
        result, store = self.run_agent()
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.code, "VERIFIED")
        self.assertEqual(result.answer, FinalAnswer("read", 1, "Lan"))
        self.assertEqual((result.model_calls, result.tool_attempts), (2, 1))
        self.assertEqual(store.write_count, 0)

    def test_no_approval_denies_note_change(self):
        result, store = self.run_agent(intent=Intent("write", 1, "Call Friday"))
        self.assertEqual(result.status, "denied")
        self.assertEqual(result.code, "APPROVAL_REQUIRED")
        self.assertEqual(store.write_count, 0)

    def test_approval_binds_exact_write(self):
        result, store = self.run_agent(
            intent=Intent("write", 1, "Call Friday"), approve=True
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, FinalAnswer("updated", 1, "Lan"))
        self.assertEqual(store.get(1)["note"], "Call Friday")
        self.assertEqual(store.write_count, 1)

    def test_altered_note_fails_even_with_approval(self):
        result, store = self.run_agent(
            "altered-write", intent=Intent("write", 1, "Call Friday"), approve=True
        )
        self.assertEqual(result.code, "ACTION_MISMATCH")
        self.assertEqual(store.write_count, 0)

    def test_repeated_write_rejected_before_second_effect(self):
        result, store = self.run_agent(
            "repeated-write", intent=Intent("write", 1, "Call Friday"), approve=True
        )
        self.assertEqual(result.code, "REPEATED_WRITE")
        self.assertEqual(store.write_count, 1)

    def test_prompt_injection_cannot_register_a_new_tool(self):
        result, store = self.run_agent("prompt-injection", malicious_note=True)
        self.assertEqual(result.status, "denied")
        self.assertEqual(result.code, "UNKNOWN_TOOL")
        self.assertEqual(store.write_count, 0)
        self.assertEqual(
            [e["tool"] for e in result.events if e["event"] == "tool_denied"],
            ["unregistered"],
        )

    def test_plain_malicious_note_is_not_a_command(self):
        result, store = self.run_agent(malicious_note=True)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, FinalAnswer("read", 1, "Lan"))
        self.assertEqual(store.write_count, 0)

    def test_unregistered_tool_name_is_not_logged_raw(self):
        class LeakingToolName:
            def complete(self, messages, tools):
                return ModelTurn(tool_calls=(ToolCall(
                    "c1", "secret_token_test_value", "{}",
                ),))
        result, store = self.run_agent(model=LeakingToolName())
        self.assertEqual(result.code, "UNKNOWN_TOOL")
        self.assertEqual(store.write_count, 0)
        self.assertNotIn("secret_token_test_value", json.dumps(result.as_dict()))
        self.assertEqual(
            [e["tool"] for e in result.events if e["event"] == "tool_denied"],
            ["unregistered"],
        )

    def test_unauthorized_tool_denied(self):
        result, store = self.run_agent("unauthorized-tool")
        self.assertEqual(result.code, "UNKNOWN_TOOL")
        self.assertEqual(store.write_count, 0)

    def test_wrong_customer_target_denied(self):
        result, store = self.run_agent("wrong-target")
        self.assertEqual(result.code, "ACTION_MISMATCH")
        self.assertEqual(store.write_count, 0)

    def test_cannot_approve_other_principal_scope(self):
        result, store = self.run_agent(principal="bob")
        self.assertEqual(result.code, "SCOPE_DENIED")
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(store.write_count, 0)

    def test_premature_success_not_trusted(self):
        result, store = self.run_agent(
            "premature-success", intent=Intent("write", 1, "Call Friday"),
            approve=True,
        )
        self.assertEqual(result.code, "UNGROUNDED_FINAL")
        self.assertEqual(store.write_count, 0)

    def test_invalid_arguments_denied(self):
        result, _ = self.run_agent("invalid-arguments")
        self.assertEqual(result.code, "ACTION_MISMATCH")

    def test_model_failure_redacts_provider_exception(self):
        result, _ = self.run_agent("provider-error")
        self.assertEqual(result.code, "MODEL_REQUEST_FAILED")
        self.assertNotIn("Secret", json.dumps(result.as_dict()))

    def test_model_call_limit_enforced(self):
        result, store = self.run_agent(
            intent=Intent("write", 1, "Call Friday"), approve=True,
            limits=Limits(max_model_calls=1),
        )
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.code, "MODEL_CALL_LIMIT")
        self.assertEqual(store.write_count, 0)

    def test_tool_attempt_limit_prevents_write(self):
        result, store = self.run_agent(
            intent=Intent("write", 1, "Call Friday"), approve=True,
            limits=Limits(max_tool_attempts=1),
        )
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.code, "TOOL_ATTEMPT_LIMIT")
        self.assertEqual(store.write_count, 0)

    def test_empty_or_duplicate_tool_call_ids(self):
        class RepeatedID:
            def complete(self, messages, tools):
                return ModelTurn(tool_calls=(ToolCall("same", "get_customer", '{"customer_id":1}'),))
        result, _ = self.run_agent(model=RepeatedID())
        self.assertEqual(result.code, "INVALID_TOOL_CALL_ID")

    def test_multiple_tool_calls_rejected(self):
        class Multiple:
            def complete(self, messages, tools):
                c = ToolCall("same", "get_customer", '{"customer_id":1}')
                return ModelTurn(tool_calls=(c, c))
        result, _ = self.run_agent(model=Multiple())
        self.assertEqual(result.code, "MULTIPLE_TOOL_CALLS")

    def test_final_wrong_name_rejected(self):
        class ForgedFinal:
            def complete(self, messages, tools):
                if messages[-1]["role"] == "tool":
                    return ModelTurn(content='{"status":"read","customer_id":1,"name":"Admin"}')
                return ModelTurn(tool_calls=(ToolCall("one", "get_customer", '{"customer_id":1}'),))
        result, _ = self.run_agent(model=ForgedFinal())
        self.assertEqual(result.code, "UNGROUNDED_FINAL")

    def test_final_wrong_kind_rejected(self):
        class ForgedKind:
            def complete(self, messages, tools):
                if messages[-1]["role"] == "tool":
                    return ModelTurn(content='{"status":"updated","customer_id":1,"name":"Lan"}')
                return ModelTurn(tool_calls=(ToolCall("one", "get_customer", '{"customer_id":1}'),))
        result, _ = self.run_agent(model=ForgedKind())
        self.assertEqual(result.code, "FINAL_ACTION_MISMATCH")

    def test_bad_final_format_rejected(self):
        class BadFinal:
            def complete(self, messages, tools):
                return ModelTurn(content="Read completed!")
        result, _ = self.run_agent(model=BadFinal())
        self.assertEqual(result.code, "INVALID_FINAL")

    def test_audit_never_contains_customer_note_or_untrusted_instruction(self):
        result, _ = self.run_agent("prompt-injection", malicious_note=True)
        serialized = json.dumps(result.as_dict())
        self.assertNotIn("SYSTEM OVERRIDE", serialized)
        self.assertNotIn("Returning customer", serialized)
        self.assertEqual([e["seq"] for e in result.events],
                         list(range(1, len(result.events) + 1)))


if __name__ == "__main__":
    unittest.main()