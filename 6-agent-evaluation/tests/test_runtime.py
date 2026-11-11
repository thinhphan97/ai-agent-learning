"""Agent runtime and grounded tool execution contracts."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Answer, Limits, ModelTurn, ToolCall
from model import MockModel, request_kind
from runtime import AgentRunner
from tools import ToolRegistry


class RuntimeTests(unittest.TestCase):
    def run_case(self, question, scenario="normal", limits=None):
        return AgentRunner(MockModel(scenario), limits=limits).run(question)

    def test_paid_revenue(self):
        result = self.run_case("What is paid order revenue for customer_id=1?")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, Answer("answered", 200000, "VND"))
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(result.tool_attempts, 3)
        self.assertEqual(list(result.evidence), [
            "get_customer", "get_orders", "sum_amounts",
        ])
        self.assertEqual([event["seq"] for event in result.events],
                         list(range(1, len(result.events) + 1)))

    def test_paid_order_count(self):
        result = self.run_case("How many paid orders does customer_id=1 have?")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, Answer("answered", 2, "orders"))
        self.assertEqual(list(result.evidence), [
            "get_customer", "get_orders", "count_orders",
        ])

    def test_zero_is_a_valid_answer_not_no_data(self):
        rev = self.run_case("Calculate paid order revenue for customer_id=3.")
        count = self.run_case("Count paid orders for customer_id=3.")
        self.assertEqual(rev.answer, Answer("answered", 0, "VND"))
        self.assertEqual(count.answer, Answer("answered", 0, "orders"))

    def test_unknown_customer_is_no_data(self):
        result = self.run_case("What is paid revenue for customer_id=999?")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, Answer("no_data", None, None))
        self.assertEqual(result.tool_attempts, 1)

    def test_forecast_abstains_without_tools(self):
        result = self.run_case("Predict tomorrow's paid revenue for customer_id=1.")
        self.assertEqual(result.answer, Answer("abstain", None, None))
        self.assertEqual(result.tool_attempts, 0)

    def test_missing_id_abstains_without_tools(self):
        result = self.run_case("How much paid revenue did the customer generate?")
        self.assertEqual(result.answer, Answer("abstain", None, None))
        self.assertEqual(result.evidence, {})

    def test_wrong_metric_can_complete_and_be_wrong(self):
        result = self.run_case(
            "What is paid order revenue for customer_id=1?", "wrong-metric"
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, Answer("answered", 2, "orders"))
        self.assertIn("count_orders", result.evidence)

    def test_hallucinated_final_can_complete_without_reaching_correct_result(self):
        result = self.run_case(
            "What is paid order revenue for customer_id=1?", "hallucinated-final"
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, Answer("answered", 200001, "VND"))

    def test_premature_final_is_completed_not_verified(self):
        result = self.run_case(
            "What is paid order revenue for customer_id=1?", "premature-final"
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.evidence, {})
        self.assertEqual(result.model_calls, 1)

    def test_unsupported_tool_rejected(self):
        result = self.run_case(
            "What is paid order revenue for customer_id=1?", "unsupported-tool"
        )
        self.assertEqual(result.status, "tool_error")
        self.assertEqual(result.error_code, "UNKNOWN_TOOL")
        self.assertEqual(result.tool_attempts, 1)
        self.assertEqual(result.evidence, {})

    def test_repeated_tool_rejected(self):
        result = self.run_case(
            "What is paid order revenue for customer_id=1?", "loop-tool"
        )
        self.assertEqual(result.status, "tool_error")
        self.assertEqual(result.error_code, "REPEATED_TOOL")
        self.assertEqual(result.model_calls, 2)

    def test_malformed_final_rejected(self):
        result = self.run_case(
            "What is paid order revenue for customer_id=1?", "invalid-final"
        )
        self.assertEqual(result.status, "protocol_error")
        self.assertEqual(result.error_code, "INVALID_FINAL_ANSWER")

    def test_provider_error_is_masked(self):
        result = self.run_case(
            "What is paid order revenue for customer_id=1?", "provider-error"
        )
        self.assertEqual(result.status, "provider_error")
        self.assertEqual(result.error_code, "MODEL_REQUEST_FAILED")

    def test_model_call_budget(self):
        result = self.run_case(
            "What is paid order revenue for customer_id=1?",
            limits=Limits(max_model_calls=2),
        )
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.error_code, "MODEL_CALL_LIMIT")
        self.assertEqual(result.model_calls, 2)

    def test_tool_call_budget(self):
        result = self.run_case(
            "What is paid order revenue for customer_id=1?",
            limits=Limits(max_tool_attempts=1),
        )
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.error_code, "TOOL_ATTEMPT_LIMIT")

    def test_duplicate_call_id_rejected(self):
        class DuplicateModel:
            def complete(self, messages, tools):
                return ModelTurn(tool_calls=(ToolCall(
                    "duplicate", "get_customer", '{"customer_id":1}'
                ),))
        result = AgentRunner(DuplicateModel()).run(
            "What is paid order revenue for customer_id=1?"
        )
        self.assertEqual(result.status, "protocol_error")
        self.assertEqual(result.error_code, "INVALID_TOOL_CALL_ID")

    def test_multi_tool_call_rejected(self):
        class MultiModel:
            def complete(self, messages, tools):
                call = ToolCall("id1", "get_customer", '{"customer_id":1}')
                return ModelTurn(tool_calls=(call, call))
        result = AgentRunner(MultiModel()).run(
            "What is paid order revenue for customer_id=1?"
        )
        self.assertEqual(result.error_code, "MULTIPLE_TOOL_CALLS")

    def test_empty_and_oversize_question_rejected_before_model(self):
        runner = AgentRunner(MockModel())
        for question in ("", " " * 6, "a" * 1001):
            with self.subTest(question=question[:30]), self.assertRaises(ValueError):
                runner.run(question)

    def test_invalid_limits(self):
        for limits in (
            {"max_model_calls": 0}, {"max_tool_attempts": -1},
            {"max_model_calls": True},
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                Limits(**limits)

    def test_parse_strict_answer_contract(self):
        good = [
            ('{"status":"answered","value":0,"unit":"VND"}', Answer("answered", 0, "VND")),
            ('{"status":"abstain","value":null,"unit":null}', Answer("abstain", None, None)),
            ('{"status":"no_data","value":null,"unit":null}', Answer("no_data", None, None)),
        ]
        for raw, expected in good:
            self.assertEqual(Answer.parse(raw), expected)
        bad = [
            None, "{bad}", "{}", "[]",
            '{"status":"answered","value":true,"unit":"VND"}',
            '{"status":"answered","value":"0","unit":"VND"}',
            '{"status":"answered","value":0,"unit":"USD"}',
            '{"status":"abstain","value":42,"unit":null}',
            '{"status":"complete","value":2,"unit":"orders"}',
            '{"status":"answered","value":1,"unit":"VND","extra":0}',
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                Answer.parse(raw)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.evidence = {}

    def call(self, name, arguments, target=1):
        result = self.registry.execute(
            ToolCall("id", name, json.dumps(arguments)), self.evidence, target
        )
        if result["ok"]:
            self.evidence[name] = result["data"]
        return result

    def test_requires_ordering(self):
        self.assertEqual(
            self.call("get_orders", {"customer": "Lan", "status": "paid"})["code"],
            "PREREQUISITE_MISSING",
        )
        self.assertTrue(self.call("get_customer", {"customer_id": 1})["ok"])
        self.assertEqual(
            self.call("sum_amounts", {"amounts": [120000, 80000]})["code"],
            "PREREQUISITE_MISSING",
        )

    def test_argument_grounding(self):
        self.call("get_customer", {"customer_id": 1})
        self.assertEqual(
            self.call("get_orders", {"customer": "Minh", "status": "paid"})["code"],
            "INVALID_ARGUMENTS",
        )
        self.call("get_orders", {"customer": "Lan", "status": "paid"})
        self.assertEqual(
            self.call("sum_amounts", {"amounts": [250000]})["code"],
            "INVALID_ARGUMENTS",
        )
        self.assertEqual(
            self.call("count_orders", {"order_ids": [103]})["code"],
            "INVALID_ARGUMENTS",
        )

    def test_unknown_tool_never_executes(self):
        self.assertEqual(
            self.call("delete_orders", {"customer_id": 1})["code"], "UNKNOWN_TOOL"
        )
        self.assertEqual(self.evidence, {})

    def test_wrong_target_id_rejected(self):
        self.assertEqual(
            self.call("get_customer", {"customer_id": 2}, target=1)["code"],
            "DATA_MISMATCH",
        )

    def test_repeated_tool_not_allowed(self):
        self.call("get_customer", {"customer_id": 1})
        self.assertEqual(
            self.call("get_customer", {"customer_id": 1})["code"],
            "REPEATED_TOOL",
        )

    def test_paid_only_excludes_pending(self):
        self.call("get_customer", {"customer_id": 1})
        result = self.call("get_orders", {"customer": "Lan", "status": "paid"})
        self.assertEqual([o["id"] for o in result["data"]["orders"]], [101, 102])
        self.assertEqual(self.call("sum_amounts", {"amounts": [120000, 80000]})["data"],
                         {"total": 200000})

    def test_invalid_json_and_bool_id(self):
        self.assertEqual(
            self.registry.execute(
                ToolCall("bad", "get_customer", "{broken"), {}, 1
            )["code"], "INVALID_ARGUMENTS",
        )
        self.assertEqual(
            self.call("get_customer", {"customer_id": True})["code"],
            "INVALID_ARGUMENTS",
        )

    def test_forecast_is_unsupported(self):
        self.assertEqual(request_kind("Predict tomorrow's revenue for customer_id=1."),
                         ("unsupported", None))


if __name__ == "__main__":
    unittest.main()
