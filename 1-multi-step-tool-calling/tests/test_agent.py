import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentRunner, MockModel, ModelTurn, ToolCall, build_registry, make_runner


class MultiStepTests(unittest.TestCase):
    def test_happy_path_order_and_final(self):
        model = MockModel(1)
        result = make_runner(model).run("Revenue for customer_id=1")
        self.assertEqual(result.status, "completed")
        self.assertEqual([t["action"] for t in result.trace], ["get_customer", "get_orders", "sum_amounts"])
        self.assertEqual(result.model_calls, 4)
        self.assertIn("200,000 VND", result.answer)
        self.assertEqual(result.trace[1]["observation"]["data"]["orders"][0]["customer"], "Lan")
        self.assertEqual(result.trace[2]["observation"]["data"]["total"], 200000)

    def test_different_customer_id_changes_data(self):
        result = make_runner(MockModel(2)).run("Revenue for customer_id=2")
        self.assertEqual(result.status, "completed")
        self.assertIn("45,000 VND", result.answer)
        self.assertEqual(result.trace[1]["observation"]["data"]["orders"][0]["customer"], "Minh")

    def test_unknown_customer_cannot_prematurely_finish(self):
        result = make_runner(MockModel(999)).run("Revenue for customer_id=999")
        self.assertEqual(result.status, "policy_violation")

    def test_wrong_order_rejected_before_execution(self):
        class WrongModel:
            def respond(self, messages, tools):
                return ModelTurn(tool_calls=(ToolCall("x", "get_orders", '{"customer":"Lan","status":"paid"}'),))
        result = make_runner(WrongModel()).run("Revenue")
        self.assertEqual(result.status, "policy_violation")
        self.assertEqual(result.trace, [])

    def test_early_finish_rejected(self):
        class EarlyModel:
            def respond(self, messages, tools):
                return ModelTurn(content="200,000 VND")
        self.assertEqual(make_runner(EarlyModel()).run("Revenue").status, "policy_violation")

    def test_batch_rejected(self):
        class BatchModel:
            def respond(self, messages, tools):
                return ModelTurn(tool_calls=(ToolCall("x", "get_customer", '{"customer_id":1}'), ToolCall("y", "get_orders", '{"customer":"Lan","status":"paid"}')))
        self.assertEqual(make_runner(BatchModel()).run("Revenue").status, "policy_violation")

    def test_invalid_arguments(self):
        result = build_registry().execute(ToolCall("x", "get_customer", '{"customer_id":true}'))
        self.assertEqual(result["error"], "INVALID_ARGUMENTS")

    def test_budget(self):
        result = AgentRunner(MockModel(1), build_registry(), max_model_calls=1, required_order=("get_customer","get_orders","sum_amounts")).run("Revenue")
        self.assertEqual(result.status, "budget_exceeded")

    def test_duplicate_call_id_rejected(self):
        class DuplicateModel:
            def respond(self, messages, tools):
                return ModelTurn(tool_calls=(ToolCall("same", "get_customer", '{"customer_id":1}'), ToolCall("same", "get_customer", '{"customer_id":1}')))
        self.assertEqual(AgentRunner(DuplicateModel(), build_registry()).run("Revenue").status, "protocol_error")


if __name__ == "__main__":
    unittest.main()
