import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentRunner, MockModel, ToolCall, ToolRegistry, build_registry


class AgentLoopTests(unittest.TestCase):
    def test_complete_with_two_tools(self):
        model = MockModel()
        result = AgentRunner(model, build_registry()).run("Revenue for Lan")
        self.assertEqual(result.status, "completed")
        self.assertEqual([x["action"] for x in result.trace], ["get_orders", "sum_amounts"])
        self.assertIn("200,000 VND", result.answer)
        self.assertEqual(result.model_calls, 3)

    def test_budget_stop(self):
        result = AgentRunner(MockModel(), build_registry(), max_model_calls=1).run("Revenue")
        self.assertEqual(result.status, "budget_exceeded")

    def test_invalid_tool_arguments(self):
        result = build_registry().execute(ToolCall("id", "sum_amounts", '{"amounts":[-1]}'))
        self.assertEqual(result["error"], "INVALID_ARGUMENTS")

    def test_unknown_tool(self):
        result = build_registry().execute(ToolCall("id", "not_available", "{}"))
        self.assertEqual(result["error"], "UNKNOWN_TOOL")


if __name__ == "__main__":
    unittest.main()
