"""Offline tests exercise real runtime decisions without paid LLM calls."""
from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent

from agent import (
    AgentRunner, DependencyPolicy, MockModel, ModelTurn, RecoveryPolicy,
    ResponseFaultInjector,
    ToolCall, ToolRegistry, build_registry,
)


QUESTION = "Calculate paid order revenue for customer_id=1."


class ScriptedModel:
    def __init__(self, turns, fallback=None):
        self.turns = list(turns)
        self.fallback = fallback or MockModel()

    def respond(self, messages, tools):
        if self.turns:
            return self.turns.pop(0)
        return self.fallback.respond(messages, tools)


def call(call_id, name, args):
    raw = args if isinstance(args, str) else json.dumps(args)
    return ModelTurn(tool_calls=(ToolCall(call_id, name, raw),))


def observed(result):
    return [event["observation"] for event in result.trace if event["event"] == "tool"]


def actions(result):
    return [event["action"] for event in result.trace if event["event"] == "tool"]


class RecoveryTests(unittest.TestCase):
    def run_agent(self, model=None, *, registry=None, recovery=None, question=QUESTION):
        return AgentRunner(
            model or MockModel(), registry or build_registry(), recovery=recovery
        ).run(question)

    def test_normal_sequence_completes(self):
        result = self.run_agent()
        self.assertEqual(result.status, "completed")
        self.assertEqual(actions(result), ["get_customer", "get_orders", "sum_amounts"])
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(result.tool_attempts, 3)
        self.assertEqual(result.errors, 0)
        self.assertIn("200,000 VND", result.answer)

    def test_customer_id_from_question_is_not_hardcoded(self):
        result = self.run_agent(question="Calculate paid order revenue for customer_id=2.")
        self.assertEqual(result.status, "completed")
        self.assertEqual(observed(result)[0]["data"]["customer"]["name"], "Minh")
        self.assertEqual(observed(result)[-1]["data"]["total"], 45000)
        self.assertIn("45,000 VND", result.answer)

    def test_wrong_order_is_rejected_then_recovered(self):
        result = self.run_agent(MockModel("wrong-order"))
        self.assertEqual(result.status, "completed")
        self.assertEqual(actions(result), ["get_orders", "get_customer", "get_orders", "sum_amounts"])
        self.assertEqual(observed(result)[0]["error"]["code"], "PREREQUISITE_MISSING")
        self.assertEqual(result.errors, 1)
        self.assertEqual(result.model_calls, 5)

    def test_invalid_arguments_recovered(self):
        result = self.run_agent(MockModel("invalid-arguments"))
        self.assertEqual(result.status, "completed")
        self.assertEqual(observed(result)[0]["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(actions(result)[:2], ["get_customer", "get_customer"])
        self.assertEqual(result.errors, 1)

    def test_transient_failure_recovered_by_agent_retry(self):
        result = self.run_agent(MockModel("tool-failure"), registry=build_registry(fail_orders_once=True))
        self.assertEqual(result.status, "completed")
        self.assertEqual(actions(result).count("get_orders"), 2)
        self.assertEqual(observed(result)[1]["error"]["code"], "TOOL_EXECUTION_FAILED")
        self.assertEqual(result.errors, 1)

    def test_early_final_is_rejected_then_model_acts(self):
        result = self.run_agent(MockModel("early-final"))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.trace[0]["event"], "final_rejected")
        self.assertEqual(result.trace[0]["reason"], "FINAL_BEFORE_COMPLETION")
        self.assertEqual(result.errors, 1)

    def test_missing_customer_can_finish_without_order_calls(self):
        result = self.run_agent(question="Calculate paid order revenue for customer_id=999.")
        self.assertEqual(result.status, "completed")
        self.assertEqual(actions(result), ["get_customer"])
        self.assertIn("not found", result.answer)

    def test_unknown_tool_rejected_then_recovery(self):
        result = self.run_agent(ScriptedModel([call("bad", "wipe_database", {})]))
        self.assertEqual(result.status, "completed")
        self.assertEqual(observed(result)[0]["error"]["code"], "UNKNOWN_TOOL")

    def test_non_json_arguments_rejected_then_recovery(self):
        result = self.run_agent(ScriptedModel([call("bad", "get_customer", "{")]))
        self.assertEqual(result.status, "completed")
        self.assertEqual(observed(result)[0]["error"]["code"], "INVALID_ARGUMENTS")

    def test_wrong_customer_name_blocked_after_lookup(self):
        first = call("a", "get_customer", {"customer_id": 1})
        wrong = call("b", "get_orders", {"customer": "Minh", "status": "paid"})
        result = self.run_agent(ScriptedModel([first, wrong]))
        self.assertEqual(result.status, "completed")
        self.assertEqual(observed(result)[1]["error"]["code"], "DATA_MISMATCH")

    def test_wrong_amounts_blocked_before_summing(self):
        script = [
            call("a", "get_customer", {"customer_id": 1}),
            call("b", "get_orders", {"customer": "Lan", "status": "paid"}),
            call("c", "sum_amounts", {"amounts": [9]}),
        ]
        result = self.run_agent(ScriptedModel(script))
        self.assertEqual(result.status, "completed")
        self.assertEqual(observed(result)[2]["error"]["code"], "DATA_MISMATCH")
        self.assertEqual(observed(result)[-1]["data"]["total"], 200000)

    def test_repeated_errors_hit_recovery_limit(self):
        class PersistentWrongOrder:
            def respond(self, messages, tools):
                return call(f"call_{len(messages)}", "get_orders", {"customer": "Lan", "status": "paid"})
        result = self.run_agent(PersistentWrongOrder(), recovery=RecoveryPolicy(max_errors=2))
        self.assertEqual(result.status, "recovery_exhausted")
        self.assertEqual(result.errors, 2)
        self.assertEqual(result.tool_attempts, 2)

    def test_repeated_early_final_hits_recovery_limit(self):
        class PersistentFinal:
            def respond(self, messages, tools):
                return ModelTurn(content="200,000 VND")
        result = self.run_agent(PersistentFinal(), recovery=RecoveryPolicy(max_errors=2))
        self.assertEqual(result.status, "recovery_exhausted")
        self.assertEqual(result.tool_attempts, 0)

    def test_model_call_budget(self):
        result = self.run_agent(
            recovery=RecoveryPolicy(max_model_calls=2, max_tool_attempts=10)
        )
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.model_calls, 2)

    def test_tool_attempt_budget(self):
        result = self.run_agent(
            recovery=RecoveryPolicy(max_tool_attempts=1)
        )
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.tool_attempts, 1)

    def test_duplicate_ids_are_not_accepted(self):
        class ReusingIDs:
            def respond(self, messages, tools):
                return call("same", "get_customer", {"customer_id": 1})
        result = self.run_agent(ReusingIDs())
        self.assertEqual(result.status, "protocol_error")
        self.assertEqual(result.tool_attempts, 1)

    def test_batch_is_rejected_without_executing_tools(self):
        batch = ModelTurn(tool_calls=(
            ToolCall("a", "get_customer", '{"customer_id": 1}'),
            ToolCall("b", "get_orders", '{"customer":"Lan","status":"paid"}'),
        ))
        result = self.run_agent(ScriptedModel([batch]))
        self.assertEqual(result.status, "completed")
        self.assertEqual([x["error"]["code"] for x in observed(result)[:2]],
                         ["MULTIPLE_TOOL_CALLS_UNSUPPORTED"] * 2)
        self.assertEqual(result.trace[2]["action"], "get_customer")

    def test_tool_result_correlation_ids_unique(self):
        result = self.run_agent(MockModel("wrong-order"))
        ids = [e["tool_call_id"] for e in result.trace if e["event"] == "tool"]
        self.assertEqual(len(ids), len(set(ids)))

    def test_empty_model_turn_can_recover(self):
        result = self.run_agent(ScriptedModel([ModelTurn()]))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.trace[0]["event"], "empty_turn")
        self.assertEqual(result.trace[0]["reason"], "EMPTY_MODEL_TURN")
        self.assertEqual(result.errors, 1)

    def test_repeated_empty_model_turns_exhaust_recovery(self):
        class PersistentEmpty:
            def respond(self, messages, tools):
                return ModelTurn()

        result = self.run_agent(PersistentEmpty(), recovery=RecoveryPolicy(max_errors=2))
        self.assertEqual(result.status, "recovery_exhausted")
        self.assertEqual(result.errors, 2)
        self.assertEqual(result.tool_attempts, 0)

    def test_inject_wrong_order_into_one_llm_response_then_recover(self):
        underlying = MockModel()
        injector = ResponseFaultInjector(underlying, "wrong-order")
        result = self.run_agent(injector)
        self.assertTrue(injector.injected)
        self.assertEqual(underlying.calls, result.model_calls)
        self.assertEqual(result.status, "completed")
        self.assertEqual(actions(result), [
            "get_orders", "get_customer", "get_orders", "sum_amounts"
        ])
        self.assertEqual(observed(result)[0]["error"]["code"], "PREREQUISITE_MISSING")
        self.assertEqual(result.errors, 1)

    def test_inject_invalid_arguments_then_real_model_turns_recover(self):
        underlying = MockModel()
        result = self.run_agent(ResponseFaultInjector(underlying, "invalid-arguments"))
        self.assertEqual(result.status, "completed")
        self.assertEqual(observed(result)[0]["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(result.errors, 1)
        self.assertEqual(underlying.calls, result.model_calls)

    def test_inject_early_final_then_continue(self):
        underlying = MockModel()
        result = self.run_agent(ResponseFaultInjector(underlying, "early-final"))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.trace[0]["reason"], "FINAL_BEFORE_COMPLETION")
        self.assertEqual(result.errors, 1)
        self.assertEqual(underlying.calls, result.model_calls)

    def test_injector_does_not_mutate_model_or_reinject(self):
        class CapturingModel:
            def __init__(self):
                self.delegate = MockModel()
                self.message_snapshots = []

            def respond(self, messages, tools):
                self.message_snapshots.append(list(messages))
                return self.delegate.respond(messages, tools)

        real = CapturingModel()
        result = self.run_agent(ResponseFaultInjector(real, "wrong-order"))
        self.assertEqual(result.status, "completed")
        # Next request to the underlying model observes the injected tool error.
        tool_messages = [m for m in real.message_snapshots[1] if m["role"] == "tool"]
        self.assertEqual(json.loads(tool_messages[-1]["content"])["error"]["code"],
                         "PREREQUISITE_MISSING")
        self.assertEqual(real.delegate.calls, result.model_calls)

    def test_provider_failure_is_terminal(self):
        class FailingProvider:
            def respond(self, messages, tools):
                raise RuntimeError("Simulated network failure")
        result = self.run_agent(FailingProvider())
        self.assertEqual(result.status, "provider_error")
        self.assertEqual(result.tool_attempts, 0)

    def test_budgets_must_be_positive(self):
        with self.assertRaises(ValueError):
            RecoveryPolicy(max_errors=0)


class CliModeTests(unittest.TestCase):
    def invoke(self, args, *, env="1", live_model=None):
        output = StringIO()
        with (
            patch.dict("os.environ", {"LIVE_MODE": env}),
            patch.object(sys, "argv", ["agent.py", *args]),
            patch.object(
                agent, "openai_compatible_model",
                side_effect=AssertionError("Unexpected live LLM call")
                if live_model is None else None,
                return_value=live_model,
            ),
            redirect_stdout(output),
        ):
            agent.main()
        return output.getvalue()

    def test_explicit_scenarios_override_exported_live_mode(self):
        cases = {
            "wrong-order": "PREREQUISITE_MISSING",
            "invalid-arguments": "INVALID_ARGUMENTS",
            "tool-failure": "TOOL_EXECUTION_FAILED",
        }
        for scenario, error_code in cases.items():
            with self.subTest(scenario=scenario):
                output = self.invoke(["--scenario", scenario], env="1")
                self.assertIn(f"Mode: mock (scenario: {scenario})", output)
                self.assertIn(error_code, output)
                self.assertIn("Recoverable errors: 1", output)
                self.assertIn("Status: completed", output)

    def test_explicit_live_can_inject_one_transient_tool_failure(self):
        output = self.invoke(
            ["--live", "--inject-tool-failure"], env="0", live_model=MockModel()
        )
        self.assertIn("Mode: live (injected get_orders failure)", output)
        self.assertIn("TOOL_EXECUTION_FAILED", output)
        self.assertIn("Recoverable errors: 1", output)
        self.assertIn("Status: completed", output)

    def test_cli_live_llm_response_injection(self):
        for scenario, marker in [
            ("wrong-order", "PREREQUISITE_MISSING"),
            ("invalid-arguments", "INVALID_ARGUMENTS"),
            ("early-final", "FINAL_BEFORE_COMPLETION"),
        ]:
            with self.subTest(scenario=scenario):
                output = self.invoke(
                    ["--live", "--inject-llm-response", scenario],
                    env="0", live_model=MockModel(),
                )
                self.assertIn(f"Mode: live (injected LLM response: {scenario})", output)
                self.assertIn("Injected LLM response:", output)
                self.assertIn(marker, output)
                self.assertIn("Recoverable errors: 1", output)
                self.assertIn("Status: completed", output)

    def test_llm_response_injection_requires_live_mode(self):
        with (
            patch.dict("os.environ", {"LIVE_MODE": "0"}),
            patch.object(sys, "argv", ["agent.py", "--inject-llm-response", "wrong-order"]),
            redirect_stdout(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            agent.main()
        self.assertEqual(raised.exception.code, 2)

    def test_two_injection_types_are_exclusive(self):
        with (
            patch.dict("os.environ", {"LIVE_MODE": "1"}),
            patch.object(
                sys, "argv",
                ["agent.py", "--live", "--inject-tool-failure", "--inject-llm-response", "wrong-order"],
            ),
            redirect_stdout(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            agent.main()
        self.assertEqual(raised.exception.code, 2)

    def test_exported_live_mode_used_if_no_explicit_scenario(self):
        output = self.invoke([], env="1", live_model=MockModel())
        self.assertIn("Mode: live", output)
        self.assertIn("Status: completed", output)

    def test_live_injection_requires_live_mode(self):
        with (
            patch.dict("os.environ", {"LIVE_MODE": "1"}),
            patch.object(sys, "argv", ["agent.py", "--scenario", "wrong-order", "--inject-tool-failure"]),
            redirect_stdout(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            agent.main()
        self.assertEqual(raised.exception.code, 2)

    def test_live_and_scenario_cannot_be_combined(self):
        with (
            patch.object(sys, "argv", ["agent.py", "--live", "--scenario", "wrong-order"]),
            redirect_stdout(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            agent.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()