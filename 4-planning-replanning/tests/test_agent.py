"""Lab 4 tests: plan contracts, evidence-gated steps, replanning, injection."""
from __future__ import annotations

import json
import os
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent
from agent import (
    Action, AgentRunner, Limits, MockModel, ResponseFaultInjector,
    ToolCall, build_registry, DEFAULT_QUESTION,
)
from planning import Plan, PlanRejected, PlanStep, PlanValidator


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.validator = PlanValidator(agent.RULES, "sum_amounts")

    def plan(self, *, steps=None):
        return self.validator.parse(json.dumps({
            "steps": steps if steps is not None else [
                {"id": "lookup", "tool": "get_customer", "objective": "Find customer", "depends_on": []},
                {"id": "orders", "tool": "get_orders", "objective": "Read orders", "depends_on": ["get_customer"]},
                {"id": "total", "tool": "sum_amounts", "objective": "Sum total", "depends_on": ["get_orders"]},
            ]
        }))

    def expect_code(self, plan, code, evidence=None):
        with self.assertRaises(PlanRejected) as raised:
            self.validator.validate(plan, evidence or {})
        self.assertEqual(raised.exception.code, code)

    def test_valid_plan(self):
        plan = self.plan()
        self.validator.validate(plan, {})
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[1].depends_on, ("get_customer",))
        self.assertEqual(plan.as_dict()["steps"][2]["tool"], "sum_amounts")

    def test_valid_plan_with_prior_evidence(self):
        steps = [
            {"id": "orders", "tool": "get_orders", "objective": "Read orders", "depends_on": ["get_customer"]},
            {"id": "total", "tool": "sum_amounts", "objective": "Sum total", "depends_on": ["get_orders"]},
        ]
        self.validator.validate(self.plan(steps=steps), {"get_customer": {"found": True}})

    def test_fenced_json_supported(self):
        raw = json.dumps(self.plan().as_dict())
        self.assertEqual(self.validator.parse("```json\n" + raw + "\n```"), self.plan())

    def test_invalid_json_rejected(self):
        with self.assertRaises(PlanRejected) as raised:
            self.validator.parse("{broken")
        self.assertEqual(raised.exception.code, "INVALID_PLAN_JSON")

    def test_wrong_object_shape(self):
        with self.assertRaises(PlanRejected) as raised:
            self.validator.parse('{"plan": []}')
        self.assertEqual(raised.exception.code, "INVALID_PLAN_SHAPE")

    def test_bad_fields(self):
        for data in [
            {"id": "", "tool": "get_customer", "objective": "x", "depends_on": []},
            {"id": "test", "tool": "get_customer", "objective": "", "depends_on": []},
            {"id": "test", "tool": "get_customer", "objective": "x", "depends_on": "get_customer"},
        ]:
            with self.subTest(data=data):
                with self.assertRaises(PlanRejected):
                    self.plan(steps=[data])

    def test_empty_plan_rejected(self):
        with self.assertRaises(PlanRejected) as raised:
            self.plan(steps=[])
        self.assertEqual(raised.exception.code, "INVALID_PLAN_LENGTH")

    def test_unknown_tool(self):
        plan = self.plan(steps=[
            {"id": "bad", "tool": "delete_all", "objective": "bad", "depends_on": []}
        ])
        self.expect_code(plan, "UNKNOWN_TOOL")

    def test_duplicate_step_id(self):
        data = list(self.plan().as_dict()["steps"])
        data[1]["id"] = data[0]["id"]
        self.expect_code(self.plan(steps=data), "DUPLICATE_STEP_ID")

    def test_duplicate_tool(self):
        data = list(self.plan().as_dict()["steps"])
        data.insert(1, {
            "id": "duplicate", "tool": "get_customer",
            "objective": "Again", "depends_on": [],
        })
        self.expect_code(self.plan(steps=data), "DUPLICATE_TOOL")

    def test_missing_declared_dependency(self):
        data = self.plan().as_dict()["steps"]
        data[1]["depends_on"] = []
        self.expect_code(self.plan(steps=data), "INVALID_DEPENDENCIES")

    def test_unreachable_step(self):
        self.expect_code(self.plan(steps=[
            {"id": "orders", "tool": "get_orders", "objective": "Too early",
             "depends_on": ["get_customer"]},
            {"id": "customer", "tool": "get_customer", "objective": "Late",
             "depends_on": []},
            {"id": "sum", "tool": "sum_amounts", "objective": "Total",
             "depends_on": ["get_orders"]},
        ]), "UNREACHABLE_STEP")

    def test_incomplete_plan(self):
        self.expect_code(self.plan(steps=[
            {"id": "lookup", "tool": "get_customer", "objective": "Lookup", "depends_on": []}
        ]), "INCOMPLETE_PLAN")

    def test_completed_tool_cannot_be_replanned(self):
        self.expect_code(self.plan(), "REDUNDANT_STEP", evidence={
            "get_customer": {"found": True}
        })


    def test_wrong_metric_plan_is_valid_at_structural_level(self):
        validator = PlanValidator(
            agent.RULES, "sum_amounts", alternative_final_tools=("count_orders",)
        )
        wrong_plan = self.plan(steps=[
            {"id": "resolve", "tool": "get_customer", "objective": "Resolve", "depends_on": []},
            {"id": "read", "tool": "get_orders", "objective": "Read", "depends_on": ["get_customer"]},
            {"id": "count", "tool": "count_orders", "objective": "Count paid orders",
             "depends_on": ["get_orders"]},
        ])
        validator.validate(wrong_plan, {})
        # The old strict plan contract still distinguishes a missing sum.
        self.expect_code(wrong_plan, "INCOMPLETE_PLAN")

    def test_structural_plan_does_not_override_goal_evaluator(self):
        from evaluation import ProgressEvaluator

        evidence = {
            "get_customer": {"customer": {"id": 1, "name": "Lan"}},
            "get_orders": {"orders": [
                {"id": 101, "amount": 120000}, {"id": 102, "amount": 80000}
            ]},
            "count_orders": {"count": 2},
        }
        evaluator = ProgressEvaluator()
        self.assertEqual(evaluator.evaluate(evidence).code, "GOAL_METRIC_MISMATCH")
        self.assertFalse(evaluator.evaluate(evidence).satisfied)
        evidence["sum_amounts"] = {"total": 200000}
        self.assertTrue(evaluator.evaluate(evidence).satisfied)
        self.assertEqual(evaluator.evaluate(evidence).code, "GOAL_SATISFIED")

    def test_inconsistent_total_is_not_accepted_as_goal(self):
        from evaluation import ProgressEvaluator
        evidence = {
            "get_customer": {"customer": {"id": 1, "name": "Lan"}},
            "get_orders": {"orders": [{"amount": 120000}]},
            "sum_amounts": {"total": 999},
        }
        evaluation = ProgressEvaluator().evaluate(evidence)
        self.assertFalse(evaluation.satisfied)
        self.assertEqual(evaluation.code, "EVIDENCE_INCONSISTENT")


class RuntimeTests(unittest.TestCase):
    def run_agent(self, model=None, *, registry=None, limits=None, question=DEFAULT_QUESTION):
        return AgentRunner(
            model or MockModel(), registry or build_registry(), limits=limits
        ).run(question)

    def test_normal_execution_and_plan_history(self):
        state = self.run_agent()
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.reason, "EVIDENCE_VERIFIED")
        self.assertEqual(state.model_calls, 4)
        self.assertEqual(state.plan_attempts, 1)
        self.assertEqual(state.tool_attempts, 3)
        self.assertEqual(state.errors, 0)
        self.assertEqual(state.replans, 0)
        self.assertEqual(len(state.plan_history), 1)
        self.assertEqual(list(state.evidence), ["get_customer", "get_orders", "sum_amounts"])
        self.assertIn("200,000 VND", state.answer)
        self.assertEqual([e["seq"] for e in state.events], list(range(1, len(state.events) + 1)))

    def test_first_plan_invalid_then_corrected(self):
        state = self.run_agent(MockModel("invalid-plan"))
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.plan_attempts, 2)
        self.assertEqual(state.replans, 1)
        self.assertEqual(state.errors, 1)
        self.assertEqual(len(state.plan_history), 1)
        self.assertEqual(next(e["code"] for e in state.events if e["event"] == "plan_rejected"),
                         "UNREACHABLE_STEP")

    def test_wrong_action_replanning(self):
        state = self.run_agent(MockModel("wrong-action"))
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.plan_attempts, 2)
        self.assertEqual(state.tool_attempts, 4)
        self.assertEqual(state.errors, 1)
        self.assertEqual(next(e["code"] for e in state.events if e["event"] == "step_failed"),
                         "ACTION_TOOL_MISMATCH")

    def test_transient_tool_failure_retries_without_replanning(self):
        state = self.run_agent(
            MockModel("tool-failure"), registry=build_registry(fail_orders_once=True)
        )
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.plan_attempts, 1)
        self.assertEqual(state.model_calls, 4)
        self.assertEqual(state.tool_attempts, 4)
        self.assertEqual(state.errors, 1)
        self.assertEqual(len(state.plan_history), 1)
        successes = [e["tool"] for e in state.events if e["event"] == "step_completed"]
        self.assertEqual(successes, ["get_customer", "get_orders", "sum_amounts"])
        self.assertEqual(
            [e["reason"] for e in state.events if e["event"] == "tool_retry"],
            ["TOOL_EXECUTION_FAILED"],
        )
        self.assertFalse(any(e["event"] == "replan_triggered" for e in state.events))

    def test_persistent_tool_failure_aborts_without_replanning(self):
        registry = build_registry()
        original = registry.execute

        def broken(call, evidence, target_customer_id):
            if call.name == "get_orders":
                return {"ok": False, "code": "TOOL_EXECUTION_FAILED"}
            return original(call, evidence, target_customer_id)

        registry.execute = broken
        state = self.run_agent(registry=registry)
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.reason, "TOOL_EXECUTION_FAILED")
        self.assertEqual(state.plan_attempts, 1)
        self.assertEqual(state.tool_attempts, 3)
        self.assertFalse(any(e["event"] == "replan_triggered" for e in state.events))

    def test_zero_retry_budget_aborts_core_failure(self):
        registry = build_registry(fail_orders_once=True)
        state = self.run_agent(registry=registry, limits=Limits(max_tool_retries=0))
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.reason, "TOOL_EXECUTION_FAILED")
        self.assertEqual(state.tool_attempts, 2)

    def test_retry_respects_total_tool_budget(self):
        state = self.run_agent(
            registry=build_registry(fail_orders_once=True),
            limits=Limits(max_tool_attempts=2),
        )
        self.assertEqual(state.status, "budget_exceeded")
        self.assertEqual(state.reason, "TOOL_ATTEMPT_LIMIT")
        self.assertEqual(state.tool_attempts, 2)

    def test_semantic_mismatch_replans_after_all_tools_succeed(self):
        state = self.run_agent(MockModel("semantic-mismatch"))
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.reason, "EVIDENCE_VERIFIED")
        self.assertEqual(state.plan_attempts, 2)
        self.assertEqual(state.replans, 1)
        self.assertEqual(state.model_calls, 6)
        self.assertEqual(state.tool_attempts, 4)
        self.assertEqual(state.errors, 0)
        self.assertEqual(state.evidence["count_orders"]["count"], 2)
        self.assertEqual(state.evidence["sum_amounts"]["total"], 200000)
        self.assertEqual(state.answer, "Paid revenue for Lan: 200,000 VND")
        self.assertEqual(
            [step["tool"] for step in state.plan_history[0]["steps"]],
            ["get_customer", "get_orders", "count_orders"],
        )
        self.assertEqual(
            [step["tool"] for step in state.plan_history[1]["steps"]],
            ["sum_amounts"],
        )
        self.assertEqual(
            [event["tool"] for event in state.events if event["event"] == "step_completed"],
            ["get_customer", "get_orders", "count_orders", "sum_amounts"],
        )
        self.assertFalse(any(event["event"] == "step_failed" for event in state.events))
        mismatch = next(e for e in state.events if e["event"] == "goal_mismatch")
        self.assertEqual(mismatch["code"], "GOAL_METRIC_MISMATCH")
        triggered = next(e for e in state.events if e["event"] == "replan_triggered")
        self.assertEqual(triggered["last_error"]["phase"], "goal_evaluation")
        self.assertEqual(
            triggered["successful_tools"], ["get_customer", "get_orders", "count_orders"]
        )
        self.assertTrue(any(
            e["event"] == "progress_evaluated" and e["code"] == "GOAL_SATISFIED"
            for e in state.events
        ))

    def test_semantic_mismatch_replan_limit_preserves_evidence(self):
        state = self.run_agent(
            MockModel("semantic-mismatch"), limits=Limits(max_plan_attempts=1)
        )
        self.assertEqual(state.status, "replan_exhausted")
        self.assertEqual(state.reason, "PLAN_ATTEMPT_LIMIT")
        self.assertEqual(state.evidence["count_orders"]["count"], 2)
        self.assertNotIn("sum_amounts", state.evidence)

    def test_replanner_cannot_repeat_completed_wrong_metric_tool(self):
        class RepeatsWrongPlan(MockModel):
            def propose_plan(self, context):
                self.plan_calls += 1
                return json.dumps({"steps": [
                    {"id": "a", "tool": "get_customer", "objective": "Resolve", "depends_on": []},
                    {"id": "b", "tool": "get_orders", "objective": "Orders", "depends_on": ["get_customer"]},
                    {"id": "c", "tool": "count_orders", "objective": "Count", "depends_on": ["get_orders"]},
                ]})
        state = self.run_agent(
            RepeatsWrongPlan(), limits=Limits(max_plan_attempts=2)
        )
        self.assertEqual(state.status, "replan_exhausted")
        self.assertEqual(state.evidence["count_orders"]["count"], 2)
        self.assertEqual(
            [event["code"] for event in state.events if event["event"] == "plan_rejected"],
            ["REDUNDANT_STEP"],
        )

    def test_real_model_wrapper_injects_only_first_wrong_plan(self):
        from agent import PlanFaultInjector
        underlying = MockModel()
        injected = PlanFaultInjector(underlying)
        with redirect_stdout(StringIO()):
            state = self.run_agent(injected)
        self.assertTrue(injected.injected)
        self.assertEqual(state.status, "completed")
        self.assertEqual(underlying.plan_calls, 2)
        self.assertEqual(underlying.action_calls, 4)
        self.assertEqual(
            [step["tool"] for step in state.plan_history[1]["steps"]], ["sum_amounts"]
        )

    def test_other_customer_uses_question_id(self):
        state = self.run_agent(question="Calculate paid revenue for customer_id=2.")
        self.assertEqual(state.status, "completed")
        self.assertIn("45,000 VND", state.answer)
        self.assertEqual(state.evidence["get_customer"]["customer"]["name"], "Minh")

    def test_missing_customer_finishes_after_lookup(self):
        state = self.run_agent(question="Calculate paid revenue for customer_id=999.")
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.tool_attempts, 1)
        self.assertEqual(list(state.evidence), ["get_customer"])
        self.assertEqual(state.answer, "Customer was not found.")

    def test_missing_request_id_is_terminal_without_model_calls(self):
        state = self.run_agent(question="Tell me the revenue")
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.reason, "MISSING_CUSTOMER_ID")
        self.assertEqual(state.model_calls, 0)

    def test_model_budget_hard_limit(self):
        state = self.run_agent(limits=Limits(max_model_calls=2))
        self.assertEqual(state.status, "budget_exceeded")
        self.assertEqual(state.model_calls, 2)

    def test_tool_budget_hard_limit(self):
        state = self.run_agent(limits=Limits(max_tool_attempts=1))
        self.assertEqual(state.status, "budget_exceeded")
        self.assertEqual(state.tool_attempts, 1)

    def test_invalid_plan_exhausts_replans(self):
        class BadPlanner(MockModel):
            def propose_plan(self, context):
                return "this is not json"

        state = self.run_agent(BadPlanner(), limits=Limits(max_plan_attempts=2, max_errors=3))
        self.assertEqual(state.status, "replan_exhausted")
        self.assertEqual(state.plan_attempts, 2)
        self.assertEqual(state.errors, 2)

    def test_error_budget(self):
        class BadPlanner(MockModel):
            def propose_plan(self, context):
                return "not JSON"

        state = self.run_agent(BadPlanner(), limits=Limits(max_errors=1))
        self.assertEqual(state.status, "recovery_exhausted")
        self.assertEqual(state.errors, 1)

    def test_provider_error_during_plan(self):
        class FailPlan(MockModel):
            def propose_plan(self, context):
                raise RuntimeError("provider unavailable")

        state = self.run_agent(FailPlan())
        self.assertEqual(state.reason, "PLANNER_PROVIDER_ERROR")
        self.assertEqual(state.model_calls, 1)

    def test_provider_error_during_execution(self):
        class FailAction(MockModel):
            def propose_action(self, context, tools):
                raise RuntimeError("provider unavailable")

        state = self.run_agent(FailAction())
        self.assertEqual(state.reason, "EXECUTOR_PROVIDER_ERROR")
        self.assertEqual(state.model_calls, 2)

    def test_missing_action_tool_call_causes_replan_and_exhaust(self):
        class EmptyAction(MockModel):
            def propose_action(self, context, tools):
                return Action(content="I am done")

        state = self.run_agent(EmptyAction(), limits=Limits(max_errors=2))
        self.assertEqual(state.status, "recovery_exhausted")
        self.assertEqual(state.errors, 2)

    def test_arguments_type_cannot_be_forged(self):
        class WrongArgs(MockModel):
            def propose_action(self, context, tools):
                act = super().propose_action(context, tools)
                return Action(tool_calls=(ToolCall(act.tool_calls[0].id, act.tool_calls[0].name,
                                                 '{"customer_id":"1"}'),))

        state = self.run_agent(WrongArgs(), limits=Limits(max_errors=2))
        self.assertEqual(state.status, "recovery_exhausted")
        self.assertEqual(state.evidence, {})

    def test_customer_id_mismatch_rejected(self):
        class WrongCustomer(MockModel):
            def propose_action(self, context, tools):
                action = super().propose_action(context, tools)
                if context["current_step"]["tool"] == "get_customer":
                    t = action.tool_calls[0]
                    return Action(tool_calls=(ToolCall(t.id, t.name, '{"customer_id":2}'),))
                return action

        state = self.run_agent(WrongCustomer(), limits=Limits(max_errors=1))
        self.assertEqual(state.status, "recovery_exhausted")
        self.assertEqual(state.evidence, {})

    def test_final_answer_uses_verified_evidence_not_model_guess(self):
        state = self.run_agent()
        self.assertEqual(state.answer, "Paid revenue for Lan: 200,000 VND")
        self.assertEqual(state.evidence["sum_amounts"]["total"], 200000)

    def test_response_injection_only_once_and_real_model_receives_error(self):
        class CaptureModel(MockModel):
            def __init__(self):
                super().__init__()
                self.contexts = []

            def propose_plan(self, context):
                self.contexts.append(context)
                return super().propose_plan(context)

        real = CaptureModel()
        injected = ResponseFaultInjector(real, "wrong-tool")
        with redirect_stdout(StringIO()):
            state = self.run_agent(injected)
        self.assertEqual(state.status, "completed")
        self.assertTrue(injected.injected)
        self.assertEqual(real.action_calls, 4)
        self.assertEqual(real.plan_calls, 2)
        self.assertEqual(real.contexts[1]["last_error"]["code"], "ACTION_TOOL_MISMATCH")

    def test_invalid_argument_injection_replans(self):
        with redirect_stdout(StringIO()):
            state = self.run_agent(ResponseFaultInjector(MockModel(), "invalid-arguments"))
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.errors, 1)
        self.assertEqual(state.replans, 1)
        self.assertEqual(next(e["code"] for e in state.events if e["event"] == "step_failed"),
                         "INVALID_ARGUMENTS")


class CliTests(unittest.TestCase):
    def invoke(self, args, *, env="1", fake_live=None):
        stdout = StringIO()
        with patch.object(sys, "argv", ["agent.py", *args]), patch.dict(
            os.environ, {"LIVE_MODE": env}
        ), patch.object(
            agent, "openai_compatible_model",
            side_effect=AssertionError("Unexpected LLM") if fake_live is None else None,
            return_value=fake_live,
        ), redirect_stdout(stdout):
            agent.main()
        return stdout.getvalue()

    def test_scenario_overrides_exported_live(self):
        out = self.invoke(["--scenario", "tool-failure"], env="1")
        self.assertIn("Mode: mock | scenario=tool-failure", out)
        self.assertIn("Tool retry: get_orders - TOOL_EXECUTION_FAILED", out)
        self.assertIn("Status: completed", out)

    def test_real_mode_can_inject_llm_response(self):
        out = self.invoke(["--live", "--inject-llm-action", "wrong-tool"],
                          env="0", fake_live=MockModel())
        self.assertIn("Injected first LLM action: wrong-tool", out)
        self.assertIn("step_failed : ACTION_TOOL_MISMATCH", out)
        self.assertIn("Status: completed", out)

    def test_real_mode_can_inject_tool_failure(self):
        out = self.invoke(["--live", "--inject-tool-failure"],
                          env="0", fake_live=MockModel())
        self.assertIn("Tool retry: get_orders - TOOL_EXECUTION_FAILED", out)
        self.assertIn("Status: completed", out)

    def test_live_wrong_metric_plan_injection_recovers(self):
        output = self.invoke(
            ["--live", "--inject-plan-fault", "wrong-metric"],
            env="0", fake_live=MockModel(),
        )
        self.assertIn("Injected first LLM plan: wrong-metric", output)
        self.assertIn("Step completed: count_orders", output)
        self.assertIn("Goal mismatch: GOAL_METRIC_MISMATCH", output)
        self.assertIn("Replan triggered: GOAL_METRIC_MISMATCH", output)
        self.assertIn("Plan revision 2 : sum_amounts", output)
        self.assertIn("Status: completed", output)

    def test_live_plan_fault_rejects_mock_mode(self):
        with (
            patch.object(sys, "argv", [
                "agent.py", "--mock", "--inject-plan-fault", "wrong-metric"
            ]),
            patch.dict(os.environ, {"LIVE_MODE": "0"}),
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            agent.main()
        self.assertEqual(raised.exception.code, 2)

    def test_multiple_fault_injections_are_mutually_exclusive(self):
        with (
            patch.object(sys, "argv", [
                "agent.py", "--live", "--inject-plan-fault", "wrong-metric",
                "--inject-tool-failure",
            ]),
            patch.dict(os.environ, {"LIVE_MODE": "0"}),
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            agent.main()
        self.assertEqual(raised.exception.code, 2)

    def test_default_exported_live_mode(self):
        out = self.invoke([], env="1", fake_live=MockModel())
        self.assertIn("Mode: live", out)
        self.assertIn("Status: completed", out)

    def test_invalid_injection_without_live_is_rejected(self):
        with patch.object(sys, "argv", ["agent.py", "--mock", "--inject-llm-action", "wrong-tool"]), \
             patch.dict(os.environ, {"LIVE_MODE": "0"}), redirect_stderr(StringIO()), \
             self.assertRaises(SystemExit):
            agent.main()

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            Limits(max_plan_attempts=0)


if __name__ == "__main__":
    unittest.main()