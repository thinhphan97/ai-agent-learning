"""Lab 3 contracts: state transitions, durable checkpoints and resume safety."""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import (
    AgentRunner, Limits, MockModel, ModelTurn, Tool, ToolCall, ToolRegistry,
    build_registry,
)
from state import AgentState, CheckpointStore, Status


QUESTION = "Calculate paid order revenue for customer_id=1."


class ScriptedModel:
    def __init__(self, turns):
        self.turns = list(turns)

    def respond(self, messages, tools):
        return self.turns.pop(0)


class TrackingModel:
    def __init__(self):
        self.decisions = 0
        self.delegate = MockModel()

    def respond(self, messages, tools):
        self.decisions += 1
        return self.delegate.respond(messages, tools)


class TrackingRegistry(ToolRegistry):
    def __init__(self):
        self.base = build_registry()
        self.executions = []
        super().__init__(list(self.base.tools.values()))

    def execute(self, call, successful):
        self.executions.append(call.name)
        return super().execute(call, successful)


def tool_response(name: str, args: dict, identifier: str = "call_1") -> ModelTurn:
    return ModelTurn(tool_calls=(ToolCall(identifier, name, json.dumps(args)),))


class LifecycleTests(unittest.TestCase):
    def test_happy_path_state_and_events(self):
        state = AgentState.new(QUESTION)
        self.assertEqual(state.status, Status.CREATED)
        result = AgentRunner(MockModel(), build_registry()).execute(state)
        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(result.tool_attempts, 3)
        self.assertEqual(result.failure_reason, "")
        self.assertIn("200,000 VND", result.answer)
        self.assertEqual(
            [e["tool"] for e in result.events if e["kind"] == "tool_result"],
            ["get_customer", "get_orders", "sum_amounts"],
        )
        self.assertTrue(all(event["seq"] == i for i, event in enumerate(result.events, 1)))
        self.assertEqual(result.events[0]["kind"], "run_created")
        self.assertEqual(result.events[-1]["current"], "completed")

    def test_transition_contract_rejects_invalid_edges(self):
        state = AgentState.new(QUESTION)
        with self.assertRaisesRegex(ValueError, "Invalid transition"):
            state.transition(Status.COMPLETED)
        state.transition(Status.RUNNING)
        state.transition(Status.COMPLETED)
        with self.assertRaises(ValueError):
            state.transition(Status.RUNNING)

    def test_different_customer_and_not_found(self):
        runner = AgentRunner(MockModel(), build_registry())
        other = runner.execute(AgentState.new("Revenue for customer_id=2"))
        self.assertEqual(other.status, Status.COMPLETED)
        self.assertIn("45,000 VND", other.answer)
        missing = runner.execute(AgentState.new("Revenue for customer_id=999"))
        self.assertEqual(missing.status, Status.COMPLETED)
        self.assertIn("not found", missing.answer)
        self.assertEqual(missing.tool_attempts, 1)

    def test_bad_tool_arguments_become_observation(self):
        state = AgentState.new(QUESTION)
        result = AgentRunner(
            ScriptedModel([tool_response("get_customer", {"customer_id": "wrong"})]),
            build_registry(), Limits(max_model_calls=1)
        ).execute(state)
        self.assertEqual(result.status, Status.BUDGET_EXCEEDED)
        self.assertEqual(result.errors, 1)
        tool_content = json.loads(next(
            m["content"] for m in state.messages if m["role"] == "tool"
        ))
        self.assertEqual(tool_content["error"], "INVALID_ARGUMENTS")

    def test_early_final_is_terminal_without_evidence(self):
        state = AgentState.new(QUESTION)
        result = AgentRunner(
            ScriptedModel([ModelTurn(content="200000")]), build_registry()
        ).execute(state)
        self.assertEqual(result.status, Status.FAILED)
        self.assertEqual(result.failure_reason, "FINAL_BEFORE_EVIDENCE")

    def test_provider_failure_is_terminal(self):
        class ProviderError:
            def respond(self, messages, tools):
                raise RuntimeError("simulated provider failure")

        state = AgentState.new(QUESTION)
        result = AgentRunner(ProviderError(), build_registry()).execute(state)
        self.assertEqual(result.status, Status.FAILED)
        self.assertEqual(result.failure_reason, "MODEL_PROVIDER_ERROR")
        self.assertEqual(result.model_calls, 1)

    def test_empty_model_turn_is_failure(self):
        result = AgentRunner(
            ScriptedModel([ModelTurn()]), build_registry()
        ).execute(AgentState.new(QUESTION))
        self.assertEqual(result.failure_reason, "EMPTY_MODEL_TURN")

    def test_multi_tool_response_fails_safely(self):
        calls = (
            ToolCall("a", "get_customer", '{"customer_id":1}'),
            ToolCall("b", "get_orders", '{"customer":"Lan","status":"paid"}'),
        )
        result = AgentRunner(
            ScriptedModel([ModelTurn(tool_calls=calls)]), build_registry()
        ).execute(AgentState.new(QUESTION))
        self.assertEqual(result.failure_reason, "MULTI_TOOL_UNSUPPORTED")
        self.assertEqual(result.tool_attempts, 0)

    def test_duplicate_call_ids_fail(self):
        calls = [
            tool_response("get_customer", {"customer_id": 1}, "same"),
            tool_response("get_customer", {"customer_id": 1}, "same"),
        ]
        result = AgentRunner(
            ScriptedModel(calls), build_registry()
        ).execute(AgentState.new(QUESTION))
        self.assertEqual(result.status, Status.FAILED)
        self.assertEqual(result.failure_reason, "DUPLICATE_TOOL_CALL_ID")
        self.assertEqual(result.tool_attempts, 1)

    def test_model_budget_counts_all_attempts(self):
        result = AgentRunner(
            MockModel(), build_registry(), Limits(max_model_calls=2)
        ).execute(AgentState.new(QUESTION))
        self.assertEqual(result.status, Status.BUDGET_EXCEEDED)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(result.tool_attempts, 2)

    def test_tool_budget_preserves_calls(self):
        result = AgentRunner(
            MockModel(), build_registry(), Limits(max_tool_attempts=1)
        ).execute(AgentState.new(QUESTION))
        self.assertEqual(result.status, Status.BUDGET_EXCEEDED)
        self.assertEqual(result.tool_attempts, 1)

    def test_resume_in_wrong_state_is_invalid(self):
        completed = AgentRunner(MockModel(), build_registry()).execute(AgentState.new(QUESTION))
        with self.assertRaisesRegex(ValueError, "cannot continue"):
            AgentRunner(MockModel(), build_registry()).execute(completed)

    def test_pause_requires_positive_limit(self):
        with self.assertRaises(ValueError):
            AgentRunner(MockModel(), build_registry()).execute(
                AgentState.new(QUESTION), pause_after_tools=0
            )

    def test_limits_require_positive_values(self):
        with self.assertRaises(ValueError):
            Limits(max_errors=0)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = CheckpointStore(Path(self.tmp.name) / "nested" / "run.json")

    def test_pause_and_resume_do_not_replay_successful_tool(self):
        tools = TrackingRegistry()
        first_model = TrackingModel()
        paused = AgentRunner(first_model, tools).execute(
            AgentState.new(QUESTION), store=self.store, pause_after_tools=1
        )
        self.assertEqual(paused.status, Status.PAUSED)
        self.assertEqual(tools.executions, ["get_customer"])
        original_id = paused.run_id
        saved = self.store.load()
        self.assertEqual(saved.run_id, original_id)
        self.assertEqual(saved.model_calls, 1)
        self.assertEqual(len([m for m in saved.messages if m["role"] == "tool"]), 1)

        second_model = TrackingModel()  # new instance, as if restarting Python
        resumed = AgentRunner(second_model, tools).execute(saved, store=self.store)
        self.assertEqual(resumed.status, Status.COMPLETED)
        self.assertEqual(resumed.run_id, original_id)
        self.assertEqual(tools.executions, ["get_customer", "get_orders", "sum_amounts"])
        self.assertEqual(resumed.model_calls, 4)
        self.assertEqual(second_model.decisions, 3)
        self.assertEqual(resumed.tool_attempts, 3)
        self.assertIn("200,000 VND", resumed.answer)
        self.assertEqual(self.store.load().status, Status.COMPLETED)
        self.assertTrue(any(e["kind"] == "run_resumed" for e in resumed.events))

    def test_multiple_pauses_keep_monotonic_event_ids(self):
        tools = TrackingRegistry()
        state = AgentState.new(QUESTION)
        runner = AgentRunner(MockModel(), tools)
        for expected_calls in (1, 2, 3):
            state = runner.execute(state, store=self.store, pause_after_tools=1)
            self.assertEqual(state.status, Status.PAUSED)
            self.assertEqual(state.tool_attempts, expected_calls)
            state = self.store.load()
        state = runner.execute(state, store=self.store)
        self.assertEqual(state.status, Status.COMPLETED)
        self.assertEqual(len(tools.executions), 3)
        self.assertEqual(
            [e["seq"] for e in state.events],
            list(range(1, len(state.events) + 1))
        )

    def test_checkpoint_counters_carry_into_new_runner(self):
        state = AgentRunner(MockModel(), build_registry()).execute(
            AgentState.new(QUESTION), store=self.store, pause_after_tools=1
        )
        self.assertEqual(state.status, Status.PAUSED)
        resumed = AgentRunner(
            MockModel(), build_registry(), Limits(max_model_calls=1)
        ).execute(self.store.load(), store=self.store)
        self.assertEqual(resumed.status, Status.BUDGET_EXCEEDED)
        self.assertEqual(resumed.model_calls, 1)

    def test_json_checkpoint_contains_all_required_context(self):
        state = AgentRunner(MockModel(), build_registry()).execute(
            AgentState.new(QUESTION), store=self.store, pause_after_tools=1
        )
        raw = json.loads(self.store.path.read_text())
        self.assertEqual(raw["schema_version"], 1)
        self.assertEqual(raw["run_id"], state.run_id)
        self.assertEqual(raw["status"], "paused")
        self.assertEqual(raw["successful"]["get_customer"]["customer"]["name"], "Lan")
        self.assertEqual(raw["used_call_ids"], ["mock_call_1"])
        self.assertEqual(raw["model_calls"], 1)
        self.assertNotIn("LLM_API_KEY", raw)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.store.path.stat().st_mode), 0o600)

    def test_cancel_persisted_pause_then_refuse_resume(self):
        state = AgentRunner(MockModel(), build_registry()).execute(
            AgentState.new(QUESTION), store=self.store, pause_after_tools=1
        )
        cancelled = AgentRunner.cancel(state, self.store)
        self.assertEqual(cancelled.status, Status.CANCELLED)
        self.assertEqual(self.store.load().status, Status.CANCELLED)
        self.assertEqual(cancelled.events[-1]["reason"], "user_cancelled")
        with self.assertRaises(ValueError):
            AgentRunner(MockModel(), build_registry()).execute(self.store.load())

    def test_cancel_created_run_is_valid(self):
        state = AgentRunner.cancel(AgentState.new(QUESTION), self.store)
        self.assertEqual(state.status, Status.CANCELLED)

    def test_completed_run_cannot_be_cancelled(self):
        state = AgentRunner(MockModel(), build_registry()).execute(AgentState.new(QUESTION))
        with self.assertRaises(ValueError):
            AgentRunner.cancel(state, self.store)

    def test_bad_json_checkpoint_fails(self):
        self.store.path.parent.mkdir(parents=True)
        self.store.path.write_text("{not-json")
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            self.store.load()

    def test_version_mismatch_is_rejected(self):
        self.store.path.parent.mkdir(parents=True)
        raw = AgentState.new(QUESTION).snapshot()
        raw["schema_version"] = 999
        self.store.path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.store.load()

    def test_event_sequence_corruption_is_rejected(self):
        state = AgentState.new(QUESTION)
        raw = state.snapshot()
        raw["events"][0]["seq"] = 22
        self.store.path.parent.mkdir(parents=True)
        self.store.path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            self.store.load()

    def test_no_checkpoint_is_written_when_disabled(self):
        state = AgentRunner(MockModel(), build_registry()).execute(AgentState.new(QUESTION))
        self.assertEqual(state.status, Status.COMPLETED)
        self.assertFalse(self.store.path.exists())


if __name__ == "__main__":
    unittest.main()