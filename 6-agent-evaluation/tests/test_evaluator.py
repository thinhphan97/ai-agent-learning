"""Independent answer-quality evaluation tests, including adversarial completions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases import CASES, EvalCase, select_cases
from contracts import Answer, Episode, Limits, ModelTurn
from evaluator import Evaluator
from harness import SuiteRunner, aggregate
from model import MockModel
from runtime import AgentRunner


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.eval = Evaluator()
        self.case = select_cases("revenue_lan")[0]

    def score(self, scenario="normal", case=None):
        case = case or self.case
        trace = AgentRunner(MockModel(scenario)).run(case.question)
        return self.eval.score(case, trace, Limits())

    def test_normal_exact_grounded_success(self):
        score = self.score()
        self.assertTrue(score.completed)
        self.assertTrue(score.answer_exact)
        self.assertTrue(score.evidence_grounded)
        self.assertTrue(score.tool_policy_ok)
        self.assertTrue(score.budget_ok)
        self.assertTrue(score.task_success)
        self.assertEqual(score.failure_reasons, ())

    def test_wrong_metric_is_grounded_but_not_correct(self):
        score = self.score("wrong-metric")
        self.assertTrue(score.completed)
        self.assertTrue(score.evidence_grounded)
        self.assertFalse(score.answer_exact)
        self.assertFalse(score.task_success)
        self.assertIn("ANSWER_MISMATCH", score.failure_reasons)

    def test_hallucinated_final_not_grounded_or_correct(self):
        score = self.score("hallucinated-final")
        self.assertTrue(score.completed)
        self.assertFalse(score.answer_exact)
        self.assertFalse(score.evidence_grounded)
        self.assertEqual(score.status, "completed")

    def test_exact_answer_without_provenance_is_not_a_success(self):
        score = self.score("premature-final")
        self.assertTrue(score.answer_exact)
        self.assertTrue(score.completed)
        self.assertFalse(score.evidence_grounded)
        self.assertFalse(score.task_success)
        self.assertEqual(score.failure_reasons, ("UNGROUNDED_ANSWER",))

    def test_unknown_tool_policy_violation(self):
        score = self.score("unsupported-tool")
        self.assertFalse(score.completed)
        self.assertFalse(score.tool_policy_ok)
        self.assertIn("TOOL_POLICY_VIOLATION", score.failure_reasons)

    def test_no_data_is_not_equivalent_to_zero(self):
        missing = select_cases("unknown_customer")[0]
        score = self.score(case=missing)
        self.assertTrue(score.task_success)
        self.assertEqual(score.observed, Answer("no_data", None, None))
        zero = select_cases("revenue_nhi_zero")[0]
        score2 = self.score(case=zero)
        self.assertTrue(score2.task_success)
        self.assertEqual(score2.observed, Answer("answered", 0, "VND"))

    def test_unsupported_requests_should_abstain(self):
        for case_id in ("unsupported_forecast", "missing_identifier"):
            with self.subTest(case_id=case_id):
                self.assertTrue(self.score(case=select_cases(case_id)[0]).task_success)

    def test_false_no_data_is_not_grounded(self):
        class FalseNoData:
            def complete(self, messages, tools):
                return ModelTurn(
                    content='{"status":"no_data","value":null,"unit":null}'
                )
        trace = AgentRunner(FalseNoData()).run(self.case.question)
        score = self.eval.score(self.case, trace, Limits())
        self.assertFalse(score.evidence_grounded)
        self.assertFalse(score.answer_exact)

    def test_false_abstain_on_supported_query_is_not_grounded(self):
        class FalseAbstain:
            def complete(self, messages, tools):
                return ModelTurn(
                    content='{"status":"abstain","value":null,"unit":null}'
                )
        trace = AgentRunner(FalseAbstain()).run(self.case.question)
        score = self.eval.score(self.case, trace, Limits())
        self.assertFalse(score.task_success)
        self.assertFalse(score.evidence_grounded)

    def test_evidence_inconsistency_detected(self):
        case = self.case
        trace = AgentRunner(MockModel()).run(case.question)
        trace.evidence["get_orders"]["orders"][0]["amount"] = 999
        score = self.eval.score(case, trace, Limits())
        self.assertFalse(score.evidence_grounded)
        self.assertFalse(score.task_success)

    def test_suite_uses_only_question_not_expected_label(self):
        class SpyModel:
            def __init__(self):
                self.messages = []

            def complete(self, messages, tools):
                self.messages.append(messages)
                # Has no access to expected value or EvalCase.
                return ModelTurn(
                    content='{"status":"abstain","value":null,"unit":null}'
                )

        spies = []

        def factory():
            model = SpyModel()
            spies.append(model)
            return model

        cases = select_cases("revenue_lan")
        results = SuiteRunner(factory).run(cases)
        self.assertEqual(len(spies), 1)
        user_msgs = [
            m["content"] for m in spies[0].messages[0] if m["role"] == "user"
        ]
        self.assertEqual(user_msgs, [cases[0].question])
        self.assertNotIn("200000", " ".join(user_msgs))
        self.assertFalse(results[0].score.task_success)

    def test_missing_expected_data_not_used_to_generate_final(self):
        normal_case = self.case
        impossible = EvalCase(
            "custom_revenue_wrong_label", "revenue", normal_case.question,
            1, Answer("answered", 999999, "VND"),
        )
        trace = AgentRunner(MockModel()).run(impossible.question)
        self.assertEqual(trace.answer.value, 200000)
        self.assertFalse(self.eval.score(impossible, trace, Limits()).answer_exact)

    def test_suite_aggregate_normal(self):
        results = SuiteRunner(lambda: MockModel()).run(CASES)
        summary = aggregate(results, mode="mock", scenario="normal")
        self.assertEqual(summary["cases"], 9)
        self.assertEqual(summary["passed"], 9)
        self.assertEqual(summary["task_success_rate"], 1.0)
        self.assertEqual(summary["answer_exact_rate"], 1.0)
        self.assertEqual(summary["abstention_accuracy"], 1.0)
        self.assertEqual(summary["no_data_accuracy"], 1.0)
        self.assertEqual(summary["failure_reasons"], {})

    def test_suite_aggregate_wrong_metric(self):
        results = SuiteRunner(lambda: MockModel("wrong-metric")).run(CASES)
        summary = aggregate(results, mode="mock", scenario="wrong-metric")
        self.assertEqual(summary["passed"], 6)
        self.assertEqual(summary["cases"], 9)
        self.assertEqual(summary["runtime_completion_rate"], 1.0)
        self.assertEqual(summary["answer_exact_rate"], 0.6667)
        self.assertEqual(summary["evidence_grounded_rate"], 1.0)
        self.assertEqual(summary["by_category"]["revenue"]["success_rate"], 0)
        self.assertEqual(summary["failure_reasons"], {"ANSWER_MISMATCH": 3})

    def test_suite_aggregate_premature_final(self):
        results = SuiteRunner(lambda: MockModel("premature-final")).run(CASES)
        summary = aggregate(results, mode="mock", scenario="premature-final")
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["runtime_completion_rate"], 1.0)
        self.assertEqual(summary["evidence_grounded_rate"], 0)

    def test_reject_duplicate_case_ids_and_empty_suite(self):
        with self.assertRaises(ValueError):
            SuiteRunner(lambda: MockModel()).run([])
        with self.assertRaises(ValueError):
            SuiteRunner(lambda: MockModel()).run((CASES[0], CASES[0]))

    def test_unknown_case_id_rejected(self):
        with self.assertRaises(ValueError):
            select_cases("does_not_exist")


if __name__ == "__main__":
    unittest.main()
