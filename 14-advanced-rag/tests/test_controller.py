"""Adversarial controller budgets, recovery, routing and stable parallel traces."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Budgets, Request
from controller import AdvancedRAG

Q = (
    "Compare FPT international software revenue in Q1 and Q2 2025, "
    "calculate growth and explain the main reasons for change."
)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.request = Request(Q, "tenant_a", "2026-11-10", ("docs",))
        self.agent = AdvancedRAG()

    def test_adaptive_recovers_specific_missing_aspect(self):
        result = self.agent.run(
            self.request, strategy="adaptive", scenario="delayed-drivers"
        )
        self.assertEqual(result["answer"]["status"], "answered")
        self.assertEqual(result["coverage_by_runtime_heuristics"], 1)
        self.assertEqual(result["usage"]["search_calls"], 4)
        self.assertEqual(result["usage"]["retrieval_rounds"], 2)
        self.assertEqual(result["stop_reason"], "sufficient_after_recovery")
        recovery = [e for e in result["events"] if e["event"] == "targeted_recovery"]
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0]["missing_aspects"], ["drivers"])
        self.assertEqual(result["events"][0]["event"], "parallel_retrieval_batch")
        self.assertEqual(result["events"][0]["step_ids"],
                         ["S1_1", "S2_1", "S3_1"])

    def test_planned_has_no_recovery_and_reports_missing(self):
        result = self.agent.run(
            self.request, strategy="planned", scenario="delayed-drivers"
        )
        self.assertEqual(result["answer"]["status"], "partial")
        self.assertEqual(result["usage"]["search_calls"], 3)
        self.assertEqual(result["usage"]["retrieval_rounds"], 1)
        self.assertEqual(result["missing_aspects"], ["drivers"])
        self.assertFalse(any(
            e["event"] == "targeted_recovery" for e in result["events"]
        ))

    def test_no_evidence_round_cannot_loop(self):
        result = self.agent.run(
            self.request, strategy="adaptive", scenario="no-drivers",
            budgets=Budgets(max_rounds=3),
        )
        self.assertEqual(result["stop_reason"], "no_new_evidence")
        self.assertEqual(result["usage"]["retrieval_rounds"], 2)
        self.assertIn("drivers", result["missing_aspects"])

    def test_strict_round_budget_forces_partial_not_guessing(self):
        result = self.agent.run(
            self.request, strategy="adaptive", scenario="delayed-drivers",
            budgets=Budgets(max_rounds=1),
        )
        self.assertEqual(result["stop_reason"], "max_rounds_exhausted")
        self.assertEqual(result["answer"]["status"], "partial")
        self.assertEqual(result["usage"]["search_calls"], 3)

    def test_strict_search_calls_budget_prevents_retry(self):
        result = self.agent.run(
            self.request, strategy="adaptive", scenario="delayed-drivers",
            budgets=Budgets(max_search_calls=3),
        )
        self.assertEqual(result["usage"]["search_calls"], 3)
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertEqual(result["answer"]["status"], "partial")

    def test_max_subqueries_prevents_recovery(self):
        result = self.agent.run(
            self.request, strategy="adaptive", scenario="delayed-drivers",
            budgets=Budgets(max_subqueries=3),
        )
        self.assertEqual(result["usage"]["subqueries"], 3)
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertIn("drivers", result["missing_aspects"])

    def test_context_limit_not_misdiagnosed_as_retrieval_gap(self):
        result = self.agent.run(
            self.request, strategy="adaptive", scenario="normal",
            budgets=Budgets(max_evidence=1),
        )
        self.assertEqual(result["stop_reason"], "context_budget_limited")
        self.assertEqual(result["usage"]["search_calls"], 3)
        self.assertEqual(result["usage"]["retrieval_rounds"], 1)
        self.assertEqual(result["answer"]["status"], "partial")
        self.assertFalse(any(
            event["event"] == "targeted_recovery" for event in result["events"]
        ))

    def test_llm_query_call_budget_fails_before_model_invocation(self):
        class SpyTransformer:
            def __init__(self):
                self.calls = 0
            def transform(self, query, *, aspect, mode):
                self.calls += 1
                return (query,)
        spy = SpyTransformer()
        controller = AdvancedRAG(transformer_provider="llm", transformer=spy)
        result = controller.run(
            self.request, strategy="multiquery",
            budgets=Budgets(max_llm_calls=1, max_search_calls=2),
        )
        self.assertEqual(spy.calls, 0)
        self.assertEqual(result["usage"]["search_calls"], 0)
        self.assertEqual(result["answer"]["status"], "abstained")

    def test_parallel_trace_is_reproducible(self):
        results = [
            self.agent.run(self.request, strategy="adaptive", scenario="delayed-drivers")
            for _ in range(3)
        ]
        for result in results[1:]:
            self.assertEqual(result["events"], results[0]["events"])
            self.assertEqual(result["evidence"], results[0]["evidence"])

    def test_router_routings_for_code_sql_and_docs(self):
        routes = [
            ("auth", "Why does ERR_AUTH_403 occur in administrator dashboard code?", "code", "auth_code"),
            ("sql", "What is paid order revenue for customer 1?", "sql", "sql_revenue_snapshot"),
            ("vllm", "How can vLLM Qwen3 reduce CUDA out of memory errors?", "docs", "vllm_guide"),
        ]
        for _, question, route, expected in routes:
            with self.subTest(question=question):
                result = self.agent.run(
                    Request(question, "tenant_a", "2026-11-10", (route,)),
                    strategy="planned",
                )
                self.assertIn(expected, [e["doc_id"] for e in result["evidence"]])
                self.assertEqual(result["answer"]["status"], "answered")

    def test_explicit_multisource_routing_and_no_scope_widening(self):
        q = "Explain vLLM Qwen3 CUDA memory config and settings"
        result = self.agent.run(
            Request(q, "tenant_a", "2026-11-10", ("docs", "code")),
            strategy="planned",
        )
        self.assertEqual({s["source"] for s in result["plan"]}, {"docs", "code"})
        with self.assertRaises(PermissionError):
            self.agent.run(
                Request(q, "tenant_a", "2026-11-10", ("docs",)),
                strategy="planned",
            )

    def test_unknown_customer_and_scoped_unknown_tenant_abstain(self):
        result = self.agent.run(
            replace(self.request, tenant_id="tenant_x"),
            strategy="adaptive",
        )
        self.assertEqual(result["answer"]["status"], "abstained")
        self.assertFalse(result["retrieved"])

    def test_budget_validation_and_provider_constraints(self):
        with self.assertRaises(ValueError):
            self.agent.run(
                self.request, strategy="adaptive",
                budgets=Budgets(max_hits_per_search=0)
            )
        with self.assertRaises(ValueError):
            AdvancedRAG(transformer_provider="unavailable")


if __name__ == "__main__":
    unittest.main()