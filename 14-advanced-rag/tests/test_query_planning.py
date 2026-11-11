"""Standalone query strategy contracts, plan dependencies and trust boundaries."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Budgets, Request, SearchPlan, SearchStep
from controller import AdvancedRAG
from corpus import Corpus
from planning import RetrievalPlanner, validate_plan
from query_analysis import analyze, protected_anchors, validate_rewrite
from routing import SourceRouter
from transform import DeterministicTransformer, LLMTransformer


FINANCE = (
    "Compare FPT international software revenue Q1 and Q2 2025, "
    "calculate growth and explain reasons for change."
)


class QueryPlanningTests(unittest.TestCase):
    def test_multi_aspect_decomposition_and_exact_entity_anchors(self):
        request = Request(FINANCE, "tenant_a", "2026-11-10", ("docs",))
        analysis = analyze(request)
        self.assertEqual(analysis.intent, "finance_compare")
        self.assertEqual(analysis.aspects, ("q1_revenue", "q2_revenue", "drivers"))
        self.assertIn("fpt", analysis.anchors)
        self.assertIn("q1", analysis.anchors)
        self.assertIn("q2", analysis.anchors)
        self.assertIn("q22025", analysis.anchors)

    def test_focused_question_does_not_add_unrequested_facets(self):
        result = analyze(Request(
            "What is the refund deadline?", "tenant_a", "2026-11-10"
        ))
        self.assertEqual(result.aspects, ("deadline",))
        result = analyze(Request(
            "FPT international software revenue Q1 2025",
            "tenant_a", "2026-11-10"
        ))
        self.assertEqual(result.aspects, ("q1_revenue",))

    def test_period_and_error_code_rewrite_drift_rejected(self):
        valid = "FPT Q1/2025 and Q2/2025"
        validate_rewrite(valid, valid + " annual report")
        with self.assertRaises(ValueError):
            validate_rewrite(valid, "FPT Q1/2024 and Q2/2025")
        with self.assertRaises(ValueError):
            validate_rewrite(valid, "FPT Q1/2025 without Q2")
        with self.assertRaises(ValueError):
            validate_rewrite("ERR_AUTH_403 admin", "ERR_AUTH_404 admin")
        with self.assertRaises(ValueError):
            validate_rewrite("Customer_id=2 paid order", "Customer_id=3 paid order")
        validate_rewrite("customer 2 paid orders", "customer_id=2 paid orders")
        with self.assertRaises(ValueError):
            validate_rewrite("customer 2 paid orders", "customer 3 paid orders")

    def test_deterministic_modes_have_distinct_search_patterns(self):
        t = DeterministicTransformer()
        query = "FPT Q1/2025 revenue"
        self.assertEqual(t.transform(query, aspect="q1_revenue", mode="baseline"), (query,))
        self.assertEqual(len(t.transform(query, aspect="q1_revenue", mode="multiquery")), 3)
        self.assertEqual(len(t.transform(query, aspect="q1_revenue", mode="rewrite")), 1)
        hyde = t.transform(query, aspect="q1_revenue", mode="hyde")[0]
        self.assertIn("hypothetical", hyde.lower())
        self.assertIn("FPT", hyde)
        self.assertNotIn("7000", hyde)

    def test_llm_transformer_proposals_are_validated_before_search(self):
        transformer = object.__new__(LLMTransformer)
        transformer.model = "fake"
        transformer.client = MagicMock()
        message = transformer.client.chat.completions.create.return_value.choices[0].message
        message.content = '{"queries":["FPT Q1/2024 revenue"]}'
        with self.assertRaises(ValueError):
            transformer.transform("FPT Q1/2025 revenue", aspect="q1_revenue", mode="rewrite")
        message.content = '{"queries":["FPT Q1/2025 quarterly revenue"]}'
        result = transformer.transform("FPT Q1/2025 revenue", aspect="q1_revenue", mode="rewrite")
        self.assertEqual(result, ("FPT Q1/2025 quarterly revenue",))
        self.assertEqual(transformer.client.chat.completions.create.call_count, 2)

    def test_source_route_is_not_an_authorization_grant(self):
        request = Request(
            "What is paid order revenue for customer 1?",
            "tenant_a", "2026-11-10", ("docs",)
        )
        with self.assertRaises(PermissionError):
            RetrievalPlanner().plan(
                request, analyze(request),
                strategy="planned", budgets=Budgets()
            )
        request = replace(request, allowed_sources=("sql",))
        plan = RetrievalPlanner().plan(
            request, analyze(request), strategy="planned", budgets=Budgets()
        )
        self.assertEqual(plan.steps[0].source, "sql")
        self.assertEqual(plan.steps[0].aspect, "revenue")

    def test_parallel_groups_and_calculation_dependencies(self):
        request = Request(FINANCE, "tenant_a", "2026-11-10", ("docs",))
        planner = RetrievalPlanner()
        plan = planner.plan(request, analyze(request), strategy="adaptive", budgets=Budgets())
        levels = planner.batches(plan)
        self.assertEqual(len(levels), 2)
        self.assertEqual({s.aspect for s in levels[0]},
                         {"q1_revenue", "q2_revenue", "drivers"})
        self.assertEqual(levels[1][0].action, "calculate")
        self.assertEqual(set(levels[1][0].depends_on), {"S1_1", "S2_1"})

    def test_invalid_dependencies_and_budget_rejected(self):
        request = Request(FINANCE, "tenant_a", "2026-11-10", ("docs",))
        bad = SearchPlan(steps=(
            SearchStep("A", "drivers", "x", "docs", depends_on=("B",)),
            SearchStep("B", "drivers", "x", "docs", depends_on=("A",)),
        ), strategy="planned")
        with self.assertRaises(ValueError):
            validate_plan(bad, request, Budgets())
        with self.assertRaises(ValueError):
            RetrievalPlanner().plan(
                request, analyze(request), strategy="planned",
                budgets=Budgets(max_subqueries=2)
            )
        with self.assertRaises(ValueError):
            Budgets(max_rounds=0).validate()

    def test_authorized_sources_reject_stale_and_cross_tenant(self):
        corpus = Corpus.load()
        request = Request(FINANCE, "tenant_a", "2026-11-10", ("docs",))
        ids = {doc.id for doc in corpus.authorized(request, "docs")}
        self.assertNotIn("tenant_b_secret", ids)
        self.assertNotIn("fpt_old", ids)
        with self.assertRaises(PermissionError):
            corpus.verify(request, "tenant_b_secret")
        with self.assertRaises(PermissionError):
            corpus.verify(request, "fpt_old")
        with self.assertRaises(PermissionError):
            corpus.authorized(request, "sql")
        with self.assertRaises(ValueError):
            corpus.authorized(replace(request, as_of_date="bad"), "docs")


if __name__ == "__main__":
    unittest.main()