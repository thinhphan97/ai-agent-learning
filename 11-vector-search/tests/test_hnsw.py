"""Layered graph construction, search, false-negative demonstrations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import synthetic_vectors
from hnsw import EducationalHNSW
from exact import ExactKNN
from test_metrics_exact import sample


class HNSWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.queries = synthetic_vectors(count=110, dimensions=8)
        cls.graph = EducationalHNSW(
            cls.records,
            m=8,
            ef_construction=32,
            seed=17,
        )
        cls.exact = ExactKNN(cls.records)

    def test_graph_has_layers_and_bounded_neighbors(self):
        levels = self.graph.level_counts()
        self.assertIn(0, levels)
        self.assertEqual(levels[0], len(self.records))
        self.assertEqual(
            sum(level > 0 for level in self.graph.levels),
            sum(1 for level in self.graph.levels if level >= 1),
        )
        self.assertTrue(
            all(
                len(neighbors) <= self.graph.m
                for nodes in self.graph.graph.values()
                for neighbors in nodes.values()
            )
        )
        self.assertGreater(self.graph.build_distance_evaluations, 0)

    def test_ann_returns_bounded_candidates(self):
        hits, stats = self.graph.search(
            self.queries[0],
            tenant_id="tenant_a",
            top_k=5,
            ef_search=16,
        )
        self.assertTrue(0 < len(hits) <= 5)
        self.assertGreater(stats.distance_evaluations, 0)
        self.assertLessEqual(stats.candidates_considered, 16)

    def test_deterministic_graph_and_top_k(self):
        graph2 = EducationalHNSW(self.records, m=8, ef_construction=32, seed=17)
        self.assertEqual(self.graph.levels, graph2.levels)
        first, _ = self.graph.search(
            self.queries[1],
            tenant_id="tenant_a",
            top_k=5,
            ef_search=32,
        )
        second, _ = graph2.search(
            self.queries[1],
            tenant_id="tenant_a",
            top_k=5,
            ef_search=32,
        )
        self.assertEqual([x.record.id for x in first], [x.record.id for x in second])

    def test_post_filter_can_return_zero_despite_authorized_record(self):
        records = [
            sample("secret", (1, 0), tenant="tenant_b"),
            sample("valid", (0.9, 0.1), tenant="tenant_a"),
            sample("another", (0.1, 0.9), tenant="tenant_a"),
        ]
        graph = EducationalHNSW(records, m=2, ef_construction=4)
        post, _ = graph.search(
            (1, 0),
            tenant_id="tenant_a",
            top_k=1,
            ef_search=3,
            filter_mode="post",
        )
        safe, _ = graph.search(
            (1, 0),
            tenant_id="tenant_a",
            top_k=1,
            ef_search=3,
            filter_mode="exact-filter",
        )
        self.assertEqual(post, [])
        self.assertEqual([h.record.id for h in safe], ["valid"])

    def test_exact_filter_is_exact_knn_not_ann(self):
        graph_hits, _ = self.graph.search(
            self.queries[0],
            tenant_id="tenant_a",
            filters={"company_id": "FPT"},
            top_k=5,
            ef_search=16,
            filter_mode="exact-filter",
        )
        exact_hits, _ = self.exact.search(
            self.queries[0],
            tenant_id="tenant_a",
            filters={"company_id": "FPT"},
            top_k=5,
        )
        self.assertEqual(
            [hit.record.id for hit in graph_hits],
            [hit.record.id for hit in exact_hits],
        )

    def test_invalid_parameters_fail(self):
        with self.assertRaises(ValueError):
            EducationalHNSW(self.records, m=0)
        with self.assertRaises(ValueError):
            EducationalHNSW(self.records, m=8, ef_construction=2)
        with self.assertRaises(ValueError):
            self.graph.search(
                self.queries[0],
                tenant_id="tenant_a",
                top_k=5,
                ef_search=3,
            )


if __name__ == "__main__":
    unittest.main()
