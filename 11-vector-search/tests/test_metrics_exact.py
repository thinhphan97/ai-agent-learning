"""Similarity math, exact KNN and model compatibility regression."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import VectorRecord
from exact import ExactKNN
from metrics import normalize, similarity, score_to_distance, validate_vector
from metadata import normalize_filters, normalize_metadata


def sample(
    record_id: str,
    vector: tuple[float, ...],
    *,
    tenant: str = "tenant_a",
    company: str = "FPT",
    model: str = "embedding-v1",
) -> VectorRecord:
    return VectorRecord(
        record_id,
        vector,
        f"text {record_id}",
        normalize_metadata(
            {
                "tenant_id": tenant,
                "company_id": company,
                "period": "Q2/2025",
                "section": "finance",
                "source": "synthetic.md",
            }
        ),
        model,
    )


class MetricTests(unittest.TestCase):
    def test_cosine_direction(self):
        self.assertAlmostEqual(similarity((1, 0), (5, 0), "cosine"), 1.0)
        self.assertAlmostEqual(similarity((1, 0), (-1, 0), "cosine"), -1.0)
        self.assertAlmostEqual(similarity((1, 0), (0, 2), "cosine"), 0.0)

    def test_dot_and_l2_differ(self):
        a, b = (1, 0), (0.9, 0.1)
        self.assertGreater(similarity(a, (2, 0), "dot"), similarity(a, b, "dot"))
        self.assertGreater(similarity(a, b, "l2"), similarity(a, (2, 0), "l2"))
        self.assertAlmostEqual(
            score_to_distance(similarity(a, b, "l2"), "l2"), math.sqrt(0.02)
        )

    def test_unit_norm_metric_rank_equivalence(self):
        query = normalize([0.7, 0.3])
        vectors = [normalize(v) for v in ([1, 0], [0, 1], [-1, 0])]
        rankings = []
        for metric in ("cosine", "dot", "l2"):
            rankings.append(
                sorted(
                    range(3),
                    key=lambda i: -similarity(query, vectors[i], metric),
                )
            )
        self.assertEqual(rankings[0], rankings[1])
        self.assertEqual(rankings[1], rankings[2])

    def test_bad_vectors_rejected(self):
        for bad in ([], [float("nan")], [float("inf")], [True], ["2"]):
            with self.assertRaises(ValueError):
                validate_vector(bad)
        with self.assertRaises(ValueError):
            normalize([0, 0])
        with self.assertRaises(ValueError):
            similarity([1], [1, 2], "dot")
        with self.assertRaises(ValueError):
            similarity([0, 0], [1, 0], "cosine")

    def test_exact_knn_sorted_and_filter(self):
        index = ExactKNN(
            [
                sample("a", (1, 0)),
                sample("b", (0.9, 0.1)),
                sample("c", (0.1, 0.9), company="HPG"),
                sample("secret", (1, 0), tenant="tenant_b"),
            ]
        )
        hits, stats = index.search((1, 0), tenant_id="tenant_a", top_k=3)
        self.assertEqual([h.record.id for h in hits], ["a", "b", "c"])
        self.assertEqual(stats.distance_evaluations, 3)
        only, _ = index.search(
            (1, 0),
            tenant_id="tenant_a",
            filters={"company_id": "Hòa Phát"},
        )
        self.assertEqual([h.record.id for h in only], ["c"])

    def test_exact_threshold_may_return_less_than_k(self):
        index = ExactKNN(
            [
                sample("a", (1, 0)),
                sample("b", (0.2, 0.98)),
                sample("c", (0, 1)),
            ]
        )
        hits, _ = index.search(
            (1, 0),
            tenant_id="tenant_a",
            top_k=3,
            min_score=0.9,
        )
        self.assertEqual([hit.record.id for hit in hits], ["a"])

    def test_mixed_model_and_dimension_rejected(self):
        with self.assertRaises(ValueError):
            ExactKNN([sample("a", (1, 0)), sample("b", (1, 0), model="embedding-v2")])
        with self.assertRaises(ValueError):
            ExactKNN([sample("a", (1, 0)), sample("b", (1, 0, 0))])

    def test_metadata_aliases_and_invalid_filters(self):
        self.assertEqual(
            normalize_filters({"company_id": "FPT Corporation", "period": "Q2/2025"}),
            {"company_id": "FPT", "period": "2025-Q2"},
        )
        for field in ("tenant_id", "source", "secret", "acl"):
            with self.assertRaises(ValueError):
                normalize_filters({field: "value"})


if __name__ == "__main__":
    unittest.main()
