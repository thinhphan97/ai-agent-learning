"""Evaluation metrics and executable offline CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from contracts import Hit
from evaluation import evaluate, score_case
from fixtures import CASES
from dense import HashingEmbedder
from pipeline import RetrievalPipeline
from storage import SQLiteFTSIndex
from fixtures import load_corpus


class EvaluationTests(unittest.TestCase):
    def test_metric_perfect_and_missed(self):
        a, b = load_corpus()[:2]
        hits = [Hit(a, 10, "test"), Hit(b, 9, "test")]
        self.assertEqual(score_case(hits, [a.id], 2), {
            "recall": 1.0, "mrr": 1.0, "ndcg": 1.0
        })
        missed = score_case(hits, ["does_not_exist"], 2)
        self.assertEqual(missed["recall"], 0.0)
        self.assertEqual(missed["mrr"], 0.0)

    def test_evaluation_runs_all_methods_with_same_gold_set(self):
        with SQLiteFTSIndex() as store:
            store.upsert_many(load_corpus())
            pipeline = RetrievalPipeline(
                store.all_chunks(), store, embedder=HashingEmbedder(),
            )
            reports = [
                evaluate(pipeline, CASES, method=m, k=3)
                for m in ("bm25", "sqlite", "dense", "hybrid")
            ]
        self.assertTrue(all(report["cases"] == len(CASES) for report in reports))
        for report in reports:
            for key in ("recall_at_k", "mrr_at_k", "ndcg_at_k"):
                self.assertTrue(0 <= report[key] <= 1)

    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, str(LAB / "agent.py"), *args],
            capture_output=True, text=True, timeout=30,
            env=dict(os.environ, LIVE_MODE="0"),
        )

    def test_cli_hybrid_json(self):
        result = self.invoke(
            "--method", "hybrid", "--question", "ERR_AUTH_403 admin error",
            "--top-k", "3", "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["results"][0]["id"], "auth_403")
        self.assertEqual(payload["method"], "hybrid")

    def test_cli_index_persistent_db_then_search(self):
        with tempfile.TemporaryDirectory() as temp:
            db = str(Path(temp) / "store.sqlite3")
            indexed = self.invoke("--db", db, "--index-only")
            self.assertEqual(indexed.returncode, 0, indexed.stderr)
            searched = self.invoke(
                "--db", db, "--method", "sqlite",
                "--question", "ERR_AUTH_403 admin permission",
                "--explain",
            )
            self.assertEqual(searched.returncode, 0, searched.stderr)
            self.assertIn("df=", searched.stdout)
            self.assertIn("auth_403", searched.stdout)

    def test_cli_benchmark_offline(self):
        result = self.invoke("--benchmark")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Recall@3", result.stdout)
        self.assertIn("hybrid+reranker", result.stdout)

    def test_cli_unknown_model_path_is_rejected(self):
        result = self.invoke(
            "--reranker", "cross-encoder", "--cross-encoder-path",
            "/this/definitely/does/not/exist",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("existing local model", result.stderr)

    def test_cli_bad_budget_and_benchmark_api_rejected(self):
        for args in (
            ("--top-k", "5", "--candidate-k", "2"),
            ("--benchmark", "--dense-provider", "api"),
        ):
            result = self.invoke(*args)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
