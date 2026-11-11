"""Service wiring and Compose safety tests; do not require Docker/network in CI."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent
sys.path.insert(0, str(LAB))

import agent
from contracts import SearchStats


class ComposeTests(unittest.TestCase):
    def test_compose_is_local_only_and_persistent(self):
        compose = (LAB / "compose.yaml").read_text(encoding="utf-8")
        self.assertIn("image: qdrant/qdrant:v1.20.0", compose)
        self.assertIn('"127.0.0.1:6333:6333"', compose)
        self.assertIn('"127.0.0.1:6334:6334"', compose)
        self.assertIn("qdrant_storage:/qdrant/storage", compose)
        self.assertNotIn("latest", compose)

    def test_root_env_and_readme_have_lifecycle(self):
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
        readme = (LAB / "README.md").read_text(encoding="utf-8")
        root_doc = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("QDRANT_URL=http://127.0.0.1:6333", env_example)
        self.assertIn("QDRANT_API_KEY=", env_example)
        for command in ("docker compose -f 11-vector-search/compose.yaml up -d",
                        "curl --noproxy",
                        "docker compose -f 11-vector-search/compose.yaml down --volumes"):
            self.assertIn(command, readme)
        self.assertIn("External services and local development", root_doc)

    def test_server_selected_by_cli_and_local_mode_separate(self):
        opts = agent.make_parser().parse_args(["--backend", "qdrant", "--index-only"])
        self.assertEqual(opts.backend, "qdrant")
        self.assertEqual(opts.qdrant_collection, "lab11_documents")
        local = agent.make_parser().parse_args(["--backend", "qdrant-local"])
        self.assertEqual(local.qdrant_path, ":memory:")

    def test_server_uses_url_and_root_config_without_connecting(self):
        mock_client = MagicMock()
        args = agent.make_parser().parse_args(["--backend", "qdrant"])
        with patch.object(agent, "QdrantIndex", return_value=mock_client) as factory:
            with patch.dict(os.environ, {
                "QDRANT_URL": "http://127.0.0.1:6333",
                "QDRANT_API_KEY": "",
            }):
                result = agent.make_qdrant(args, [], initialize=False)
        self.assertIs(result, mock_client)
        self.assertEqual(factory.call_args.kwargs["url"], "http://127.0.0.1:6333")
        self.assertIsNone(factory.call_args.kwargs["api_key"])
        self.assertFalse(factory.call_args.kwargs["initialize"])

    def test_server_index_only_writes_qdrant_and_closes_client(self):
        client = MagicMock()
        out = io.StringIO()
        with patch.object(agent, "QdrantIndex", return_value=client) as factory:
            with contextlib.redirect_stdout(out):
                exit_code = agent.main(["--backend", "qdrant", "--index-only"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(client.upsert_many.call_args.args[0]), 10)
        self.assertTrue(factory.call_args.kwargs["initialize"])
        client.close.assert_called_once()
        self.assertEqual(json.loads(out.getvalue())["count"], 10)

    def test_server_search_does_not_bootstrap_collection(self):
        fake = MagicMock()
        fake.search.return_value = ([], SearchStats(
            distance_evaluations=0,
            visited_nodes=0,
            candidates_considered=0,
        ))
        out = io.StringIO()
        with patch.object(agent, "QdrantIndex", return_value=fake) as factory:
            with contextlib.redirect_stdout(out):
                rc = agent.main(["--backend", "qdrant", "--json"])
        self.assertEqual(rc, 0)
        self.assertFalse(factory.call_args.kwargs["initialize"])
        self.assertFalse(factory.call_args.kwargs["reindex"])
        fake.upsert_many.assert_not_called()
        fake.close.assert_called_once()

    def test_destructive_reindex_requires_explicit_index_only(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                agent.main(["--backend", "qdrant", "--reindex"])
        self.assertEqual(error.exception.code, 2)

    def test_server_does_not_accept_local_path(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                agent.main([
                    "--backend", "qdrant", "--qdrant-path", "/tmp/unused",
                ])
        self.assertEqual(error.exception.code, 2)


class QdrantIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_QDRANT_INTEGRATION") == "1",
        "Start Docker Compose and set RUN_QDRANT_INTEGRATION=1",
    )
    def test_real_server_collection_and_filtered_search(self):
        """Only this opt-in test calls Docker Qdrant; deletes its own collection."""
        import uuid
        from embeddings import HashingEmbedder
        from fixtures import load_documents
        from qdrant_adapter import QdrantIndex

        collection = "lab11_it_" + uuid.uuid4().hex[:12]
        embedder = HashingEmbedder()
        records = load_documents(embedder)
        query = tuple(embedder.embed(
            ["FPT international software revenue Q2 2025"]
        )[0])
        url = os.getenv("QDRANT_URL") or "http://127.0.0.1:6333"
        index = QdrantIndex(records, url=url, collection=collection, initialize=True)
        try:
            hits, _ = index.search(
                query, tenant_id="tenant_a",
                filters={"company_id": "FPT Corporation", "period": "Q2/2025"},
            )
            self.assertTrue(hits)
            self.assertEqual(hits[0].record.id, "fpt_q2_sw")
            self.assertTrue(all(hit.record.metadata["tenant_id"] == "tenant_a"
                                for hit in hits))

            second = QdrantIndex(
                records, url=url, collection=collection, initialize=False,
            )
            try:
                again, _ = second.search(
                    query, tenant_id="tenant_a",
                    filters={"section": "software"},
                )
                self.assertTrue(again)
            finally:
                second.close()
        finally:
            index.client.delete_collection(collection)
            index.close()


if __name__ == "__main__":
    unittest.main()