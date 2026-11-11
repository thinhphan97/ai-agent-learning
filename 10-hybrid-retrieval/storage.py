"""Persistent SQLite FTS5 BM25 index with transactional metadata synchronization.

FTS5's bm25() returns lower (often negative) values for better matches.
The API converts to descending scores for the shared Hit contract.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping, Sequence

from analyzer import tokenize, validate_query
from contracts import Chunk, Hit
from metadata import QUERY_FILTER_FIELDS, normalize_filters, normalize_metadata


class SQLiteFTSIndex:
    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                original_text TEXT NOT NULL,
                contextual_text TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                company_id TEXT NOT NULL,
                period TEXT NOT NULL,
                segment TEXT NOT NULL,
                document_type TEXT NOT NULL,
                source TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(chunk_id UNINDEXED, contextual_text, tokenize='unicode61')
        """)
        self.conn.commit()

    def __enter__(self) -> SQLiteFTSIndex:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def count(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0])

    def _upsert(self, chunk: Chunk) -> None:
        meta = normalize_metadata(chunk.metadata)
        # Same transaction updates source-of-truth row and FTS materialization.
        self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.id,))
        self.conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, original_text, contextual_text, tenant_id,
                company_id, period, segment, document_type, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                original_text=excluded.original_text,
                contextual_text=excluded.contextual_text,
                tenant_id=excluded.tenant_id,
                company_id=excluded.company_id,
                period=excluded.period,
                segment=excluded.segment,
                document_type=excluded.document_type,
                source=excluded.source
        """,
            (
                chunk.id,
                chunk.text,
                chunk.contextual_text,
                meta["tenant_id"],
                meta["company_id"],
                meta["period"],
                meta["segment"],
                meta["document_type"],
                meta["source"],
            ),
        )
        self.conn.execute(
            "INSERT INTO chunks_fts (chunk_id, contextual_text) VALUES (?, ?)",
            (chunk.id, chunk.contextual_text),
        )

    def upsert_many(self, chunks: Sequence[Chunk]) -> None:
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("Duplicate IDs within update batch")
        with self.conn:
            for chunk in chunks:
                self._upsert(chunk)

    def delete(self, chunk_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
            self.conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))

    @staticmethod
    def _row_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["chunk_id"],
            text=row["original_text"],
            contextual_text=row["contextual_text"],
            metadata={
                field: row[field]
                for field in (
                    "tenant_id",
                    "company_id",
                    "period",
                    "segment",
                    "document_type",
                    "source",
                )
            },
        )

    def all_chunks(self) -> list[Chunk]:
        """Internal index reconstruction; NEVER expose without tenant authorization."""
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall()
        return [self._row_chunk(row) for row in rows]

    def search(
        self,
        question: str,
        *,
        tenant_id: str,
        filters: Mapping[str, str] | None = None,
        top_k: int = 5,
    ) -> list[Hit]:
        validate_query(question, top_k)
        canonical = normalize_filters(filters)
        if not tenant_id:
            raise ValueError("Trusted tenant_id is required")
        words = list(dict.fromkeys(tokenize(question)))
        if not words:
            return []
        # Quoting each analyzed token safely builds an OR lexical query.
        # Parameter binding then protects the SQL statement itself.
        match_expr = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
        where = ["chunks_fts MATCH ?", "c.tenant_id = ?"]
        params: list[str | int] = [match_expr, tenant_id]
        for key in sorted(canonical):
            if key not in QUERY_FILTER_FIELDS:
                raise ValueError(f"Unsupported filter: {key}")
            where.append(
                f"c.{key} = ?"
            )  # Identifiers come only from a hardcoded allowlist.
            params.append(canonical[key])
        params.append(top_k)
        statement = (
            "SELECT c.*, bm25(chunks_fts) AS fts_score "
            "FROM chunks_fts JOIN chunks AS c ON chunks_fts.chunk_id = c.chunk_id "
            "WHERE "
            + " AND ".join(where)
            + " ORDER BY fts_score ASC, c.chunk_id ASC LIMIT ?"
        )
        rows = self.conn.execute(statement, params).fetchall()
        return [
            Hit(self._row_chunk(row), -float(row["fts_score"]), "sqlite_fts5")
            for row in rows
        ]
