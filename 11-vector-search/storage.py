"""SQLite-backed durable vector source records for local Lab 11.

Vector index is intentionally rebuilt on CLI execution; production systems
must persist/synchronize their ANN index separately.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from contracts import VectorRecord
from metadata import normalize_metadata
from metrics import validate_vector


class VectorStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                original_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def upsert_many(self, records: Sequence[VectorRecord]) -> None:
        if len({record.id for record in records}) != len(records):
            raise ValueError("Duplicate IDs within upsert batch")
        with self.conn:
            for record in records:
                vector = validate_vector(record.vector)
                metadata = normalize_metadata(record.metadata)
                self.conn.execute(
                    """
                    INSERT INTO vectors(
                        id, embedding_model, dim, vector_json, original_text, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        embedding_model=excluded.embedding_model,
                        dim=excluded.dim,
                        vector_json=excluded.vector_json,
                        original_text=excluded.original_text,
                        metadata_json=excluded.metadata_json
                """,
                    (
                        record.id,
                        record.embedding_model,
                        len(vector),
                        json.dumps(vector),
                        record.text,
                        json.dumps(metadata),
                    ),
                )

    def delete(self, record_id: str) -> bool:
        with self.conn:
            changed = self.conn.execute(
                "DELETE FROM vectors WHERE id = ?", (record_id,)
            ).rowcount
        return bool(changed)

    def get(self, record_id: str) -> VectorRecord | None:
        row = self.conn.execute(
            """
            SELECT id, embedding_model, vector_json, original_text, metadata_json
            FROM vectors WHERE id = ?
        """,
            (record_id,),
        ).fetchone()
        if row is None:
            return None
        return VectorRecord(
            id=row[0],
            embedding_model=row[1],
            vector=validate_vector(json.loads(row[2])),
            text=row[3],
            metadata=normalize_metadata(json.loads(row[4])),
        )

    def all_records(self) -> list[VectorRecord]:
        rows = self.conn.execute("SELECT id FROM vectors ORDER BY id").fetchall()
        return [self.get(row[0]) for row in rows]
