"""Durable, user-scoped explicit facts in SQLite (standard library only)."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


_USER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}\Z")
_FACT_KEY = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")


@dataclass(frozen=True)
class MemoryFact:
    key: str
    value: str


class MemoryStore:
    """One explicit fact per (user_id, key); no implicit conversation retention.

    The caller supplies the authenticated user scope. Never accept it from a
    model-generated tool argument. A new MemoryStore instance can open the
    same database in another process and retrieve prior facts.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("Memory database path cannot be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.path.exists():
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            os.close(descriptor)
        with self._connection() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                )
                """
            )

    def _connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=5)

    @staticmethod
    def validate_user(user_id: str) -> str:
        if not isinstance(user_id, str) or not _USER_ID.fullmatch(user_id):
            raise ValueError("Invalid user ID (1-40 letters, digits, _ or -)")
        return user_id

    @staticmethod
    def validate_key(key: str) -> str:
        if not isinstance(key, str) or not _FACT_KEY.fullmatch(key):
            raise ValueError("Invalid fact key (lowercase snake_case, up to 40)")
        return key

    @staticmethod
    def validate_value(value: str) -> str:
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
            raise ValueError("Memory value must contain 1-200 characters")
        return value.strip()

    def save(self, user_id: str, key: str, value: str) -> MemoryFact:
        user_id = self.validate_user(user_id)
        key = self.validate_key(key)
        value = self.validate_value(value)
        with self._connection() as db:
            db.execute(
                """
                INSERT INTO facts(user_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value
                """,
                (user_id, key, value),
            )
        return MemoryFact(key, value)

    def get(self, user_id: str, key: str) -> MemoryFact | None:
        user_id = self.validate_user(user_id)
        key = self.validate_key(key)
        with self._connection() as db:
            row = db.execute(
                "SELECT value FROM facts WHERE user_id=? AND key=?",
                (user_id, key),
            ).fetchone()
        return MemoryFact(key, row[0]) if row else None

    def forget(self, user_id: str, key: str) -> bool:
        user_id = self.validate_user(user_id)
        key = self.validate_key(key)
        with self._connection() as db:
            result = db.execute(
                "DELETE FROM facts WHERE user_id=? AND key=?",
                (user_id, key),
            )
            return result.rowcount == 1
