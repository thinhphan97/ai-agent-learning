"""SQLite long-term memory: durability, validation, privacy boundaries."""
from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "private" / "facts.sqlite3"
        self.store = MemoryStore(self.path)

    def test_store_then_recall(self):
        saved = self.store.save("alice", "programming_language", "Python")
        self.assertEqual(saved.key, "programming_language")
        self.assertEqual(self.store.get("alice", "programming_language").value, "Python")

    def test_reopen_is_durable(self):
        self.store.save("alice", "programming_language", "Python")
        fresh = MemoryStore(self.path)
        self.assertEqual(fresh.get("alice", "programming_language").value, "Python")

    def test_update_replaces_one_fact(self):
        self.store.save("alice", "programming_language", "Python")
        self.store.save("alice", "programming_language", "Go")
        self.assertEqual(self.store.get("alice", "programming_language").value, "Go")

    def test_user_isolation(self):
        self.store.save("alice", "programming_language", "Python")
        self.store.save("bob", "programming_language", "Rust")
        self.assertEqual(self.store.get("bob", "programming_language").value, "Rust")
        self.assertEqual(self.store.get("alice", "programming_language").value, "Python")
        self.assertIsNone(self.store.get("charlie", "programming_language"))

    def test_forget_does_not_affect_other_user(self):
        self.store.save("alice", "programming_language", "Python")
        self.store.save("bob", "programming_language", "Rust")
        self.assertTrue(self.store.forget("alice", "programming_language"))
        self.assertFalse(self.store.forget("alice", "programming_language"))
        self.assertIsNone(self.store.get("alice", "programming_language"))
        self.assertEqual(self.store.get("bob", "programming_language").value, "Rust")

    def test_validate_user_and_key(self):
        for user in ("", "bob/x", "../alice", "A" * 41):
            with self.subTest(user=user), self.assertRaises(ValueError):
                self.store.get(user, "programming_language")
        for key in ("", "Programming_Language", "../secrets", "a" * 41):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.store.get("alice", key)

    def test_validate_value(self):
        for value in ("", "   ", "x" * 201, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.store.save("alice", "programming_language", value)

    def test_whitespace_trimmed(self):
        self.store.save("alice", "programming_language", " Python ")
        self.assertEqual(self.store.get("alice", "programming_language").value, "Python")

    def test_sql_injection_is_data(self):
        self.store.save("alice", "language", "Python'); DROP TABLE facts; --")
        self.assertEqual(
            self.store.get("alice", "language").value,
            "Python'); DROP TABLE facts; --",
        )
        self.store.save("bob", "language", "Go")
        self.assertEqual(self.store.get("bob", "language").value, "Go")

    @unittest.skipIf(os.name == "nt", "POSIX permission test")
    def test_file_permissions_are_private_for_new_db(self):
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_database_symlink_rejected(self):
        link = Path(self.temp.name) / "linked.sqlite3"
        try:
            link.symlink_to(self.path)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks unavailable")
        with self.assertRaises(ValueError):
            MemoryStore(link)


if __name__ == "__main__":
    unittest.main()
