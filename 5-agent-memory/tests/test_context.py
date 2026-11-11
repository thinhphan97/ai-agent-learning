"""Short-term context window: bounded turns and message integrity."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context import ConversationWindow


class ConversationWindowTests(unittest.TestCase):
    def test_keep_complete_turns_only(self):
        buffer = ConversationWindow(max_turns=2)
        for n in range(3):
            buffer.add_turn([
                {"role": "user", "content": f"question-{n}"},
                {"role": "assistant", "content": f"answer-{n}"},
            ])
        messages = buffer.messages()
        self.assertEqual(buffer.turn_count, 2)
        self.assertEqual(
            [m["content"] for m in messages],
            ["question-1", "answer-1", "question-2", "answer-2"],
        )

    def test_tool_call_response_pair_never_split(self):
        buffer = ConversationWindow(max_turns=1)
        buffer.add_turn([
            {"role": "user", "content": "remember"},
            {"role": "assistant", "tool_calls": [{"id": "abc"}]},
            {"role": "tool", "tool_call_id": "abc", "content": "{}"},
            {"role": "assistant", "content": "saved"},
        ])
        self.assertEqual([m["role"] for m in buffer.messages()],
                         ["user", "assistant", "tool", "assistant"])

    def test_no_reference_leaks(self):
        buffer = ConversationWindow(max_turns=1)
        original = [
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "Two"},
        ]
        buffer.add_turn(original)
        original[0]["content"] = "Changed"
        returned = buffer.messages()
        returned[1]["content"] = "Changed"
        self.assertEqual(buffer.messages()[0]["content"], "One")
        self.assertEqual(buffer.messages()[1]["content"], "Two")

    def test_invalid_turns_rejected(self):
        buffer = ConversationWindow()
        for turn in (
            [],
            [{"role": "tool", "content": "bad"}],
            [{"role": "user", "content": "unfinished"}],
        ):
            with self.subTest(turn=turn), self.assertRaises(ValueError):
                buffer.add_turn(turn)

    def test_invalid_limits_rejected(self):
        for value in (0, -1, 2.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ConversationWindow(value)


if __name__ == "__main__":
    unittest.main()
