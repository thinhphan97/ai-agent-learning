"""Deterministic teaching tokenizer; not a Vietnamese morphological analyzer."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[^\W_]+(?:[_-][^\W_]+)*", re.UNICODE)
STOPWORDS = frozenset(
    "a an and are at be by can do does for from how i in is it of on or the "
    "their there these this to was were what when where which who why will "
    "with you your during while please show me tell about".split()
)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("Search text must be a string")
    return [
        term.casefold()
        for term in WORD_RE.findall(text)
        if term.casefold() not in STOPWORDS
    ]


def validate_query(question: str, top_k: int) -> None:
    if not isinstance(question, str) or not question.strip() or len(question) > 1000:
        raise ValueError("Query must contain 1..1000 characters")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer between 1 and 100")
