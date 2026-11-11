"""Small, inspectable text utilities; no imported code from earlier labs."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from contracts import Document

_WORD = re.compile(r"\S+")
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+(?=\s|$)|(?=\n|$))", re.MULTILINE)
_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def load_document(path: Path) -> Document:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Input must be a regular, non-symlinked file")
    if path.stat().st_size > 200_000:
        raise ValueError("Input exceeds 200 KB")
    if path.suffix not in {".txt", ".md", ".py"}:
        raise ValueError("Only .txt, .md and .py are supported")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Input file is empty")
    return Document(path.stem, text, str(path))


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Naive sentence boundaries for readable examples, not robust NLP."""
    return [
        (
            m.start() + len(m.group().rstrip()) - len(m.group().strip()),
            m.end() - (len(m.group()) - len(m.group().rstrip())),
        )
        for m in _SENTENCE.finditer(text)
        if m.group().strip()
    ]


def sentence_units(text: str) -> list[tuple[str, int, int]]:
    units = []
    for start, end in sentence_spans(text):
        # Normalize leading whitespace without changing source locations.
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            units.append((text[start:end], start, end))
    return units


def lexical_vector(text: str) -> dict[str, float]:
    words = Counter(w.casefold() for w in _TOKEN.findall(text) if len(w) > 1)
    norm = math.sqrt(sum(x * x for x in words.values()))
    return {k: v / norm for k, v in words.items()} if norm else {}


def similarity(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def vector_cosine(
    a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]
) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Vector dimensions must agree")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (*a, *b)):
        raise ValueError("Invalid vector value")
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def check_size(size: int, overlap: int = 0) -> None:
    if type(size) is not int or not 2 <= size <= 4096:
        raise ValueError("size must be an integer in [2, 4096]")
    if type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")


def cap_chars(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")
