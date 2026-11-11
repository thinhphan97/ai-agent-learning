"""Load bounded local text documents and create overlapping word chunks.

No PDFs, OCR, external network, or implicit reads outside the given directory.
Words, not model tokens, are used for this introductory demonstration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    text: str
    word_start: int
    word_end: int


def load_documents(
    folder: Path, *, max_files: int = 30, max_bytes: int = 50_000
) -> list[Document]:
    folder = Path(folder)
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Document source must be a real directory")
    paths = sorted(folder.glob("*.txt"))
    if not paths or len(paths) > max_files:
        raise ValueError("Document directory must contain 1 to 30 .txt files")
    documents = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Symlink or non-file documents are unsupported")
        if path.stat().st_size > max_bytes:
            raise ValueError("Document exceeds the size limit")
        source = path.read_text(encoding="utf-8").strip()
        if not source:
            raise ValueError(f"Empty document: {path.name}")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", path.stem):
            raise ValueError("Document filenames must be ASCII identifiers")
        title = source.splitlines()[0].strip()
        documents.append(Document(id=path.stem, title=title, text=source))
    return documents


def chunk_documents(
    documents: list[Document], *, chunk_size: int = 40, overlap: int = 10
) -> list[Chunk]:
    if (
        type(chunk_size) is not int
        or type(overlap) is not int
        or not 5 <= chunk_size <= 200
        or not 0 <= overlap < chunk_size
    ):
        raise ValueError("Use 5 <= chunk_size <= 200 and 0 <= overlap < chunk_size")
    if not documents:
        raise ValueError("Cannot chunk an empty corpus")
    chunks: list[Chunk] = []
    ids: set[str] = set()
    for document in documents:
        if document.id in ids:
            raise ValueError("Duplicate document ID")
        ids.add(document.id)
        words = document.text.split()
        if not words:
            raise ValueError("Empty document")
        start = 0
        position = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(
                Chunk(
                    id=f"{document.id}:{position}",
                    document_id=document.id,
                    title=document.title,
                    text=" ".join(words[start:end]),
                    word_start=start,
                    word_end=end,
                )
            )
            if end == len(words):
                break
            start = end - overlap
            position += 1
    return chunks
