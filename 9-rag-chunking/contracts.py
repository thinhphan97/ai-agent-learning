"""Standalone, source-grounded contracts shared by all chunking demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str = "inline"


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    kind: str
    start: int | None = None
    end: int | None = None
    parent_id: str | None = None
    section_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Chunk text must be nonempty")
        if (self.start is None) != (self.end is None):
            raise ValueError("Start/end must both be provided or absent")
        if self.start is not None and not 0 <= self.start <= self.end:
            raise ValueError("Invalid source offsets")


@dataclass(frozen=True)
class ChunkSet:
    """Indexable child chunks plus optional larger source/summary nodes."""

    strategy: str
    chunks: tuple[Chunk, ...]
    parents: tuple[Chunk, ...] = ()
    vectors: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        identifiers = [c.id for c in self.chunks + self.parents]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Chunk IDs must be unique")
        if self.vectors and len(self.vectors) != len(self.chunks):
            raise ValueError("One vector required per indexed chunk")


def make_chunk(
    doc: Document, strategy: str, index: int, text: str, *,
    kind: str = "text", start: int | None = None, end: int | None = None,
    parent_id: str | None = None, section_path: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    return Chunk(
        id=f"{doc.id}:{strategy}:{index}",
        document_id=doc.id, text=text.strip(), kind=kind,
        start=start, end=end, parent_id=parent_id,
        section_path=section_path, metadata=metadata or {},
    )
