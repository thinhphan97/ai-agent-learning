"""Fixture parsing and strict parent-child provenance validation."""
from __future__ import annotations

import json
from pathlib import Path

from contracts import Child, Parent

PATH = Path(__file__).resolve().parent / "data" / "corpus.json"


class Corpus:
    def __init__(self, parents: list[Parent], children: list[Child]):
        self.parents = {p.id: p for p in parents}
        self.children = tuple(children)
        if len(self.parents) != len(parents) or len({c.id for c in children}) != len(children):
            raise ValueError("Duplicate parent or child IDs")
        if not parents or not children:
            raise ValueError("Empty corpus")
        for child in children:
            parent = self.parents.get(child.parent_id)
            if parent is None:
                raise ValueError(f"Orphan child {child.id}: {child.parent_id}")
            if (
                child.tenant_id != parent.tenant_id
                or child.source != parent.source
                or child.version != parent.version
                or child.section != parent.section
                or child.aspect != parent.aspect
            ):
                raise ValueError(f"Child {child.id} mismatches its parent provenance")
            if child.text not in parent.text:
                raise ValueError(f"Child {child.id} text must be a literal excerpt of parent")
        self._index = {c.id: c for c in children}

    @classmethod
    def load(cls, path: Path = PATH) -> "Corpus":
        raw = json.loads(path.read_text(encoding="utf-8"))
        parents, children = [], []
        for row in raw["parents"]:
            parent = Parent(**{key: row[key] for key in (
                "id", "source", "section", "title", "version", "tenant_id", "aspect", "text"
            )})
            parents.append(parent)
            for i, excerpt in enumerate(row["children"], 1):
                children.append(Child(
                    id=f"{parent.id}:c{i}", parent_id=parent.id,
                    source=parent.source, section=parent.section,
                    version=parent.version, tenant_id=parent.tenant_id,
                    aspect=parent.aspect, text=excerpt,
                ))
        return cls(parents, children)

    def with_context(self, notes: dict[str, str]) -> "Corpus":
        children = [
            Child(
                id=c.id, parent_id=c.parent_id,
                source=c.source, section=c.section,
                version=c.version, tenant_id=c.tenant_id,
                aspect=c.aspect, text=c.text,
                contextual_note=notes.get(c.id, ""),
            )
            for c in self.children
        ]
        return Corpus(list(self.parents.values()), children)

    def authorized(self, tenant_id: str) -> tuple[Child, ...]:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Trusted tenant_id is required")
        return tuple(c for c in self.children if c.tenant_id == tenant_id)

    def get_parent(self, child: Child, tenant_id: str) -> Parent:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Trusted tenant_id required")
        parent = self.parents[child.parent_id]
        if child.tenant_id != tenant_id or parent.tenant_id != tenant_id:
            raise PermissionError("Parent expansion would cross the tenant boundary")
        if (parent.version != child.version or parent.source != child.source):
            raise ValueError("Stale child-to-parent provenance")
        return parent
