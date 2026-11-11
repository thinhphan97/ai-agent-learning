"""Small synthetic corpus with strict authorization and source validity filters."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from contracts import Document, Request

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus.json"
SOURCE_TYPES = frozenset({"docs", "code", "sql"})


def _date(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("as_of_date must be an ISO date string")
    return date.fromisoformat(value)


class Corpus:
    def __init__(self, documents: list[Document]):
        if not documents or len({d.id for d in documents}) != len(documents):
            raise ValueError("Corpus must be nonempty with unique document IDs")
        for doc in documents:
            if not doc.id or not doc.title or not doc.text or not doc.tenant_id:
                raise ValueError("Missing document provenance or content")
            if doc.source_type not in SOURCE_TYPES:
                raise ValueError("Unknown document source type")
            if doc.valid_until and _date(doc.valid_until) < _date(doc.valid_from):
                raise ValueError("Document has invalid effective dates")
        self.documents = {d.id: d for d in documents}

    @classmethod
    def load(cls, path: Path = CORPUS_PATH) -> "Corpus":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls([Document(**row) for row in raw["documents"]])

    def authorized(self, request: Request, source: str) -> list[Document]:
        if not request.tenant_id or not request.tenant_id.strip():
            raise ValueError("Trusted tenant is mandatory")
        requested_date = _date(request.as_of_date)
        if source not in SOURCE_TYPES or source not in request.allowed_sources:
            raise PermissionError(f"Source {source} not on the trusted allowlist")
        return [
            d for d in self.documents.values()
            if d.tenant_id == request.tenant_id
            and d.source_type == source
            and _date(d.valid_from) <= requested_date
            and (d.valid_until is None or requested_date <= _date(d.valid_until))
        ]

    def verify(self, request: Request, doc_id: str) -> Document:
        if doc_id not in self.documents:
            raise ValueError("Unknown document")
        doc = self.documents[doc_id]
        if doc not in self.authorized(request, doc.source_type):
            raise PermissionError("Cross-tenant, stale or disallowed document")
        return doc
