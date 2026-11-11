"""Validated corpus, evaluation cases and human-labeled synthetic oracle.

The generator and scripted runtime MUST NOT import this module's gold labels.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from contracts import Document, EvaluationCase, RequiredFact

DATA = Path(__file__).resolve().parent / "data"


def _date(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("Dates must be ISO-8601 strings")
    return date.fromisoformat(value)


def load_documents(path: Path = DATA / "documents.json") -> dict[str, Document]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs: dict[str, Document] = {}
    for item in raw["documents"]:
        doc = Document(**item)
        if not doc.id or not doc.tenant_id or not doc.text:
            raise ValueError("Document identifiers, tenant and text are mandatory")
        if doc.id in docs:
            raise ValueError(f"Duplicate document ID {doc.id}")
        if doc.valid_until and _date(doc.valid_from) > _date(doc.valid_until):
            raise ValueError(f"Document has invalid effective interval: {doc.id}")
        _date(doc.valid_from)
        docs[doc.id] = doc
    if not docs:
        raise ValueError("Empty document catalog")
    return docs


def is_current(document: Document, as_of_date: str) -> bool:
    point = _date(as_of_date)
    return _date(document.valid_from) <= point and (
        document.valid_until is None or point <= _date(document.valid_until)
    )


def load_cases(
    docs: dict[str, Document], path: Path = DATA / "evaluation_cases.json"
) -> tuple[str, dict[str, EvaluationCase]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw["dataset_version"]
    if not version or not isinstance(version, str):
        raise ValueError("Dataset version required")
    cases = {}
    for row in raw["cases"]:
        facts = tuple(
            RequiredFact(
                id=f["id"], description=f["description"],
                supporting_doc_ids=tuple(f["supporting_doc_ids"]),
            )
            for f in row["required_facts"]
        )
        case = EvaluationCase(
            id=row["id"], question=row["question"], tenant_id=row["tenant_id"],
            as_of_date=row["as_of_date"],
            relevance_grades=dict(row["relevance_grades"]),
            required_facts=facts, reference_answer=row["reference_answer"],
            answerable=row["answerable"],
        )
        if case.id in cases or not case.id or not case.question or not case.tenant_id:
            raise ValueError("Duplicate or invalid evaluation case")
        _date(case.as_of_date)
        if not isinstance(case.answerable, bool):
            raise ValueError("Answerable must be a boolean")
        if not case.reference_answer:
            raise ValueError("Gold reference answer required")
        if (not case.answerable and (facts or case.relevance_grades)) or (
            case.answerable and (not facts or not case.relevance_grades)
        ):
            raise ValueError("Answerability and gold labels contradict")
        if len({f.id for f in facts}) != len(facts):
            raise ValueError("Duplicate required fact IDs")
        for doc_id, grade in case.relevance_grades.items():
            if type(grade) is not int or not 0 <= grade <= 3:
                raise ValueError("Relevance grades must be integers in [0,3]")
            if doc_id not in docs:
                raise ValueError("Gold relevance refers to missing document")
            doc = docs[doc_id]
            if doc.tenant_id != case.tenant_id or not is_current(doc, case.as_of_date):
                raise ValueError("Gold document must be authorized and current")
        for f in facts:
            if not f.id or not f.description or not f.supporting_doc_ids:
                raise ValueError("Missing gold fact definition")
            for doc_id in f.supporting_doc_ids:
                if case.relevance_grades.get(doc_id, 0) <= 0:
                    raise ValueError("Gold fact support must have positive relevance")
        cases[case.id] = case
    if not cases:
        raise ValueError("Evaluation dataset must contain cases")
    return version, cases


def load_suite():
    docs = load_documents()
    version, cases = load_cases(docs)
    return version, docs, cases
