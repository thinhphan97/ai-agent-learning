"""Canonical metadata schema shared by indexing and querying.

The tenant scope is supplied by trusted application code, NOT query extraction.
"""

from __future__ import annotations

import re
from typing import Mapping

from contracts import Chunk

REQUIRED_FIELDS = frozenset(
    {"tenant_id", "company_id", "period", "segment", "document_type", "source"}
)
QUERY_FILTER_FIELDS = frozenset({"company_id", "period", "segment", "document_type"})

_COMPANY_ALIASES = {
    "fpt corporation": "FPT",
    "cong ty fpt": "FPT",
    "công ty fpt": "FPT",
    "fpt": "FPT",
    "hpg": "HPG",
    "hoa phat": "HPG",
    "hòa phát": "HPG",
    "none": "NONE",
}
_SEGMENT_ALIASES = {
    "phần mềm": "software",
    "software services": "software",
    "phần cứng": "hardware",
    "none": "general",
}


def canonical_value(field: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Metadata field {field!r} must have a nonempty string")
    v = " ".join(value.strip().split())
    if field == "company_id":
        return _COMPANY_ALIASES.get(v.casefold(), v.upper())
    if field == "period":
        normalized = v.upper().replace("QUÝ ", "Q").replace("QUY ", "Q")
        match = re.fullmatch(r"Q([1-4])[/ -](\d{4})", normalized)
        if match:
            return f"{match.group(2)}-Q{match.group(1)}"
        match = re.fullmatch(r"(\d{4})-Q([1-4])", normalized)
        if match:
            return f"{match.group(1)}-Q{match.group(2)}"
        if v.casefold() == "none":
            return "none"
        raise ValueError(f"Unrecognized period: {value!r}")
    if field == "segment":
        return _SEGMENT_ALIASES.get(v.casefold(), v.casefold().replace(" ", "_"))
    if field == "document_type":
        return v.casefold().replace(" ", "_")
    if field == "tenant_id":
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", v):
            raise ValueError("Invalid tenant_id")
        return v
    if field == "source":
        return v
    raise ValueError(f"Unknown metadata field: {field}")


def normalize_metadata(raw: Mapping[str, str]) -> dict[str, str]:
    missing = REQUIRED_FIELDS - raw.keys()
    unknown = raw.keys() - REQUIRED_FIELDS
    if missing or unknown:
        raise ValueError(
            f"Invalid metadata fields: missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    return {key: canonical_value(key, raw[key]) for key in sorted(REQUIRED_FIELDS)}


def normalize_filters(raw: Mapping[str, str] | None) -> dict[str, str]:
    result = {}
    for field, value in (raw or {}).items():
        if field not in QUERY_FILTER_FIELDS:
            raise ValueError(f"Unsupported or protected query filter: {field}")
        result[field] = canonical_value(field, value)
    return result


def canonical_chunk(chunk_id: str, text: str, metadata: Mapping[str, str]) -> Chunk:
    meta = normalize_metadata(metadata)
    # Use the original text as ground truth; contextual prefix only aids retrieval.
    context = (
        f"Document: {meta['source']}\n"
        f"Company: {meta['company_id']}\n"
        f"Period: {meta['period']}\n"
        f"Section: {meta['segment']}\n"
        f"{text}"
    )
    return Chunk(chunk_id, text, context, meta)


def matches(chunk: Chunk, tenant_id: str, filters: Mapping[str, str]) -> bool:
    return chunk.metadata["tenant_id"] == tenant_id and all(
        chunk.metadata.get(field) == value for field, value in filters.items()
    )
