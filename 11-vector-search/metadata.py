"""Canonical document metadata and trusted-scope search filters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CANONICAL_FIELDS = frozenset({"tenant_id", "company_id", "period", "section", "source"})
ALLOWED_QUERY_FILTERS = frozenset({"company_id", "period", "section"})


def _canon(field: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    v = " ".join(value.strip().split())
    if field == "company_id":
        return {
            "fpt corporation": "FPT",
            "công ty fpt": "FPT",
            "hòa phát": "HPG",
            "hoa phat": "HPG",
        }.get(v.casefold(), v.upper())
    if field == "period":
        import re

        standardized = v.upper()
        if standardized.casefold() == "none":
            return "none"
        pattern = re.fullmatch(r"Q([1-4])[/ -](\d{4})", standardized)
        if pattern:
            return f"{pattern.group(2)}-Q{pattern.group(1)}"
        if re.fullmatch(r"\d{4}-Q[1-4]", standardized):
            return standardized
        raise ValueError(f"Unsupported period format: {value!r}")
    if field in {"tenant_id", "section", "source"}:
        return v if field in {"tenant_id", "source"} else v.casefold().replace(" ", "_")
    raise ValueError(f"Unknown metadata field: {field}")


def normalize_metadata(raw: Mapping[str, str]) -> dict[str, str]:
    if set(raw) != CANONICAL_FIELDS:
        raise ValueError(f"Metadata keys must equal {sorted(CANONICAL_FIELDS)}")
    return {key: _canon(key, raw[key]) for key in sorted(CANONICAL_FIELDS)}


def normalize_filters(filters: Mapping[str, str] | None) -> dict[str, str]:
    output: dict[str, str] = {}
    for field, value in (filters or {}).items():
        if field not in ALLOWED_QUERY_FILTERS:
            raise ValueError(f"Disallowed metadata filter: {field}")
        output[field] = _canon(field, value)
    return output


def matches(
    metadata: Mapping[str, str],
    tenant_id: str,
    filters: Mapping[str, str],
) -> bool:
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("Trusted tenant_id must be provided")
    return metadata.get("tenant_id") == tenant_id and all(
        metadata.get(key) == value for key, value in filters.items()
    )
