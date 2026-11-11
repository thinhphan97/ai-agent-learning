"""Deterministic checks of citation IDs and coverage; NOT semantic fact checking."""
from __future__ import annotations

import re

from contracts import ContextResult


def check_citations(answer: str, context: ContextResult) -> dict[str, object]:
    """Verify references point to evidence actually supplied to the generator.

    This does not prove the cited source supports a particular natural-language
    claim. Full claim entailment/faithfulness requires a separate evaluator.
    """
    found = list(dict.fromkeys(re.findall(r"\[(E\d+)\]", answer)))
    known = {e.id for e in context.evidence}
    unknown = [id for id in found if id not in known]
    return {
        "citation_ids": found,
        "unknown_citations": unknown,
        "known_citations": [id for id in found if id in known],
        "citation_ids_valid": bool(found) and not unknown,
        "semantic_faithfulness_verified": False,
    }


def aspect_coverage(context: ContextResult, required: tuple[str, ...]) -> float:
    required = tuple(dict.fromkeys(required))
    if not required:
        return 1.0
    covered = {e.aspect for e in context.evidence}
    return sum(aspect in covered for aspect in required) / len(required)
