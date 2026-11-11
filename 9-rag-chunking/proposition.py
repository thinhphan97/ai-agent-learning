"""Source-linked proposition extraction: conservative offline, optional LLM."""

from __future__ import annotations

from typing import Protocol

from contracts import ChunkSet, Document, make_chunk
from shared import sentence_units


class PropositionExtractor(Protocol):
    def extract_propositions(self, passage: str) -> list[str]: ...


def propositions(
    doc: Document,
    *,
    extractor: PropositionExtractor | None = None,
) -> ChunkSet:
    """Offline: each original sentence is a candidate, NOT verified atomic fact.

    With extractor: generated propositions have source offsets omitted; each has
    a pointer to its original sentence for independent human/evaluator review.
    """
    units = sentence_units(doc.text)
    if not units:
        raise ValueError("No sentences to extract")
    chunks = []
    for sentence, start, end in units:
        candidates = (
            extractor.extract_propositions(sentence) if extractor else [sentence]
        )
        if not 1 <= len(candidates) <= 30:
            raise ValueError("Extractor must return 1..30 propositions per sentence")
        for value in candidates:
            if not isinstance(value, str) or not value.strip() or len(value) > 1000:
                raise ValueError("Invalid proposition")
            chunks.append(
                make_chunk(
                    doc,
                    "proposition",
                    len(chunks),
                    value,
                    kind="fact_candidate",
                    start=None if extractor else start,
                    end=None if extractor else end,
                    metadata={
                        "source_sentence": sentence,
                        "source_span": [start, end],
                        "extraction": (
                            "llm_unverified" if extractor else "sentence_candidate"
                        ),
                        "requires_source_verification": bool(extractor),
                    },
                )
            )
    return ChunkSet("proposition", tuple(chunks))
