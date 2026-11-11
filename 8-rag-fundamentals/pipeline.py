"""RAG orchestration: retrieve -> context budget -> generate -> check citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from generation import Generator
from retrieval import Hit, Retriever

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+:\d+)\]")


@dataclass(frozen=True)
class RAGAnswer:
    status: str  # answered | abstain
    text: str
    citations: tuple[str, ...]
    hits: tuple[Hit, ...]
    reason: str

    @property
    def source_documents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(hit.chunk.document_id for hit in self.hits))


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        top_k: int = 3,
        min_score: float = 0.1,
        max_context_chars: int = 3000,
    ):
        if type(max_context_chars) is not int or max_context_chars < 100:
            raise ValueError("max_context_chars must be an integer >= 100")
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_chars = max_context_chars

    def answer(self, question: str) -> RAGAnswer:
        raw_hits = self.retriever.retrieve(
            question, top_k=self.top_k, min_score=self.min_score
        )
        hits: list[Hit] = []
        context_used = 0
        for hit in raw_hits:
            new_size = len(hit.chunk.text) + len(hit.chunk.id) + 4
            if context_used + new_size > self.max_context_chars:
                continue
            hits.append(hit)
            context_used += new_size

        if not hits:
            return RAGAnswer(
                "abstain",
                "I don't know based on the documents.",
                (),
                (),
                "NO_RETRIEVAL_EVIDENCE",
            )

        try:
            output = self.generator.generate(question, hits).strip()
        except Exception:
            return RAGAnswer(
                "abstain",
                "Answer generation was unavailable.",
                (),
                tuple(hits),
                "GENERATION_FAILED",
            )

        if not output or output == "I don't know based on the documents.":
            return RAGAnswer(
                "abstain",
                "I don't know based on the documents.",
                (),
                tuple(hits),
                "INSUFFICIENT_CONTEXT",
            )

        cited = tuple(dict.fromkeys(CITATION_PATTERN.findall(output)))
        allowed = {hit.chunk.id for hit in hits}
        if not cited or not set(cited).issubset(allowed):
            return RAGAnswer(
                "abstain",
                "Answer rejected: missing or invalid source citations.",
                (),
                tuple(hits),
                "INVALID_CITATIONS",
            )

        return RAGAnswer("answered", output, cited, tuple(hits), "CITATIONS_VALID")
