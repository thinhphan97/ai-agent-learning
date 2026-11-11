"""Second-stage rerankers; toy lexical vs a real local Cross-Encoder."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from contracts import Hit, RerankedHit
from retrieval import tokenize


class ScoreProvider(Protocol):
    def scores(self, query: str, candidates: list[Hit]) -> list[float]: ...


class LexicalReranker:
    """Offline lexical overlap heuristic — NOT a Cross-Encoder."""

    def scores(self, query: str, candidates: list[Hit]) -> list[float]:
        question_terms = set(tokenize(query))
        if not question_terms:
            raise ValueError("Query must contain tokens")
        output = []
        for hit in candidates:
            # Score original excerpt AND contextual metadata to compare different
            # stages without inventing a semantic neural model.
            terms = set(tokenize(hit.child.indexed_text))
            output.append(len(question_terms & terms) / len(question_terms))
        return output


class LocalCrossEncoder:
    """Genuine Cross-Encoder with an EXISTING on-disk model directory.

    Never downloads weights implicitly. Uses the root requirements.txt dependency.
    """

    def __init__(self, model_dir: str):
        path = Path(model_dir).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("Provide an existing local Cross-Encoder model directory")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Install root requirements.txt for sentence-transformers") from exc
        self.model = CrossEncoder(str(path))

    def scores(self, query: str, candidates: list[Hit]) -> list[float]:
        if not candidates:
            return []
        scores = self.model.predict(
            [(query, hit.child.indexed_text) for hit in candidates],
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def rerank(
    query: str, candidates: list[Hit],
    scorer: ScoreProvider,
) -> list[RerankedHit]:
    if not candidates:
        return []
    scores = scorer.scores(query, candidates)
    if len(scores) != len(candidates):
        raise ValueError("Reranker returned wrong score count")
    if any(not math.isfinite(float(x)) for x in scores):
        raise ValueError("Reranker returned nonfinite scores")
    pairs = sorted(
        zip(candidates, scores),
        key=lambda p: (-p[1], p[0].child.id),
    )
    return [
        RerankedHit(
            child=hit.child,
            retrieval_rank=hit.rank,
            retrieval_score=hit.score,
            score=float(score),
            rank=i,
        )
        for i, (hit, score) in enumerate(pairs, 1)
    ]
