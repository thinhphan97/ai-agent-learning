"""Optional second-stage rerankers; no model downloads in default mode."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, Sequence

from analyzer import tokenize
from contracts import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


class LexicalOverlapReranker:
    """Deterministic educational scoring, NOT a neural Cross-Encoder."""

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        query_terms = set(tokenize(query))
        ranked = []
        for hit in hits:
            document_terms = set(tokenize(hit.chunk.contextual_text))
            overlap = len(query_terms & document_terms)
            # Second stage is a different score: do not add incompatible raw scores.
            relevance = overlap / len(query_terms) if query_terms else 0.0
            ranked.append(Hit(hit.chunk, relevance, "lexical_rerank", hit.sources))
        return sorted(ranked, key=lambda h: (-h.score, h.chunk.id))


class LocalCrossEncoderReranker:
    """Optional genuine Cross-Encoder; requires a pre-downloaded local model."""

    def __init__(self, local_model_dir: str | Path):
        model_path = Path(local_model_dir).expanduser().resolve()
        if not model_path.is_dir():
            raise ValueError("Cross-Encoder requires an existing local model directory")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to use the real Cross-Encoder"
            ) from exc
        self.model = CrossEncoder(str(model_path))

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict(
            [(query, hit.chunk.contextual_text) for hit in hits],
            show_progress_bar=False,
        )
        if len(scores) != len(hits):
            raise ValueError("Cross-Encoder score count mismatch")
        ranked = []
        for hit, raw in zip(hits, scores):
            score = float(raw)
            if not math.isfinite(score):
                raise ValueError("Cross-Encoder returned a nonfinite score")
            ranked.append(Hit(hit.chunk, score, "cross_encoder", hit.sources))
        return sorted(ranked, key=lambda h: (-h.score, h.chunk.id))
