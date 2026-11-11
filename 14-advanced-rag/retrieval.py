"""Tenant-scoped lexical BM25, multiple search queries, deterministic RRF.

No simulated dense hashing is mislabeled as neural similarity. Any new source
adapter must enforce tenant, date and allowlist BEFORE scoring/egress.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from contracts import Request, SearchHit
from corpus import Corpus


def tokenize(value: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:_[^\W_]+)*", value.casefold())


class LocalRetriever:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def search(
        self, request: Request, *, query: str, source: str,
        aspect: str, limit: int, scenario: str = "normal",
        round_number: int = 1,
    ) -> list[SearchHit]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Search limit outside 1..100")
        if scenario not in {"normal", "delayed-drivers", "no-drivers"}:
            raise ValueError("Unknown simulation scenario")
        docs = self.corpus.authorized(request, source)
        if scenario == "no-drivers" or (
            scenario == "delayed-drivers" and round_number == 1
        ):
            # Explicit, controlled fault injection for recovery exercises.
            docs = [d for d in docs if d.id != "fpt_drivers"]
        terms = set(tokenize(query))
        if not terms or not docs:
            return []
        counter = [
            Counter(tokenize(doc.title + " " + doc.text)) for doc in docs
        ]
        doclens = [sum(c.values()) for c in counter]
        mean_len = sum(doclens) / len(doclens)
        n = len(docs)
        df = Counter(term for c in counter for term in c)
        scored = []
        for doc, count, size in zip(docs, counter, doclens):
            score = 0.0
            for term in terms:
                tf = count.get(term, 0)
                if not tf:
                    continue
                idf = math.log1p((n - df[term] + .5) / (df[term] + .5))
                score += idf * (tf * 2.5) / (
                    tf + 1.5 * (.25 + .75 * size / mean_len)
                )
            if score > 0:
                scored.append((doc.id, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [
            SearchHit(
                doc_id=doc_id, score=round(score, 7),
                rank=i, query=query, source=source, aspect=aspect,
            )
            for i, (doc_id, score) in enumerate(scored[:limit], 1)
        ]


def rrf_fuse(groups: list[list[SearchHit]], *, k: int = 60) -> list[SearchHit]:
    if type(k) is not int or k < 1:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    matches: dict[str, SearchHit] = {}
    for group in groups:
        if len({h.doc_id for h in group}) != len(group):
            raise ValueError("Each query result group must have distinct document IDs")
        for rank, hit in enumerate(group, 1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1 / (k + rank)
            # Preserve the earliest evidence hint in deterministic query order.
            matches.setdefault(hit.doc_id, hit)
    result = sorted(scores, key=lambda id: (-scores[id], id))
    return [
        SearchHit(
            doc_id=id, score=round(scores[id], 9), rank=rank,
            query=matches[id].query, source=matches[id].source,
            aspect=matches[id].aspect,
        )
        for rank, id in enumerate(result, 1)
    ]
