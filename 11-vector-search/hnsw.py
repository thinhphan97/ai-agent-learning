"""Educational, layered HNSW implementation in pure Python.

This implements actual graph construction and beam-search traversal but omits
production optimizations and uses a basic deterministic neighbor heuristic.
It is a TEACHING GRAPH, not a benchmark substitute for hnswlib/Qdrant.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

from contracts import SearchHit, SearchStats, VectorRecord
from exact import ExactKNN
from metadata import matches, normalize_filters
from metrics import similarity, validate_search, validate_vector


class EducationalHNSW:
    def __init__(
        self,
        records: Sequence[VectorRecord],
        *,
        metric: str = "cosine",
        m: int = 8,
        ef_construction: int = 32,
        seed: int = 17,
        max_level_cap: int = 8,
    ):
        self.exact = ExactKNN(records, metric)
        if type(m) is not int or m < 2:
            raise ValueError("M must be >= 2")
        if type(ef_construction) is not int or ef_construction < m:
            raise ValueError("ef_construction must be >= M")
        if type(max_level_cap) is not int or max_level_cap < 1:
            raise ValueError("max_level_cap must be >= 1")
        self.records = self.exact.records
        self.metric = metric
        self.dimensions = self.exact.dimensions
        self.model_id = self.exact.model_id
        self.m = m
        self.ef_construction = ef_construction
        self.max_level_cap = max_level_cap
        self.random = random.Random(seed)
        self.graph: dict[int, dict[int, set[int]]] = defaultdict(dict)
        self.levels: list[int] = []
        self.entrypoint: int | None = None
        self.max_level = -1
        self.build_distance_evaluations = 0
        for idx in range(len(self.records)):
            self._insert(idx)

    def _score(self, vector: Sequence[float], index: int, counter: list[int]) -> float:
        counter[0] += 1
        return similarity(vector, self.records[index].vector, self.metric)

    def _random_level(self) -> int:
        level = 0
        while level < self.max_level_cap and self.random.random() < 1 / math.e:
            level += 1
        return level

    def _greedy(
        self, vector: Sequence[float], current: int, level: int, counter: list[int]
    ) -> int:
        best_score = self._score(vector, current, counter)
        while True:
            better: tuple[float, int] | None = None
            for neighbor in sorted(self.graph[level].get(current, ())):
                score = self._score(vector, neighbor, counter)
                if score > best_score:
                    if better is None or (score, -neighbor) > (better[0], -better[1]):
                        better = (score, neighbor)
            if better is None:
                return current
            best_score, current = better

    def _search_layer(
        self,
        vector: Sequence[float],
        entrypoints: Sequence[int],
        ef: int,
        layer: int,
        counter: list[int],
    ) -> list[tuple[float, int]]:
        """A greedy beam search; output sorted by decreasing similarity."""
        visited = set(entrypoints)
        candidates: list[tuple[float, int]] = []
        best: list[tuple[float, int]] = []
        for idx in entrypoints:
            s = self._score(vector, idx, counter)
            heapq.heappush(candidates, (-s, idx))
            heapq.heappush(best, (s, -idx))
        while candidates:
            neg_score, current = heapq.heappop(candidates)
            current_score = -neg_score
            if len(best) >= ef and current_score < best[0][0]:
                break
            for neighbor in sorted(self.graph[layer].get(current, ())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                s = self._score(vector, neighbor, counter)
                if len(best) < ef or s > best[0][0]:
                    heapq.heappush(candidates, (-s, neighbor))
                    heapq.heappush(best, (s, -neighbor))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted(
            ((score, -negative_idx) for score, negative_idx in best),
            key=lambda pair: (-pair[0], self.records[pair[1]].id),
        )

    def _prune(self, index: int, layer: int) -> None:
        neighbors = self.graph[layer][index]
        if len(neighbors) <= self.m:
            return
        own = self.records[index].vector
        kept = sorted(
            neighbors,
            key=lambda j: (
                -similarity(own, self.records[j].vector, self.metric),
                self.records[j].id,
            ),
        )[: self.m]
        removed = neighbors - set(kept)
        self.graph[layer][index] = set(kept)
        for other in removed:
            # Keep graph edges symmetric after pruning.
            self.graph[layer][other].discard(index)

    def _insert(self, index: int) -> None:
        level = self._random_level()
        self.levels.append(level)
        for l in range(level + 1):
            self.graph[l][index] = set()
        if self.entrypoint is None:
            self.entrypoint = index
            self.max_level = level
            return
        assert self.entrypoint is not None
        entry = self.entrypoint
        counter = [0]
        vector = self.records[index].vector
        for layer in range(self.max_level, level, -1):
            entry = self._greedy(vector, entry, layer, counter)
        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(
                vector,
                [entry],
                self.ef_construction,
                layer,
                counter,
            )
            neighbors = [idx for _, idx in candidates if idx != index][: self.m]
            for neighbor in neighbors:
                self.graph[layer][index].add(neighbor)
                self.graph[layer][neighbor].add(index)
                self._prune(neighbor, layer)
            self._prune(index, layer)
            if candidates:
                entry = candidates[0][1]
        self.build_distance_evaluations += counter[0]
        if level > self.max_level:
            self.max_level = level
            self.entrypoint = index

    def search(
        self,
        vector: Sequence[float],
        *,
        tenant_id: str,
        filters: Mapping[str, str] | None = None,
        top_k: int = 5,
        ef_search: int = 32,
        filter_mode: str = "post",
        min_score: float | None = None,
    ) -> tuple[list[SearchHit], SearchStats]:
        validate_search(top_k, ef_search)
        query = validate_vector(vector)
        if len(query) != self.dimensions:
            raise ValueError("Query dimension mismatch")
        canonical = normalize_filters(filters)
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Trusted tenant scope required")
        if filter_mode not in ("post", "exact-filter"):
            raise ValueError("filter_mode must be post or exact-filter")
        if filter_mode == "exact-filter":
            # Important: this is exact pre-filtered KNN, NOT graph ANN.
            return self.exact.search(
                query,
                tenant_id=tenant_id,
                filters=canonical,
                top_k=top_k,
                min_score=min_score,
            )
        assert self.entrypoint is not None
        counter = [0]
        entry = self.entrypoint
        for layer in range(self.max_level, 0, -1):
            entry = self._greedy(query, entry, layer, counter)
        pool = self._search_layer(query, [entry], ef_search, 0, counter)
        # Demonstrates the post-filter shortfall: Top-K FIRST, filter AFTER.
        top_global = pool[:top_k]
        authorized = [
            (score, idx)
            for score, idx in top_global
            if matches(self.records[idx].metadata, tenant_id, canonical)
            and (min_score is None or score >= min_score)
        ]
        hits = [
            SearchHit(self.records[idx], score, rank)
            for rank, (score, idx) in enumerate(authorized, 1)
        ]
        return hits, SearchStats(
            distance_evaluations=counter[0],
            visited_nodes=counter[
                0
            ],  # upper-bound count; includes repeated calculations
            candidates_considered=len(pool),
        )

    def level_counts(self) -> dict[int, int]:
        return {
            layer: len(self.graph[layer]) for layer in sorted(self.graph, reverse=True)
        }
