"""Label-based retrieval-stage evaluation; labels never enter runtime ranking."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from pipeline import RAGPipeline

GOLD_FILE = Path(__file__).resolve().parent / "data" / "evaluation.json"


def _coverage(parents: set[str], labels: dict[str, list[str]]) -> float:
    if not labels:
        return 1.0
    return sum(bool(parents.intersection(ids)) for ids in labels.values()) / len(labels)


def evaluate(
    *, contextualization: str = "metadata",
    candidate_strategy: str = "topk",
    context_strategy: str = "coverage",
    retrieval_method: str = "hybrid",
    retrieval_k: int = 50,
    candidate_k: int = 20,
    max_evidence: int = 5,
    token_budget: int = 240,
) -> dict[str, object]:
    cases = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["cases"]
    pipeline = RAGPipeline(contextualization=contextualization)
    results = []
    for case in cases:
        result = pipeline.run(
            case["question"], tenant_id=case["tenant_id"],
            retrieval_method=retrieval_method,
            retrieval_k=retrieval_k, candidate_k=candidate_k,
            candidate_strategy=candidate_strategy,
            context_strategy=context_strategy,
            max_evidence=max_evidence, token_budget=token_budget,
            required_aspects=tuple(case["required_aspects"]),
        )
        labels = case["gold_aspect_parents"]
        retrieved = {h["parent"] for h in result["retrieved"]}
        candidate_ids = set(result["candidates"])
        selected_candidates = {
            h["parent"] for h in result["retrieved"]
            if h["id"] in candidate_ids
        }
        packed = {e["parent_id"] for e in result["context"]["evidence"]}
        # Coverage is assessed independently using gold parent IDs,
        # NOT aspect hints embedded in the runtime fixture.
        results.append({
            "case": case["id"],
            "retrieval_gold_coverage": round(_coverage(retrieved, labels), 4),
            "candidate_gold_coverage": round(_coverage(selected_candidates, labels), 4),
            "context_gold_coverage": round(_coverage(packed, labels), 4),
            "context_tokens": result["context"]["tokens_used"],
            "missing_metadata_aspects": result["context"]["missing_aspects"],
        })
    return {
        "benchmark": "fixed_synthetic_parent_evidence_labels",
        "contextualization": contextualization,
        "retrieval_method": retrieval_method,
        "candidate_strategy": candidate_strategy,
        "context_strategy": context_strategy,
        "cases": results,
        "mean_retrieval_gold_coverage": round(mean(
            c["retrieval_gold_coverage"] for c in results
        ), 4),
        "mean_candidate_gold_coverage": round(mean(
            c["candidate_gold_coverage"] for c in results
        ), 4),
        "mean_context_gold_coverage": round(mean(
            c["context_gold_coverage"] for c in results
        ), 4),
    }
