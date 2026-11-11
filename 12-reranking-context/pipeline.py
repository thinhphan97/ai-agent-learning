"""Independent Lab 12 orchestrator and observable stage traces.

Any generated contextual note is an indexing aid, not authoritative evidence.
Gold labels are used by evaluation only, never by this pipeline.
"""
from __future__ import annotations

from dataclasses import asdict

from contextualization import contextualize
from context_builder import ContextBuilder, TikTokenCounter, WhitespaceTokenCounter
from corpus import Corpus
from retrieval import Retriever
from reranking import LexicalReranker, LocalCrossEncoder, rerank
from selection import select_candidates
from validation import aspect_coverage


class RAGPipeline:
    def __init__(
        self, *, contextualization: str = "metadata",
        reranker: str = "toy", model_path: str | None = None,
        tokenizer: str = "whitespace",
        dense_provider: str = "toy",
        tenant_id: str | None = None,
    ):
        if contextualization == "llm" and not tenant_id:
            raise ValueError("LLM contextualization requires trusted tenant_id at construction")
        self.tenant_id = tenant_id
        self.corpus = contextualize(
            Corpus.load(), strategy=contextualization, tenant_id=tenant_id
        )
        self.retriever = Retriever(self.corpus, dense_provider=dense_provider)
        self.dense_provider = dense_provider
        if reranker == "toy":
            self.reranker = LexicalReranker()
        elif reranker == "cross-encoder":
            if not model_path:
                raise ValueError("cross-encoder requires --model-path")
            self.reranker = LocalCrossEncoder(model_path)
        else:
            raise ValueError("Unknown reranker")
        if tokenizer == "whitespace":
            counter = WhitespaceTokenCounter()
        elif tokenizer == "tiktoken":
            counter = TikTokenCounter()
        else:
            raise ValueError("Unknown tokenizer")
        self.context_builder = ContextBuilder(self.corpus, counter)
        self.contextualization = contextualization
        self.reranker_name = reranker

    def run(
        self, question: str, *, tenant_id: str,
        retrieval_k: int = 50, candidate_k: int = 20,
        retrieval_method: str = "hybrid",
        candidate_strategy: str = "topk",
        context_strategy: str = "coverage",
        max_evidence: int = 5, token_budget: int = 240,
        expansion: str = "child",
        required_aspects: tuple[str, ...] = (),
    ) -> dict[str, object]:
        if self.tenant_id is not None and tenant_id != self.tenant_id:
            raise PermissionError("LLM-contextualized index is restricted to the selected tenant")
        if type(retrieval_k) is not int or type(candidate_k) is not int:
            raise ValueError("Retrieval and candidate budgets must be integers")
        if not 1 <= candidate_k <= retrieval_k <= 200:
            raise ValueError("Expected 1 <= candidate_k <= retrieval_k <= 200")
        if len(set(required_aspects)) != len(required_aspects):
            raise ValueError("Required aspects must be unique")
        if any(not a.strip() for a in required_aspects):
            raise ValueError("Empty aspect")
        retrieved = self.retriever.search(
            question, tenant_id=tenant_id, top_k=retrieval_k,
            mode=retrieval_method,
        )
        candidates = select_candidates(
            retrieved, budget=candidate_k,
            strategy=candidate_strategy,
            required_aspects=required_aspects,
        )
        ranked = rerank(question, candidates, self.reranker)
        context = self.context_builder.build(
            ranked, tenant_id=tenant_id,
            token_budget=token_budget, max_evidence=max_evidence,
            required_aspects=required_aspects,
            strategy=context_strategy,
            expansion=expansion,
        )
        return {
            "question": question,
            "tenant_id": tenant_id,
            "retrieval_method": retrieval_method,
            "contextualization": self.contextualization,
            "reranker": self.reranker_name,
            "dense_provider": self.dense_provider,
            "candidate_strategy": candidate_strategy,
            "context_strategy": context_strategy,
            "retrieved": [
                {"id": h.child.id, "parent": h.child.parent_id, "score": round(h.score, 6),
                 "rank": h.rank, "aspect_hint": h.child.aspect}
                for h in retrieved
            ],
            "candidates": [h.child.id for h in candidates],
            "reranked": [
                {"id": h.child.id, "retrieval_rank": h.retrieval_rank,
                 "rerank_rank": h.rank, "score": round(h.score, 6),
                 "aspect_hint": h.child.aspect}
                for h in ranked
            ],
            "context": asdict(context),
            "evidence_coverage_by_curated_metadata": round(
                aspect_coverage(context, required_aspects), 4
            ),
        }