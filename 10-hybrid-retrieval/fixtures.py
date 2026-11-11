"""Small synthetic corpus and fixed independent retrieval labels."""

from __future__ import annotations

import json
from pathlib import Path

from contracts import Chunk, EvalCase
from metadata import canonical_chunk

CORPUS_PATH = Path(__file__).parent / "data" / "corpus.json"


def load_corpus() -> list[Chunk]:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    chunks = [
        canonical_chunk(entry["id"], entry["text"], entry["metadata"]) for entry in data
    ]
    if not chunks or len({c.id for c in chunks}) != len(chunks):
        raise ValueError("Empty corpus or duplicate chunk IDs")
    return chunks


# Labels are fixed independently of the retriever and not used for indexing.
CASES = (
    EvalCase(
        "How much did FPT international software revenue grow in Q2 2025?",
        ("fpt_sw_q2",),
        {"company_id": "FPT", "period": "Q2/2025", "segment": "software"},
    ),
    EvalCase(
        "FPT hardware revenue growth in Q2 2025?",
        ("fpt_hw_q2",),
        {"company_id": "FPT", "period": "2025-Q2", "segment": "hardware"},
    ),
    EvalCase(
        "What causes ERR_AUTH_403 on the admin dashboard?",
        ("auth_403",),
        {"document_type": "runbook"},
    ),
    EvalCase(
        "When do password reset links expire?",
        ("password_reset",),
        {"segment": "security"},
    ),
    EvalCase(
        "How do I fix CUDA out of memory when serving Qwen3 with vLLM?",
        ("vllm_oom",),
        {"segment": "infrastructure"},
    ),
    EvalCase(
        "How long can customers return unopened hardware?",
        ("returns_30",),
        {"segment": "commerce"},
    ),
    EvalCase(
        "FPT software Q1 2025 revenue growth?",
        ("fpt_sw_q1",),
        {"company_id": "FPT", "period": "2025-Q1"},
    ),
    EvalCase(
        "How much did HPG steel revenue grow in Q2 2025?",
        ("hpg_q2",),
        {"company_id": "HPG", "period": "2025-Q2"},
    ),
    EvalCase(
        "How many days do customers have to get a refund?",
        ("tenant_b_refund",),
        {"segment": "commerce"},
        tenant_id="tenant_b",
    ),
    EvalCase(
        "How is tensor parallelism configured on vLLM for Qwen3?",
        ("vllm_tensor_parallel",),
        {"segment": "infrastructure"},
    ),
    # Deliberate paraphrase with no exact matching key terms:
    # lexical BM25 and toy hashed vectors should expose their limitations.
    EvalCase(
        "What is the deadline for sending back a sealed device?",
        ("returns_30",),
        {"segment": "commerce"},
    ),
)
