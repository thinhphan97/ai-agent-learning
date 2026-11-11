"""Deterministic query-intent analysis, with protected exact anchors."""
from __future__ import annotations

import re

from contracts import Analysis, Request


def protected_anchors(query: str) -> tuple[str, ...]:
    """Capture immutable explicit entities, periods, and technical error IDs."""
    anchors: list[str] = []
    for pattern in (
        r"\bQ[1-4][/\s-]*20\d{2}\b",
        r"\bQ[1-4]\b",
        r"\bERR_[A-Z0-9_]+\b",
        r"\bFPT\b",
        r"\bQwen3\b",
        r"\bvLLM\b",
        r"\bcustomer[_\s-]*id\s*=?\s*\d+\b",
        r"\bcustomer\s+\d+\b",
    ):
        for found in re.findall(pattern, query, flags=re.IGNORECASE):
            if found.casefold().startswith("customer"):
                digits = re.search(r"\d+", found)
                token = f"customer:{digits.group()}" if digits else ""
            else:
                token = re.sub(r"\s+", "", found).lower()
            if token not in anchors:
                anchors.append(token)
    return tuple(anchors)


def analyze(request: Request) -> Analysis:
    if not isinstance(request.question, str) or not request.question.strip():
        raise ValueError("Query cannot be empty")
    if len(request.question) > 4000:
        raise ValueError("Query too long for Lab 14")
    text = request.question.lower()
    if "fpt" in text and ("q1" in text or "q2" in text) and (
        "revenue" in text or "doanh thu" in text
    ):
        intent = "finance_compare"
        wanted = []
        if "q1" in text:
            wanted.append("q1_revenue")
        if "q2" in text:
            wanted.append("q2_revenue")
        if any(term in text for term in (
            "why", "reason", "explain", "cause", "driver",
            "nguyên nhân", "explanation", "change",
        )):
            wanted.append("drivers")
        aspects = tuple(wanted)
    elif any(t in text for t in ("refund", "hoàn tiền", "return policy")):
        intent = "refund_policy"
        wanted = []
        if any(t in text for t in ("deadline", "time", "how long", "thời hạn", "bao lâu")):
            wanted.append("deadline")
        if any(t in text for t in ("condition", "eligib", "điều kiện")):
            wanted.append("conditions")
        if any(t in text for t in ("exception", "exclusion", "ngoại lệ")):
            wanted.append("exceptions")
        aspects = tuple(wanted or ["deadline", "conditions", "exceptions"])
    elif any(t in text for t in ("paid order", "customer_id", "order revenue", "customer 1")):
        intent, aspects = "customer_revenue", ("revenue",)
    elif "err_auth_403" in text or ("admin" in text and "authorization" in text):
        intent, aspects = "authorization", ("diagnosis",)
    elif "vllm" in text or "qwen3" in text:
        intent, aspects = "technical_serving", ("mitigation",)
    else:
        intent, aspects = "general_docs", ("answer",)
    return Analysis(intent=intent, aspects=aspects, anchors=protected_anchors(request.question))


def validate_rewrite(original_query: str, rewritten_query: str) -> None:
    """Reject missing/new protected constraints; not a complete semantic guard."""
    if not isinstance(rewritten_query, str) or not rewritten_query.strip():
        raise ValueError("Rewrite must be nonempty")
    if len(rewritten_query) > 600:
        raise ValueError("Rewrite exceeds max length")
    original = set(protected_anchors(original_query))
    rewritten = set(protected_anchors(rewritten_query))
    if original != rewritten:
        raise ValueError(f"Query drift detected: anchors {original} -> {rewritten}")