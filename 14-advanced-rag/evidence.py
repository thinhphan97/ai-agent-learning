"""Conservative text-based evidence heuristics, context packing and grounded output.

The matching rules are deliberately narrow teaching fixtures, NOT semantic
entailment. Gold benchmark labels are independent and never imported here.
"""
from __future__ import annotations

import re

from contracts import Analysis, Budgets, Evidence, Request, SearchHit
from corpus import Corpus


def supports(aspect: str, text: str, question: str = "") -> bool:
    lower = text.casefold()
    year_match = re.search(r"\b20\d{2}\b", question)
    expected_year = year_match.group() if year_match else None
    if aspect in {"q1_revenue", "q2_revenue"}:
        quarter = "q1" if aspect == "q1_revenue" else "q2"
        pattern = (
            rf"\b{quarter}/(?:{expected_year if expected_year else '20[0-9]{2}'})"
            rf"\s+quarterly revenue of\s+[\d,]+\s+billion vnd"
        )
        return bool(re.search(pattern, lower))
    if aspect == "drivers":
        if expected_year and (
            f"q1/{expected_year}" not in lower or f"q2/{expected_year}" not in lower
        ):
            return False
        return all(t in lower for t in ("q1/", "q2/", "attributed by management"))
    if aspect == "deadline":
        return "refund" in lower and "30 calendar days" in lower
    if aspect == "conditions":
        return "refund" in lower and "unused" in lower and "proof of purchase" in lower
    if aspect == "exceptions":
        return "activated digital subscriptions" in lower and "non-refundable" in lower
    if aspect == "mitigation":
        return "max_model_len" in lower and "memory" in lower
    if aspect == "diagnosis":
        return "err_auth_403" in lower and "authorization" in lower
    if aspect == "revenue":
        match = re.search(
            r"customer(?:_id)?[=\s-]*(\d+)", question, re.IGNORECASE
        )
        if match and f"customer {match.group(1)}" not in lower:
            return False
        return "paid orders revenue total" in lower
    if aspect == "answer":
        return bool(lower.strip())
    return False


def detected_aspects(
    text: str, wanted: tuple[str, ...], question: str = ""
) -> tuple[str, ...]:
    return tuple(a for a in wanted if supports(a, text, question))


class EvidenceBuilder:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def build(
        self, request: Request, ranked: list[SearchHit],
        wanted: tuple[str, ...], budget: Budgets,
        *, coverage_aware: bool,
    ) -> tuple[list[Evidence], tuple[str, ...], list[str]]:
        order = list(ranked)
        if coverage_aware:
            selected: list[SearchHit] = []
            used: set[str] = set()
            for aspect in wanted:
                candidate = next(
                    (hit for hit in order
                     if hit.doc_id not in used
                     and supports(
                         aspect, self.corpus.verify(request, hit.doc_id).text,
                         request.question,
                     )),
                    None,
                )
                if candidate:
                    selected.append(candidate)
                    used.add(candidate.doc_id)
            order = selected + [hit for hit in order if hit.doc_id not in used]
        selected_evidence: list[Evidence] = []
        covered: set[str] = set()
        seen: set[str] = set()
        dropped: list[str] = []
        for hit in order:
            if hit.doc_id in seen:
                continue
            seen.add(hit.doc_id)
            if len(selected_evidence) >= budget.max_evidence:
                dropped.append(hit.doc_id)
                continue
            doc = self.corpus.verify(request, hit.doc_id)
            content = doc.text
            claims = detected_aspects(content, wanted, request.question)
            if not claims and coverage_aware:
                dropped.append(hit.doc_id)
                continue
            selected_evidence.append(Evidence(
                id=f"E{len(selected_evidence) + 1}",
                doc_id=doc.id,
                source_type=doc.source_type,
                aspect=claims[0] if claims else "unverified",
                content=content,
            ))
            covered.update(claims)
        missing = tuple(a for a in wanted if a not in covered)
        return selected_evidence, missing, dropped


def _revenue_value(text: str, quarter: str) -> int | None:
    match = re.search(
        rf"\b{quarter}/20\d{{2}}\s+quarterly revenue of\s+([\d,]+)\s+billion vnd",
        text, re.IGNORECASE,
    )
    return int(match.group(1).replace(",", "")) if match else None


def grounded_answer(
    analysis: Analysis, evidence: list[Evidence],
    missing: tuple[str, ...],
) -> dict[str, object]:
    lookup = {e.aspect: e for e in evidence if e.aspect != "unverified"}
    if not evidence or all(a in missing for a in analysis.aspects):
        return {
            "status": "abstained",
            "answer": "The available authorized evidence is insufficient to answer.",
            "calculation": None,
        }
    pieces: list[str] = []
    calculation = None
    if analysis.intent == "finance_compare":
        e1, e2 = lookup.get("q1_revenue"), lookup.get("q2_revenue")
        if e1:
            q1 = _revenue_value(e1.content, "Q1")
            if q1 is not None:
                pieces.append(f"Q1 revenue: {q1} billion VND [{e1.id}].")
        if e2:
            q2 = _revenue_value(e2.content, "Q2")
            if q2 is not None:
                pieces.append(f"Q2 revenue: {q2} billion VND [{e2.id}].")
        if e1 and e2:
            q1, q2 = _revenue_value(e1.content, "Q1"), _revenue_value(e2.content, "Q2")
            if q1 is not None and q2 is not None and q1 > 0:
                calculation = round((q2 - q1) / q1 * 100, 2)
                pieces.append(f"QoQ revenue growth: {calculation:.2f}% (calculated from [{e1.id}], [{e2.id}]).")
        reason = lookup.get("drivers")
        if reason:
            match = re.search(
                r"attributed by management to ([^.]+)\.", reason.content,
                re.IGNORECASE,
            )
            if match:
                pieces.append(f"Reported reasons: {match.group(1)} [{reason.id}].")
    else:
        for aspect in analysis.aspects:
            e = lookup.get(aspect)
            if e:
                pieces.append(f"{aspect}: {e.content} [{e.id}]")
    if missing:
        pieces.append(
            "Unsupported aspects: " + ", ".join(missing)
            + ". No unsupported conclusions are inferred."
        )
    if not pieces:
        return {
            "status": "abstained",
            "answer": "The available authorized evidence is insufficient to answer.",
            "calculation": None,
        }
    return {
        "status": "partial" if missing else "answered",
        "answer": " ".join(pieces),
        "calculation": calculation,
    }