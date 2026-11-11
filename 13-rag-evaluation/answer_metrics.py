"""Human-labeled synthetic claims scored deterministically.

Do NOT confuse citation ID existence with semantic support.
Labels are independent evaluation-only annotations and NOT runtime input.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from contracts import AnswerLabels, ClaimLabel, EvaluationCase, Trace

DATA = Path(__file__).resolve().parent / "data"
CITE_RE = re.compile(r"\[(E[1-9]\d*)\]")


def load_labels(path: Path = DATA / "answer_labels.json") -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "baseline" not in raw:
        raise ValueError("Labels must contain a baseline mapping")
    return raw


def select_labels(
    raw: dict, scenario: str, trace: Trace, *, allow_external: bool = False
) -> AnswerLabels | None:
    # External/live answers require an EXPLICIT separate human annotation file.
    if trace.provenance != "fixture" and not allow_external:
        return None
    item = raw.get(scenario, {}).get(trace.case_id)
    if item is None:
        item = raw["baseline"].get(trace.case_id)
    if item is None or item["answer"] != trace.answer:
        raise ValueError("Missing exact human annotation for scripted trace")
    return AnswerLabels(
        answer=item["answer"],
        claims=tuple(ClaimLabel(
            span=claim["span"],
            supporting_doc_ids=tuple(claim["supporting_doc_ids"]),
            correct=claim["correct"],
            covers_fact_ids=tuple(claim["covers_fact_ids"]),
        ) for claim in item["claims"]),
        injection_followed=item.get("injection_followed", False),
        answer_relevancy=item.get("answer_relevancy"),
    )


def score_answer(
    case: EvaluationCase,
    trace: Trace,
    labels: AnswerLabels | None,
) -> dict[str, object]:
    context_lookup = {ref.evidence_id: ref.doc_id for ref in trace.context}
    all_citations = list(dict.fromkeys(CITE_RE.findall(trace.answer)))
    unknown = [ref for ref in all_citations if ref not in context_lookup]
    abstention_correct = trace.abstained == (not case.answerable)
    result: dict[str, object] = {
        "evaluation_method": "unjudged" if labels is None else "hand_labeled_synthetic",
        "citation_ids": all_citations,
        "unknown_citation_ids": unknown,
        "citation_ids_valid": not unknown and (
            not case.answerable or bool(all_citations)
        ),
        "abstention_correct": abstention_correct,
        "faithfulness": None,
        "claim_correctness": None,
        "completeness": None,
        "answer_relevancy": None,  # Requires an independent semantic rubric.
        "claim_count": None,
        "unfaithful_claim_count": None,
        "invalid_citation_count": None,
        "injection_followed": None,
    }
    if labels is None:
        return result
    if labels.answer != trace.answer:
        raise ValueError("Oracle annotation does not match observed answer")
    if not isinstance(labels.injection_followed, bool):
        raise ValueError("Injection label must be boolean")
    if labels.answer_relevancy is not None and (
        type(labels.answer_relevancy) not in (int, float)
        or not 0 <= labels.answer_relevancy <= 1
    ):
        raise ValueError("Answer relevancy annotation must lie in [0,1]")
    if " ".join(c.span for c in labels.claims) != trace.answer and labels.claims:
        raise ValueError("Oracle claims must cover full answer exactly")
    if not labels.claims and not trace.abstained:
        raise ValueError("An asserted answer needs annotated atomic claims")
    gold_ids = {fact.id for fact in case.required_facts}
    observed_context = set(context_lookup.values())
    correct_claims = 0
    faithful_claims = 0
    covered_facts = set()
    invalid_citations = 0
    for claim in labels.claims:
        if type(claim.correct) is not bool:
            raise ValueError("Claim correctness annotation must be boolean")
        if not set(claim.covers_fact_ids).issubset(gold_ids):
            raise ValueError("Claim annotation references unknown gold fact")
        if claim.correct:
            correct_claims += 1
            covered_facts.update(claim.covers_fact_ids)
        if observed_context.intersection(claim.supporting_doc_ids):
            faithful_claims += 1
        citations = CITE_RE.findall(claim.span)
        cited_docs = {context_lookup[x] for x in citations if x in context_lookup}
        # Claim-level reference must exist AND the cited document must actually
        # be annotated as supporting the specific claim. Wrong citations fail.
        if not citations or not citations_exist(citations, context_lookup) or not cited_docs.intersection(claim.supporting_doc_ids):
            invalid_citations += 1

    count = len(labels.claims)
    if case.answerable:
        completeness = (len(covered_facts) / len(gold_ids)) if gold_ids else 0.0
    else:
        completeness = float(abstention_correct and count == 0)
    result.update({
        "faithfulness": faithful_claims / count if count else None,
        "claim_correctness": correct_claims / count if count else None,
        "completeness": completeness,
        "claim_count": count,
        "unfaithful_claim_count": count - faithful_claims,
        "invalid_citation_count": invalid_citations,
        "injection_followed": labels.injection_followed,
        "answer_relevancy": labels.answer_relevancy,
        "evaluation_method": (
            "hand_labeled_synthetic" if trace.provenance == "fixture"
            else "hand_labeled_external"
        ),
    })
    return result


def citations_exist(citations: list[str], lookup: dict[str, str]) -> bool:
    return all(ref in lookup for ref in citations)