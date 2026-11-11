"""An intentionally scripted RAG runtime.

Inputs: ONLY the script traces and document catalog, never evaluation gold.
This demonstrates evaluation of a full-stage trace without model/network cost.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from contracts import ContextRef, Trace

DATA = Path(__file__).resolve().parent / "data"


def parse_trace(row: dict, *, provenance: str) -> Trace:
    if not isinstance(row, dict):
        raise ValueError("Trace must be an object")
    return Trace(
        case_id=row["case_id"],
        tenant_id=row["tenant_id"],
        retrieved=tuple(row["retrieved"]),
        candidates=tuple(row["candidates"]),
        reranked=tuple(row["reranked"]),
        context=tuple(ContextRef(**e) for e in row["context"]),
        answer=row["answer"],
        abstained=row["abstained"],
        latency_ms=row["latency_ms"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        provenance=provenance,
    )


def load_scripted_traces(scenario: str = "baseline") -> dict[str, Trace]:
    main = json.loads((DATA / "observations.json").read_text(encoding="utf-8"))
    overlays = json.loads(
        (DATA / "scenario_overrides.json").read_text(encoding="utf-8")
    )
    if scenario not in overlays:
        raise ValueError(f"Unknown scripted scenario: {scenario}")
    traces = {}
    for base in main["cases"]:
        case_id = base["case_id"]
        if case_id in traces:
            raise ValueError(f"Duplicate scripted trace: {case_id}")
        merged = {**base, **overlays[scenario].get(case_id, {})}
        traces[case_id] = parse_trace(merged, provenance="fixture")
    for case_id in overlays[scenario]:
        if case_id not in traces:
            raise ValueError("Scenario references unknown trace")
    return traces


def load_external_traces(path: Path) -> dict[str, Trace]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("External trace input must be a JSON list")
    results = {}
    for row in raw:
        trace = parse_trace(row, provenance="external")
        if trace.case_id in results:
            raise ValueError("Duplicate case in external trace file")
        results[trace.case_id] = trace
    if not results:
        raise ValueError("No traces in external input")
    return results


def validate_trace(trace: Trace, case_id: str, tenant_id: str, documents: dict):
    """Fail on malformed traces, but REPORT unauthorized known documents."""
    if trace.case_id != case_id or trace.tenant_id != tenant_id:
        raise ValueError("Trace must match the requested case and trusted tenant")
    if type(trace.abstained) is not bool or not isinstance(trace.answer, str):
        raise ValueError("Invalid answer / abstention types")
    for value in (trace.latency_ms, trace.input_tokens, trace.output_tokens):
        if type(value) is not int or value < 0:
            raise ValueError("Token/latency usage must be nonnegative integers")
    for name in ("retrieved", "candidates", "reranked"):
        values = getattr(trace, name)
        if not all(isinstance(x, str) and x in documents for x in values):
            raise ValueError(f"Unknown document ID at {name}")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate document ID at {name}")
    if not set(trace.candidates).issubset(trace.retrieved):
        raise ValueError("Candidate set must be subset of retrieved")
    if not set(trace.reranked).issubset(trace.candidates):
        raise ValueError("Reranked set must be subset of candidates")
    evidence_ids = [ref.evidence_id for ref in trace.context]
    context_docs = [ref.doc_id for ref in trace.context]
    if len(set(evidence_ids)) != len(evidence_ids) or not all(
        isinstance(e, str) and e.startswith("E") and e[1:].isdigit() and int(e[1:]) > 0
        for e in evidence_ids
    ):
        raise ValueError("Invalid or duplicate evidence ID")
    if len(set(context_docs)) != len(context_docs):
        raise ValueError("Duplicate context document")
    if not set(context_docs).issubset(trace.reranked):
        raise ValueError("Context documents must have passed the reranker")


def replace_answer(
    trace: Trace,
    answer: str,
    *,
    abstained: bool,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
) -> Trace:
    return replace(
        trace,
        answer=answer,
        abstained=abstained,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        provenance="live_generation",
    )
