# Lab 13 — RAG Evaluation, Quality Gates and Regression Harness

This is a **standalone** Python lab for evaluating RAG stage traces and grounded answers. It imports **no code from Labs 0–12**, shares the single workspace-root requirements.txt and .env, and needs **no external service or model calls in default mode**. All code, documents and tests are in English.

The benchmark is deliberately **small and synthetic**. Offline answer scores come from independently hand-annotated claims for the EXACT scripted answers. They are not claims about real-world model quality or automatically verified semantic truth.

## 1. Evaluation architecture

~~~text
                       INDEPENDENT EVALUATION ORACLE
                       ┌───────────────────────────────┐
                       │ Gold relevant documents       │
                       │ Gold required answer facts    │
                       │ Source versions and tenant    │
                       │ Human-labeled answer claims   │
                       └───────────────┬───────────────┘
                                       │
RAG EXECUTION TRACE                    │       METRICS
User Query → Retrieval IDs ────────────┼──→ Precision / Recall / MRR / nDCG
                   ↓                   │
              Candidate IDs ───────────┼──→ Candidate Evidence Coverage
                   ↓                   │
              Reranked IDs ────────────┼──→ Ranking/stage trace
                   ↓                   │
              Final Context IDs ───────┼──→ Context Coverage / Scope / Freshness
                   ↓                   │
              Answer + citations ──────┼──→ Faithfulness / Correctness
                                       │    Completeness / Relevancy
                                       │    Abstention / Citation quality
                                       ↓
                             Aggregation + Quality Gates
                                       ↓
                             Reports + Baseline Comparison
~~~

The scripted RAG runtime reads only observed stage traces; the evaluator separately loads gold labels. The generator must NEVER receive the gold reference answer, relevant-document labels or human judgments.

### Files

| File | Responsibility |
| --- | --- |
| agent.py | CLI entrypoint, fixtures, optional provider selection |
| contracts.py | Immutable documents, gold cases, observations, claims |
| dataset.py | Validate catalog, gold labels, effective dates and tenants |
| runtime_stub.py | Scripted traces, failure injection, external JSON import |
| retrieval_metrics.py | Precision@K, Recall@K, RR/MRR@K, graded nDCG@K |
| answer_metrics.py | Exact hand-labeled atomic claims and citation checks |
| evaluator.py | Stage metrics, answer scores, security checks |
| harness.py | Independent evaluation orchestration and aggregation |
| diagnostics.py | Identify the earliest observable evidence-loss stage |
| quality_gates.py | Full-RAG and retrieval-only gate policies |
| reporting.py | JSON/JSONL/Markdown reports and baseline deltas |
| llm_adapter.py | Optional real generation and LLM-as-a-Judge |
| data/documents.json | Synthetic multi-tenant document catalog |
| data/evaluation_cases.json | Independent gold documents and required facts |
| data/answer_labels.json | Hand-annotated scripted answer spans |
| data/observations.json | Baseline stage traces; NO gold labels |
| data/scenario_overrides.json | Deliberately broken pipeline outcomes |
| data/gates.json | Strict full-RAG gates |
| data/retrieval_gates.json | Separate retrieval-only gates |
| tests/ | Offline unit, CLI, gate and security tests |

## 2. Quick start

~~~bash
cd ~/lab/ai-agent-learning

# Offline, deterministic, six cases:
.venv/bin/python 13-rag-evaluation/agent.py --scenario baseline --explain

# Full intermediate traces and case scores:
.venv/bin/python 13-rag-evaluation/agent.py --json --no-save

# Only refund case, compare ranking at K=3:
.venv/bin/python 13-rag-evaluation/agent.py \
  --case refund_multi --k 3 --json --no-save
~~~

The baseline passes the strict gates **on the synthetic fixture**. By default, run artifacts are saved under 13-rag-evaluation/reports/RUN_ID/ (gitignored). Use --no-save to avoid persistence.

## 3. Retrieval metrics

With gold relevant docs A, B, C and a retrieved ranking A, X, B, Y, C:

| Metric | Definition | Worked example |
| --- | --- | --- |
| Precision@3 | 2 relevant items / 3 retrieved slots | 0.667 |
| Recall@3 | 2 retrieved gold items / 3 gold relevant docs | 0.667 |
| Recall@5 | 3 retrieved gold items / 3 gold relevant docs | 1.000 |
| RR@K | 1 / first relevant rank | 1.000 |
| MRR@K | Mean RR over answerable/eligible questions | Mean of RR |
| nDCG@K | DCG / ideal DCG, graded relevance 0–3 | Between 0 and 1 |

The lab uses graded gain and logarithmic rank discount:

~~~text
DCG@K = SUM((2**relevance_i - 1) / log2(i + 1)), i=1..K
nDCG@K = DCG@K / IDCG@K
~~~

**Precision always divides by K**, including empty result slots. **Recall divides by the TOTAL count of gold relevant items**, not the retrieval limit. When the gold relevant set is empty, Recall, RR and nDCG are **undefined (JSON null)** and excluded from their averages; this is not interpreted as 100% performance.

Evidence Coverage is a DIFFERENT metric: the fraction of independently labeled required facts with at least one supporting document still present after Retrieval, Candidate Selection or Context Construction. It is a **gold-document presence proxy**, not proof that a generated claim is semantically entailed.

## 4. Answer-level metrics

The independent annotation fixture identifies each **atomic claim span** in the scripted output, relevant source IDs, factual correctness and required facts addressed.

~~~text
Faithfulness      = Claims supported by actual FINAL context / All atomic claims
Claim Correctness = Claims annotated as factually correct / All atomic claims
Completeness      = Gold required facts addressed correctly / All required facts
~~~

Answer Relevancy is a **separate human-labeled rubric score** in this teaching fixture. Abstention Accuracy checks whether the system abstains on unanswerable questions and answers on answerable questions.

A model may correctly repeat an OLD policy: Faithfulness=1, Correctness=0. It may also emit a fabricated claim with an EXISTING citation [E1]. Citation-ID validity does not imply claim support. The evaluator therefore keeps citation-ID validation and claim-level citation support separate.

For a real model answer, the offline fixture labels are NEVER silently reused. Without separate exact human annotations or a consciously enabled LLM Judge, answer scores are unmeasured rather than fabricated.

## 5. Seven scripted scenarios

~~~bash
# Healthy baseline; expected exit code 0.
.venv/bin/python 13-rag-evaluation/agent.py --scenario baseline --no-save

# Required refund exception disappears before the Cross-Encoder; exit code 1.
.venv/bin/python 13-rag-evaluation/agent.py --scenario candidate-loss --explain --no-save

# Fabricated claim, even though citation E1 exists; exit code 1.
.venv/bin/python 13-rag-evaluation/agent.py --scenario hallucination --explain --no-save

# 45-day superseded policy is present, but actual current policy is 30; exit code 1.
.venv/bin/python 13-rag-evaluation/agent.py --scenario stale-evidence --explain --no-save

# Tenant B confidential source leaks into tenant A trace; exit code 1.
.venv/bin/python 13-rag-evaluation/agent.py --scenario tenant-leak --explain --no-save

# Injected source instruction is echoed by the mock answer; exit code 1.
.venv/bin/python 13-rag-evaluation/agent.py --scenario injection-followed --explain --no-save

# Model invents an ungrounded salary instead of abstaining; exit code 1.
.venv/bin/python 13-rag-evaluation/agent.py --scenario abstention-failure --explain --no-save
~~~

Exit codes: **0=PASS**, **1=quality gate FAIL**, **2=invalid input / provider error**. Intentional failures returning 1 are the EXPECTED demonstration behavior.

| Scenario | Earliest cause | Important observation |
| --- | --- | --- |
| Candidate loss | Candidate Selection | Faithfulness can remain high while Completeness drops |
| Hallucination | Generation | An existing citation ID can still cite the wrong fact |
| Stale evidence | Source version / context | Faithful to an old policy, incorrect as-of the question |
| Tenant leak | Authorization boundary | Security hard-fail even when normal averages look good |
| Injection followed | Source trust boundary | This is a HAND-ANNOTATED fixture, not a universal injection detector |
| Abstention failure | Generation policy | Missing evidence must not become a fabricated answer |

## 6. Quality gates and CI usage

The strict data/gates.json requires perfect score thresholds ONLY for these controlled synthetic fixtures, zero cross-tenant or stale-context documents, zero ungrounded claims/invalid citations, and complete answer evaluations.

Hard security violations **never average away**. Unknown metrics in full-RAG mode cause **FAIL**, not fake 100% results.

To score **only retrieval stages** without any answer judgment:

~~~bash
.venv/bin/python 13-rag-evaluation/agent.py \
  --judge none \
  --gates-file 13-rag-evaluation/data/retrieval_gates.json \
  --no-save --explain
~~~

This can PASS with an explicit **retrieval_only** scope and a list of skipped answer gates. Never treat it as proof of generation quality. Under the default full-RAG gate policy, --judge none intentionally FAILS if answer metrics are unmeasured.

In production, calibrate thresholds against held-out cases, measure uncertainty and treat security violations independently from aggregate scores.

## 7. Save a baseline and compare changes

~~~bash
# Baseline:
.venv/bin/python 13-rag-evaluation/agent.py \
  --scenario baseline --run-id my-baseline

# Deliberately degraded candidate (expect exit code 1):
.venv/bin/python 13-rag-evaluation/agent.py \
  --scenario candidate-loss --run-id my-candidate \
  --compare-with 13-rag-evaluation/reports/my-baseline/summary.json
~~~

For each run, inspect summary.json, cases.jsonl and report.md. The report includes stage IDs, case diagnosis, aggregate metrics and gate failures; comparison displays candidate minus baseline.

The comparator refuses mismatched dataset versions, K, selected cases or answer-evaluation mode. Run names are validated and existing report folders are not silently overwritten.

**Security/privacy:** The reports include the question text, observed answer and document IDs. Reports are gitignored, but they still require appropriate local access controls when real data is used. No API keys are written to reports. Synthetic fixture latency and token counts are labeled as illustrations, not measured model performance.

## 8. Evaluate traces from an independent RAG system

External input is a JSON ARRAY of stage observations, each following this schema:

~~~json
{
  "case_id": "refund_multi",
  "tenant_id": "tenant_a",
  "retrieved": ["refund_deadline_v2", "shipping", "refund_conditions", "refund_exceptions"],
  "candidates": ["refund_deadline_v2", "refund_conditions", "refund_exceptions"],
  "reranked": ["refund_conditions", "refund_deadline_v2", "refund_exceptions"],
  "context": [
    {"evidence_id": "E1", "doc_id": "refund_deadline_v2"},
    {"evidence_id": "E2", "doc_id": "refund_conditions"},
    {"evidence_id": "E3", "doc_id": "refund_exceptions"}
  ],
  "answer": "The refund deadline is 30 days [E1].",
  "abstained": false,
  "latency_ms": 350,
  "input_tokens": 150,
  "output_tokens": 16
}
~~~

Stage invariants: retrieved ⊇ candidates ⊇ reranked ⊇ context. IDs must exist in the document catalog. Known unauthorized source IDs are flagged as violations, not silently discarded. Source effective dates are checked at the question as-of date.

Build an import sample from observations.json:

~~~bash
.venv/bin/python -c '
import json
from pathlib import Path
src = Path("13-rag-evaluation/data/observations.json")
dst = Path("13-rag-evaluation/reports/example-external.json")
dst.parent.mkdir(exist_ok=True)
dst.write_text(json.dumps(json.loads(src.read_text())["cases"], indent=2))
'

.venv/bin/python 13-rag-evaluation/agent.py \
  --traces-file 13-rag-evaluation/reports/example-external.json \
  --judge none \
  --gates-file 13-rag-evaluation/data/retrieval_gates.json \
  --no-save
~~~

To use a real corpus, replace the synthetic data/documents.json and **independently labeled** data/evaluation_cases.json with matching document IDs and facts. Lab 12 uses different parent/child IDs, so integrating its trace requires an explicit ID mapping. Lab 13 never imports Lab 12 code.

### Independent human annotations for imported answers

Create a separate annotation file in the same shape as data/answer_labels.json, mapping exact observed answers to human-labeled atomic claims. A changed answer must have new labels.

~~~bash
.venv/bin/python 13-rag-evaluation/agent.py \
  --traces-file 13-rag-evaluation/reports/example-external.json \
  --annotations-file 13-rag-evaluation/data/answer_labels.json \
  --judge labels --no-save
~~~

The example works only because the imported answers are exactly the original SCRIPTED answers. Copying the model output and labeling it correct without independent review defeats the purpose of evaluation.

## 9. Optional REAL generation and LLM-as-a-Judge

The default lab uses the standard library and NO network or model. An OpenAI-compatible Chat Completions endpoint, including an already-running vLLM server, is used only when you explicitly select live generation or LLM Judge.

Configure the single root .env; no lab-local .env:

~~~dotenv
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=YOUR_GENERATOR_MODEL
LLM_API_KEY=YOUR_PRIVATE_KEY
LLM_TIMEOUT=30

# Optional separate evaluator (recommended for reducing self-evaluation bias)
EVAL_JUDGE_MODEL=YOUR_JUDGE_MODEL
EVAL_JUDGE_BASE_URL=
EVAL_JUDGE_API_KEY=
~~~

~~~bash
# One judge call, one scripted answer:
.venv/bin/python 13-rag-evaluation/agent.py \
  --case refund_multi --judge llm --no-save

# One generator call OVER SCRIPTED RETRIEVAL and one judge call:
.venv/bin/python 13-rag-evaluation/agent.py \
  --case refund_multi --live --judge llm --no-save

# One generator call only; default full-RAG gates fail without answer judgment:
.venv/bin/python 13-rag-evaluation/agent.py \
  --case refund_multi --live --judge none --no-save
~~~

Explicit limitations:

- The live switch exercises **real generation over scripted upstream retrieval**, NOT live end-to-end retrieval or Qdrant indexing.
- The generator sees only question, date, tenant and authorized context. No reference answers or gold labels are sent to it.
- LLM Judge sees reference facts but is FALLIBLE. Its result is tagged llm_judge_unverified, not a trusted semantic oracle.
- Provider modes require exactly ONE selected case, to bound accidental cost. Malformed JSON model output fails rather than silently substituting a mock response.
- An unauthorized context is blocked before provider egress; cross-tenant traces are not sent to a model judge.
- Source text is untrusted. A system prompt instructing the model to ignore injected instructions is not a security guarantee.

## 10. External services, persistence and reproducibility

No database or new self-hosted server is required by Lab 13, hence **no Docker Compose file is needed**. The optional managed API / pre-existing vLLM endpoint uses the existing root .env configuration. If a future lab introduces a lab-owned self-hosted service, add its own Compose file, health check, persistence, start/stop and reset instructions, as done for Lab 11 Qdrant.

The scripted corpus contains invented policies, synthetic security examples and fabricated latency/token usage. Do NOT interpret these as real business rules or measured service performance. Use independent held-out labels when evaluating real model or search changes.

## 11. Regression tests and exercises

~~~bash
cd ~/lab/ai-agent-learning
.venv/bin/python -m py_compile 13-rag-evaluation/*.py
.venv/bin/python -m unittest discover -s 13-rag-evaluation/tests -v

# Previous standalone RAG labs:
.venv/bin/python -m unittest discover -s 12-reranking-context/tests -q
.venv/bin/python -m unittest discover -s 11-vector-search/tests -q
.venv/bin/python -m unittest discover -s 10-hybrid-retrieval/tests -q
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -q
.venv/bin/python -m unittest discover -s 8-rag-fundamentals/tests -q
~~~

Suggested exercises: manually compute Recall@3 vs Precision@3; identify the first evidence-loss stage in candidate-loss; explain why stale evidence can have high faithfulness but low correctness; show how a fabricated claim can cite an existing evidence ID; demonstrate report comparison; import a new answer and verify that stale hand labels are rejected.

**Next planned lesson:** Advanced / Agentic RAG with retrieval planning, query routing and recovery, after mastering the evaluation gates here.
