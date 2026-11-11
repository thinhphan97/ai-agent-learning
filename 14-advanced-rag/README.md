# Lab 14 — Advanced RAG: Query Transformation, Routing and Retrieval Planning

Lab 14 is a standalone Python teaching lab for the **retrieval decision-making layer** of RAG. It imports no code from Labs 0–13. Source code, docs and tests are entirely in English, with one shared workspace-root requirements.txt and .env.

**Default mode: deterministic, offline, zero external model calls.** The lab implements in-process lexical BM25 and Reciprocal Rank Fusion (RRF); it does NOT include neural dense search, real SQL execution, persistent indexing, real-world business data or LLM-based answer generation. An optional OpenAI-compatible LLM may propose search queries when explicitly enabled.

## 1. Learning objectives

1. Understand when a single original query loses required evidence.
2. Distinguish rewriting, multiquery search, decomposition and HyDE hints.
3. Route queries among authorized document, code and mock SQL sources.
4. Plan retrieval steps and independently executable parallel batches.
5. Verify actual evidence coverage, rather than assuming relevance equals support.
6. Recover specific missing aspects within strict retry and call budgets.
7. Distinguish retrieval gaps from evidence/context packing gaps.
8. Benchmark strategies using gold labels inaccessible to the runtime.
9. Return a partial grounded answer or abstain when support is missing.

## 2. Pipeline architecture

~~~text
Trusted Request (question, tenant_id, as_of_date, allowed_sources)
   |
   v
Query Analyzer -> intent, required facets and protected anchors
   |
   v
Retrieval Planner -> typed steps, dependencies, parallel batches
   |
   +--> Query Transformer: identity / rewrite / multiquery / HyDE
   +--> Source Router: docs / code / mock SQL
   |
   v
Concurrent BM25 Retrieval (filter tenant/source/version BEFORE scoring)
   |
   v
RRF Fusion (deduplicate query results)
   |
   v
Evidence Validator -> Context Selection
   |
   +--> Complete ------------------------------> Grounded Answer
   +--> Context limit, evidence already found --> Partial Answer
   +--> Missing at retrieval, budget remains --> Targeted Retry
   +--> Missing, budget exhausted -------------> Partial / Abstain

Independent gold labels (evaluator ONLY) -> benchmark all strategies.
~~~

The most important distinction is **retrieval gap vs context budget gap**. When useful evidence is in retrieved hits but cannot fit in the final context, repeating retrieval is not an effective recovery action.

### Files

| Module | Responsibility |
| --- | --- |
| contracts.py | Immutable request, plan, hit, evidence and budget contracts |
| corpus.py | Original source storage, source allowlist, tenant and as-of-date checks |
| query_analysis.py | Query intent and facet extraction; protected-anchor checks |
| transform.py | Deterministic/optional LLM rewriting, multiquery and HyDE |
| routing.py | Authorized source choices independent of LLM output |
| planning.py | Typed retrieval DAG with dependency validation |
| retrieval.py | In-process BM25, deterministic RRF across multiple searches |
| evidence.py | Narrow original-text support heuristics and deterministic grounded answer |
| controller.py | Parallel retrieval, budget accounting, targeted recovery and event trace |
| evaluation.py | Held-out gold document and required-fact coverage evaluation |
| agent.py | CLI demo, JSON trace and strategy comparison |
| data/corpus.json | Synthetic source documents, no gold annotations |
| data/evaluation_cases.json | Fixed evaluation questions, trusted scopes and injected faults |
| data/gold_labels.json | Evaluator-only gold documents and aspect mappings |
| tests/ | Deterministic tests for correctness, drift, security, cost and recovery |

## 3. First run: financial comparison and dependency planning

~~~bash
cd ~/lab/ai-agent-learning

# Planned retrieval, without any injected faults:
.venv/bin/python 14-advanced-rag/agent.py --strategy planned --show-trace

# Controlled failure: the management commentary is hidden in round 1:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy planned --scenario delayed-drivers --show-trace

# Adaptive recovers ONLY the missing drivers in round 2:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy adaptive --scenario delayed-drivers --show-trace

# No drivers exist in any round. Do not invent an explanation:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy adaptive --scenario no-drivers --show-trace
~~~

The financial numbers are SYNTHETIC and do not describe actual FPT results:

~~~text
Q1/2025 revenue: 7000 billion VND
Q2/2025 revenue: 8400 billion VND
QoQ growth: (8400 - 7000) / 7000 * 100 = 20.00%

Mock management evidence:
The quarterly increase was attributed to overseas contracts and
increased delivery volume in Japan.
~~~

Q1 and Q2 searches are independent and execute as one concurrent batch. The calculation step depends on both search steps and only runs on verified quarter-specific numeric values. A missing reason cannot be inferred from the revenue values.

Expected controlled-failure sequence:

~~~text
Round 1: Q1 revenue + Q2 revenue + drivers retrieval
Calculated growth from source figures
Coverage check: missing drivers
  planned  -> Partial answer; no retry
  adaptive -> Targeted Round 2, drivers query only
Round 2: management commentary found
Coverage check: complete
Adaptive answer: grounded Q1/Q2 numbers, growth and cited reasons
~~~

The delayed-drivers scenario deliberately withholds the synthetic commentary document during Round 1. This is an EXPLICIT fault-injection test, not evidence of a production retriever's performance.

## 4. Seven strategies

~~~bash
.venv/bin/python 14-advanced-rag/agent.py --strategy baseline --show-trace
.venv/bin/python 14-advanced-rag/agent.py --strategy rewrite --show-trace
.venv/bin/python 14-advanced-rag/agent.py --strategy multiquery --show-trace
.venv/bin/python 14-advanced-rag/agent.py --strategy decompose --show-trace
.venv/bin/python 14-advanced-rag/agent.py --strategy planned --show-trace
.venv/bin/python 14-advanced-rag/agent.py --strategy adaptive --show-trace
.venv/bin/python 14-advanced-rag/agent.py --strategy hyde --show-trace
~~~

| Strategy | Behavior in this lab |
| --- | --- |
| baseline | One user query into authorized BM25 |
| rewrite | Original query with deterministic terminology expansion |
| multiquery | Three query variants, ranked and fused with RRF |
| decompose | Separate required aspects into independent queries |
| planned | Decomposition plus dependency-aware calculation step |
| adaptive | Planned retrieval plus bounded targeted evidence recovery |
| hyde | Deterministic hypothetical document-like search hint, never cited as evidence |

Multi-Query expands the **ways to ask a question**. Query Decomposition separates **the facts that must be collected**. HyDE generates search text, not authoritative statements.

This lab's default HyDE text is a deterministic **teaching substitute**, not a learned hypothetical document embedding. The optional LLM transform creates a model-generated hypothetical passage, but retrieval remains **BM25**, not a neural HyDE embedding system. Do not misrepresent it as the full published HyDE algorithm.

## 5. Source routing and authorization

~~~bash
# Technical manual
.venv/bin/python 14-advanced-rag/agent.py \
  --example vllm --strategy planned --sources docs

# Exact error code in a source-code snippet
.venv/bin/python 14-advanced-rag/agent.py \
  --example auth --strategy planned --sources code

# Synthetic read-only SQL aggregation snapshot (not actual SQL execution)
.venv/bin/python 14-advanced-rag/agent.py \
  --example sql --strategy planned --sources sql

# Expected authorization error, not a fallback to a different route
.venv/bin/python 14-advanced-rag/agent.py \
  --example sql --strategy planned --sources docs
~~~

The route returns docs, code or a PRECOMPUTED SQL SNAPSHOT. This lab does not implement a SQL engine or execute database queries. As-of-date, tenant and trusted source allowlist checks are enforced **before retrieval**, and again before evidence enters final context. LLM output can never grant a new source or tenant permission.

The CLI source/tenant flags are simulations of TRUSTED APPLICATION CONTEXT. In production, derive authorization from an authenticated principal, not arbitrary end-user strings.

## 6. Query drift protection and unsupported-answer handling

The query transformer preserves explicit protected anchors including company, quarter/year, exact error code and customer ID. This validation detects known entity/period drift, but does not prove full semantic equivalence.

~~~bash
# Request for 2026 must NOT be answered from 2025-only documents:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy planned \
  --question "Compare FPT international software revenue Q1 and Q2 2026"

# Customer 2 must NOT receive customer 1's source values:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy planned --sources sql \
  --question "What is paid order revenue for customer 2?"
~~~

Evidence validation is intentionally a **strict, corpus-specific heuristic**. It checks source excerpts for required phrases, year, customer ID and currency unit. It is NOT an NLI model or universal semantic fact checker. After selection, the deterministic answerer uses original source excerpts and evidence IDs such as [E1]; it never treats a generated HyDE hint as a source.

## 7. Budgets, stop reasons and controlled failures

~~~bash
# No second retrieval round:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy adaptive --scenario delayed-drivers --max-rounds 1

# Search-call budget exhausted after original plan:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy adaptive --scenario delayed-drivers --max-search-calls 3

# Already retrieved the facts, but context fits only one evidence block:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy adaptive --max-evidence 1 --show-trace
~~~

| Budget | Default | Meaning |
| --- | ---: | --- |
| max_subqueries | 5 | Total search steps, including recovery |
| max_rounds | 2 | Maximum retrieval rounds |
| max_search_calls | 12 | Total searches, including multiquery fan-out |
| max_hits_per_search | 5 | Retrieval Top-K for each query |
| max_evidence | 6 | Maximum selected final evidence passages |
| max_llm_calls | 3 | Optional model query-transform calls |

Budget preflight occurs before fan-out and before optional model calls. The controller cannot override these limits.

Possible stop reasons: sufficient, sufficient_after_recovery, no_adaptive_retry, context_budget_limited, no_new_evidence, max_rounds_exhausted, budget_exhausted. A missing supported aspect is always exposed in the trace. Lack of evidence leads to partial answer or abstention, never invented drivers.

## 8. Benchmark variants on the same held-out synthetic labels

~~~bash
# Compare all seven strategies across eight identical cases:
.venv/bin/python 14-advanced-rag/agent.py --benchmark --compare

# Get full per-case coverage, source IDs and search-call counts:
.venv/bin/python 14-advanced-rag/agent.py --benchmark --compare --json

# Focus on the recoverable missing-driver scenario:
.venv/bin/python 14-advanced-rag/agent.py \
  --benchmark --compare --case finance_delayed --json

# Save a local report without silently overwriting an old run:
.venv/bin/python 14-advanced-rag/agent.py \
  --benchmark --compare \
  --output 14-advanced-rag/reports/strategy-comparison.json
~~~

The eight fixture cases include a financial comparison, delayed and unavailable management commentary, a multi-aspect refund policy, vLLM docs, source-code auth handler, synthetic SQL snapshot, and an unanswerable question.

Gold labels are located ONLY in data/gold_labels.json, imported by evaluation.py, not the runtime controller. The benchmark measures:

- Gold Document Recall;
- Gold Required-Fact Coverage at retrieval and final context;
- Number of searches and retrieval rounds;
- Partial-answer / abstention count;
- Tenant and source isolation.

Document presence is only a **proxy**, not actual answer correctness, faithfulness or completeness. Lab 13 covers answer-quality evaluation on independently annotated claims. No real LLM quality or latency benchmark is claimed.

In this deterministic synthetic fixture, Adaptive Retrieval improves coverage on one recoverable evidence fault, at the cost of another search. Other variants may return identical scores. Do not assume that more queries or HyDE automatically improve every dataset.

## 9. Optional OpenAI-compatible LLM query proposals

Optional provider mode uses the EXISTING root .env and root requirements.txt, without a new per-lab environment:

~~~dotenv
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=YOUR_CHAT_MODEL
LLM_API_KEY=YOUR_PRIVATE_KEY
LLM_TIMEOUT=30
~~~

Explicitly enable potentially billable model calls:

~~~bash
# A single LLM-generated rewrite:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy rewrite --transform-provider llm

# Three LLM-proposed query alternatives from one call:
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy multiquery --transform-provider llm

# LLM-generated hypothetical document search text (NOT evidence):
.venv/bin/python 14-advanced-rag/agent.py \
  --strategy hyde --transform-provider llm
~~~

The model must return strict JSON with a queries array. Protected anchors are validated; malformed outputs fail closed rather than silently substituting unverified search text. Only user query/intent text is sent to the rewrite provider: no documents, tenants' private source text or gold labels are exposed by this adapter. User question egress still needs appropriate authorization.

LLM is only a **query proposal generator** in this lab. Retrieval is local BM25, answer generation is deterministic, and the planner itself is rule-based. This is not a trained Self-RAG model or an autonomous LLM planner.

## 10. External services, persistence, tests

No dedicated external service is needed, so NO Docker Compose file is introduced. Everything needed for offline tests uses the Python standard library. The only optional network capability is the pre-existing chat-completion endpoint.

~~~bash
cd ~/lab/ai-agent-learning
.venv/bin/python -m py_compile 14-advanced-rag/*.py
.venv/bin/python -m unittest discover -s 14-advanced-rag/tests -v

# Prior RAG regressions:
.venv/bin/python -m unittest discover -s 13-rag-evaluation/tests -q
.venv/bin/python -m unittest discover -s 12-reranking-context/tests -q
.venv/bin/python -m unittest discover -s 11-vector-search/tests -q
.venv/bin/python -m unittest discover -s 10-hybrid-retrieval/tests -q
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -q
.venv/bin/python -m unittest discover -s 8-rag-fundamentals/tests -q
~~~

All tests are offline and mock the optional model call; they never contact paid providers. Local JSON reports are optional and gitignored. For a future new lab-owned external service, provide a lab-local Compose file, health checks, persistence and explicit start/stop/reset instructions (see Lab 11).

## 11. Exercises

1. Why is decomposition better than multiquery paraphrases for Q1, Q2, growth and management commentary?
2. Identify the parallel batch in the event trace and why calculation waits for its dependencies.
3. Run Planned and Adaptive under delayed-drivers: compare evidence coverage and search budget.
4. Reduce max_evidence to 1 and explain why the recovery controller should NOT search again.
5. Force no-drivers and confirm the answer has a growth calculation but no invented causal claims.
6. Query a wrong year, customer or tenant and confirm no unauthorized evidence appears.
7. Explain why HyDE's hypothetical text is an embedding/search hint, never ground truth.
8. Run the benchmark and discuss the trade-off between additional search calls and context coverage.

**Next lesson:** Agentic RAG, including stateful reasoning, retrieval tool choice, evidence-based replanning and loop control.
