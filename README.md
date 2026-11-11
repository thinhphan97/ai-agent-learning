# AI Agent Learning Workspace

A progressive, independently runnable Python learning lab for building an AI Agent Framework from first principles.

## Learning path

| Level | Folder | Focus |
| --- | --- | --- |
| 0 | [0-agent-loop-react](0-agent-loop-react/README.md) | ReAct fundamentals, observations, tool execution and termination |
| 1 | [1-multi-step-tool-calling](1-multi-step-tool-calling/README.md) | Chained tool calls, observation-based arguments, strict ordering |
| 2 | [2-error-recovery](2-error-recovery/README.md) | Structured error feedback, prerequisite policy, LLM-driven recovery and bounded retries |
| 3 | [3-agent-state-management](3-agent-state-management/README.md) | Explicit lifecycle, event trace, JSON checkpoint and safe pause/resume |
| 4 | [4-planning-replanning](4-planning-replanning/README.md) | Plan contracts, goal evaluation, strategic replanning, and bounded technical retry |
| 5 | [5-agent-memory](5-agent-memory/README.md) | Bounded short-term context, explicit long-term memory, SQLite persistence and scoped retrieval |
| 6 | [6-agent-evaluation](6-agent-evaluation/README.md) | Independent oracle, evidence-aware scoring, CI thresholds and regression reports |
| 7 | [7-agent-safety](7-agent-safety/README.md) | Runtime guardrails, prompt injection boundary, exact approvals, user scopes and redacted audit |
| 8 | [8-rag-fundamentals](8-rag-fundamentals/README.md) | Chunking, TF-IDF vector retrieval, optional dense embeddings, grounded generation and source evaluation |
| 9 | [9-rag-chunking](9-rag-chunking/README.md) | Twelve chunking strategy families, source-grounded metadata, per-method tutorials, and evidence benchmarks |
| 10 | [10-hybrid-retrieval](10-hybrid-retrieval/README.md) | BM25 inverted index, SQLite FTS5 persistence, metadata filters, dense retrieval, RRF, optional reranking and evaluation |
| 11 | [11-vector-search](11-vector-search/README.md) | Embedding metrics, Exact KNN, layered educational HNSW, Docker Compose Qdrant server/local SDK, ANN recall benchmarking |
| 12 | [12-reranking-context](12-reranking-context/README.md) | Contextual BM25/dense retrieval, Parent–Child expansion, Candidate Selection, MMR, Cross-Encoder reranking, evidence packing and gold coverage evaluation |
| 13 | [13-rag-evaluation](13-rag-evaluation/README.md) | Independent RAG evaluation: Precision/Recall/MRR/nDCG, evidence coverage, annotated claim faithfulness/correctness, quality gates, failure injection and report comparison |
| 14 | [14-advanced-rag](14-advanced-rag/README.md) | Query rewriting, multi-query/RRF, decomposition, authorized source routing, retrieval planning, targeted evidence recovery and held-out benchmark |

Upcoming lessons: Agentic RAG and integrated agent architecture.

## Workspace rules

1. Each numbered folder is a standalone lab with its own `agent.py`, tests and README. Never import code from another lab. Dependencies are shared via the **root** `requirements.txt`.
2. Exactly one shared `.env` lives at workspace root. Each lesson resolves it using `Path(__file__).resolve().parents[1] / ".env"`.
3. `.env` is gitignored; `.env.example` documents non-secret configuration. Never put API keys in source files or docs.
4. Default run is offline with a deterministic mock model. Set `LIVE_MODE=1` for real tool-calling LLMs.
5. Use OpenAI Python SDK for both OpenAI API and OpenAI-compatible vLLM. The selected model/endpoint must support Chat Completions tool calling.
6. All source code, tests and documents are in English; numbered folders define the learning progression.
7. Tests validate the implementation with offline mocks, not the accuracy of a remote LLM.

## External services and local development

A self-hosted service such as Qdrant must be started explicitly. The **Lab 11** Qdrant server uses its own Docker Compose file and named persistent volume (see the service setup and verification in 11-vector-search/README.md):

~~~bash
docker compose -f 11-vector-search/compose.yaml up -d
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:6333/healthz
.venv/bin/python 11-vector-search/agent.py --backend qdrant --index-only
.venv/bin/python 11-vector-search/agent.py --backend qdrant \
  --question "ERR_AUTH_403 admin access"
docker compose -f 11-vector-search/compose.yaml down
~~~

Conventions for future labs: any externally running self-hosted service gets a lab-local Compose definition, a persistent data strategy, startup/readiness commands, credentials described only in the **shared root** .env.example, and shutdown/reset steps in its lab README. Hosted LLM/embedding APIs require endpoint/credential/connectivity instructions, but no Compose file when no self-hosted service is supplied. SQLite, hnswlib, NumPy, and in-process clients do not need containers.

The Lab 11 Compose instance is **localhost-only and unauthenticated**. Never expose it to untrusted networks. For remote deployments, configure authentication, TLS and access controls.

## Setup (once for the entire workspace)

```bash
cd ~/lab/ai-agent-learning
test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Includes NumPy, hnswlib, qdrant-client and sentence-transformers for real vector backends.
# hnswlib may need a local C++ compiler; neural dependencies may be large.

# Preserve an existing configured .env file:
test -f .env || cp .env.example .env
```

Fill in `.env` before real LLM execution. No lab has its own `.env` or `requirements.txt`.

## Execute a lab

```bash
cd ~/lab/ai-agent-learning

python 0-agent-loop-react/agent.py
python 1-multi-step-tool-calling/agent.py

# Lab 2 demonstrates recovering from three different failures.
# --scenario overrides even an exported LIVE_MODE=1:
python 2-error-recovery/agent.py --scenario wrong-order
python 2-error-recovery/agent.py --scenario invalid-arguments
python 2-error-recovery/agent.py --scenario tool-failure

# Optional: real provider, uses shared root .env:
python 2-error-recovery/agent.py --live \
  --question "Calculate paid order revenue for customer_id=1."

# Force an actual tool failure once while using the real LLM:
python 2-error-recovery/agent.py --live --inject-tool-failure \
  --question "Calculate paid order revenue for customer_id=1."

# Replace the FIRST real LLM response with a wrong tool call;
# later turns still use the real LLM and receive the structured error:
python 2-error-recovery/agent.py --live --inject-llm-response wrong-order \
  --question "Calculate paid order revenue for customer_id=1."

# Lab 3: pause after a complete tool result; then resume in another process:
python 3-agent-state-management/agent.py --mock \
  --checkpoint 3-agent-state-management/checkpoints/practice.json \
  --pause-after-tools 1
python 3-agent-state-management/agent.py --mock \
  --checkpoint 3-agent-state-management/checkpoints/practice.json \
  --resume

# Lab 4: structurally valid plan, successful tools, WRONG outcome -> strategic replan:
python 4-planning-replanning/agent.py --scenario semantic-mismatch

# Compare with transient tool execution failure: bounded retry WITHOUT replan:
python 4-planning-replanning/agent.py --scenario tool-failure

# Real LLM: replace only its FIRST plan with a valid-but-wrong plan.
# All executor calls and the subsequent replanner response are from the real LLM:
python 4-planning-replanning/agent.py --live --inject-plan-fault

# Lab 5: bounded context vs durable memory, across sessions:
python 5-agent-memory/agent.py --mock --demo

# Persist a preference, then recall it from a NEW Python process:
python 5-agent-memory/agent.py --mock --user alice \\
  --ask "Remember that my preferred programming language is Python."
python 5-agent-memory/agent.py --mock --user alice \\
  --ask "What programming language do I prefer?"

# Optional: real LLM selects memory tools (shared root .env):
python 5-agent-memory/agent.py --live --user alice \\
  --ask "What programming language do I prefer?"

# Lab 6: deterministic baseline and deliberately wrong-metric candidate:
python 6-agent-evaluation/agent.py --mock
python 6-agent-evaluation/agent.py --scenario wrong-metric \
  --compare-with 6-agent-evaluation/reports/mock-normal/summary.json

# One real LLM call sequence evaluated against the offline oracle:
python 6-agent-evaluation/agent.py --live --case revenue_lan

# Lab 7: safe read, denied write, explicitly approved sandbox write:
python 7-agent-safety/agent.py --mock
python 7-agent-safety/agent.py --mock \
  --question "Set note for customer_id=1 to 'Call Friday'."
python 7-agent-safety/agent.py --mock \
  --question "Set note for customer_id=1 to 'Call Friday'." --approve

# Simulated malicious tool output; the runtime must reject the forged tool:
python 7-agent-safety/agent.py --scenario prompt-injection

# Real model, read-only customer tool:
python 7-agent-safety/agent.py --live --question "Show customer_id=1."

# Lab 8: RAG Fundamentals (offline, no new dependencies):
python 8-rag-fundamentals/agent.py --mock
python 8-rag-fundamentals/agent.py --mock --eval

# RAG with real LLM generation, still using offline TF-IDF retrieval:
python 8-rag-fundamentals/agent.py --live \
  --question "How much does expedited shipping cost?"

# Optional real dense embeddings; requires EMBEDDING_MODEL in root .env:
python 8-rag-fundamentals/agent.py --mock --retriever dense

# Lab 9: explore and compare all chunking strategy families (offline):
python 9-rag-chunking/agent.py --strategy all --limit 1
python 9-rag-chunking/agent.py --strategy structure --json
python 9-rag-chunking/agent.py --strategy parent-child --size 20 --overlap 3
python 9-rag-chunking/agent.py --benchmark

# Lab 10: BM25 / SQLite FTS5 / hybrid retrieval, offline by default:
python 10-hybrid-retrieval/agent.py --method bm25 --question "ERR_AUTH_403 admin error" --explain
python 10-hybrid-retrieval/agent.py --method sqlite --question "FPT revenue Q2" --period Q2/2025
python 10-hybrid-retrieval/agent.py --method hybrid --question "password reset links expire"
python 10-hybrid-retrieval/agent.py --benchmark

# Lab 11: Embedding, Exact KNN, ANN/HNSW (offline default):
python 11-vector-search/agent.py --geometry
python 11-vector-search/agent.py --backend exact --question "CUDA memory vLLM"
python 11-vector-search/agent.py --backend educational-hnsw --ef-search 24
python 11-vector-search/agent.py --benchmark

# Lab 12: Contextual Retrieval + Reranking + Context Construction (offline):
python 12-reranking-context/agent.py --show-context
python 12-reranking-context/agent.py --scenario vllm --show-context
python 12-reranking-context/agent.py --benchmark --compare

# Lab 13: offline RAG evaluation / explicit failures / baseline comparison:
python 13-rag-evaluation/agent.py --scenario baseline --explain
python 13-rag-evaluation/agent.py --scenario candidate-loss --no-save --explain
python 13-rag-evaluation/agent.py --judge none \
  --gates-file 13-rag-evaluation/data/retrieval_gates.json --no-save

# Lab 14: Advanced RAG planning and bounded evidence recovery (offline):
python 14-advanced-rag/agent.py --strategy planned --scenario delayed-drivers
python 14-advanced-rag/agent.py --strategy adaptive --scenario delayed-drivers --show-trace
python 14-advanced-rag/agent.py --benchmark --compare

```

## Run tests

```bash
python -m unittest discover -s 0-agent-loop-react/tests -v
python -m unittest discover -s 1-multi-step-tool-calling/tests -v
python -m unittest discover -s 2-error-recovery/tests -v
python -m unittest discover -s 3-agent-state-management/tests -v
python -m unittest discover -s 4-planning-replanning/tests -v
python -m unittest discover -s 5-agent-memory/tests -v
python -m unittest discover -s 6-agent-evaluation/tests -v
python -m unittest discover -s 7-agent-safety/tests -v
python -m unittest discover -s 8-rag-fundamentals/tests -v
python -m unittest discover -s 9-rag-chunking/tests -v
python -m unittest discover -s 10-hybrid-retrieval/tests -v
python -m unittest discover -s 11-vector-search/tests -v
python -m unittest discover -s 12-reranking-context/tests -v
python -m unittest discover -s 13-rag-evaluation/tests -v
python -m unittest discover -s 14-advanced-rag/tests -v
```

## Key distinctions

- **Lab 0:** Minimal loop with a model selecting actions from observations.
- **Lab 1:** Strict required tool order. A violation ends immediately.
- **Lab 2:** Data dependency policy replaces fixed iteration order; rejected actions return observations so the model can repair them. Error and iteration budgets prevent infinite loops.
- **Lab 3:** State transitions and sequenced events make execution inspectable; settled-boundary JSON checkpoints allow pause/resume without repeating already observed tools.
- **Lab 4:** A plan can be structurally valid while achieving the wrong goal. An independent ProgressEvaluator detects semantic mismatches and triggers a revised plan that reuses evidence. Transient tool failures use bounded retry instead of replanning.
- **Lab 5:** Recent complete turns live only in a bounded session window; selected, explicitly requested facts live in a user-scoped SQLite store and survive restarting the Python process.
- **Lab 6:** Runtime completion is not goal satisfaction. A separate evaluation harness scores exact answers, tool-grounded evidence, policy compliance and budgets, then compares deterministic baseline/candidate reports.
- **Lab 7:** The LLM proposes actions but cannot authorize them. A deterministic policy enforces allowlists, exact write approvals, user scope, bounded execution and evidence-verified final claims.
- **Lab 8:** RAG separates indexing, retrieval, context construction and generation; the offline TF-IDF baseline is lexical vector search, while true dense embeddings require an embedding model. Citations are checked against retrieved chunks.
- **Lab 9:** Compare 12 distinct chunking and context-representation strategies; keep source offsets and hierarchy links. Advanced strategies have explicit toy versus real-model modes and individual READMEs. The small evidence benchmark is not a production accuracy estimate.
- **Lab 10:** Full-query BM25 with postings, persistent SQLite FTS5 and canonical metadata filtering, plus hybrid RRF. Offline hashed vectors and lexical reranking are teaching surrogates, not genuine neural embeddings or Cross-Encoders; optional real providers require explicit selection.
- **Lab 11:** Exact KNN, vector metrics and ANN recall against exact neighbors. Pure-Python HNSW is an educational graph; hnswlib and Qdrant service/local adapters require explicit optional installs. Default embeddings are deterministic lexical hashing, not neural semantics.
- **Lab 12:** Contextual Retrieval is performed at indexing time (offline metadata or opt-in LLM). Candidate Selection and reranking remain separate from token-budgeted Context Construction. Curated aspect metadata is a teaching hint, gold evidence labels are separate; citation ID validity does not prove factual support. Optional genuine dense embeddings and local Cross-Encoder require explicit opt-in.
- **Lab 13:** Independent ground-truth evaluation of full RAG stage traces, with exact IR metrics, document-level evidence coverage and answer claim labels. Quality gates fail on missing judgments or tenant/version violations; a separate retrieval-only profile explicitly marks answer metrics unmeasured. Real generation and LLM Judge are opt-in, and real generation still uses scripted retrieval.
- **Lab 14:** Query transformations, decomposition, trusted docs/code/mock-SQL routing, typed dependency-aware retrieval plans, deterministic in-process BM25/RRF, bounded targeted recovery and independent gold-document coverage benchmark. Optional LLM only proposes search queries; it does not plan autonomously, search neural vectors or generate answers.

A `completed` status means the teaching runtime reached its completion condition, not that the LLM's final statement was independently verified. Lab 6 introduces independent outcome evaluation for its own standalone teaching agent; it does not yet evaluate the earlier labs' runtimes automatically.