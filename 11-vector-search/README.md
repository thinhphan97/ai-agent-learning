# Lab 11 — Embedding, Exact KNN, ANN/HNSW and Vector Database

This is a standalone, independently executable Python lab. It follows Lab 10's BM25 and hybrid retrieval lesson but imports **no code from other labs**. Default runtime execution requires no model downloads or external calls. The shared root requirements.txt now declares the optional real-backend packages; install that file to run all modes. All documents and numerical vectors are synthetic.

**Purpose:** understand and measure how vector search works rather than treating vector search as an opaque API.

## 1. Implemented components

| Component          | File                         | Function                                                        |
| ------------------ | ---------------------------- | --------------------------------------------------------------- |
| Vector metrics     | metrics.py                   | Real cosine, dot product, Euclidean distance and normalization  |
| Embeddings         | embeddings.py                | Toy lexical hashing and opt-in real embedding API               |
| Exact KNN          | exact.py                     | Brute-force similarity over all authorized vectors              |
| NumPy Exact KNN    | numpy_adapter.py             | Optional matrix-based exact KNN                                 |
| Educational HNSW   | hnsw.py                      | Actual layered graph + greedy descent + beam candidate search   |
| Real HNSW          | hnswlib_adapter.py           | Optional hnswlib adapter                                        |
| Persistent storage | storage.py                   | SQLite vector/metadata records with transactional CRUD          |
| Qdrant local       | qdrant_adapter.py            | Optional Qdrant SDK, collection and payload filtering           |
| Metadata           | metadata.py                  | Shared schema, normalization and tenant scope                   |
| ANN benchmark      | evaluation.py                | Exact KNN ground truth, ANN Recall@K, build/search timing       |
| CLI and fixtures   | agent.py, fixtures.py, data/ | Repeatable offline examples and diagnostics                     |
| Tests              | tests/                       | Deterministic tests; optional adapter tests when packages exist |

Code and documents are written in English. Dependencies and environment settings are shared with the repository root. Install them once using python -m pip install -r requirements.txt; hnswlib may require a C++ build toolchain on macOS.

## 2. Run the lab

~~~bash
cd ~/lab/ai-agent-learning

# Compare actual cosine, dot and L2 calculations in 2D.
.venv/bin/python 11-vector-search/agent.py --geometry

# Exact KNN: compare with every authorized vector.
.venv/bin/python 11-vector-search/agent.py --backend exact \
  --question "CUDA out of memory in vLLM"

# Multi-layer educational HNSW traversal.
.venv/bin/python 11-vector-search/agent.py --backend educational-hnsw \
  --ef-search 24 --question "CUDA out of memory in vLLM"

# Observe filtering mismatch between post-filtering and exact filtering.
.venv/bin/python 11-vector-search/agent.py \
  --backend educational-hnsw --filter-mode post \
  --company "FPT Corporation" --period "Q2/2025" \
  --question "FPT international software revenue Q2 2025"

.venv/bin/python 11-vector-search/agent.py \
  --backend educational-hnsw --filter-mode exact-filter \
  --company FPT --period 2025-Q2 \
  --question "FPT international software revenue Q2 2025"

# Compare approximate nearest neighbors to exact results.
.venv/bin/python 11-vector-search/agent.py --benchmark

# Structured trace for inspecting returned payload.
.venv/bin/python 11-vector-search/agent.py \
  --backend exact --question "ERR_AUTH_403 admin" --json
~~~

## 3. Embedding: text, vector, model versions

An embedding API normally accepts a string or a list of strings. The lab builds an input string from document/section metadata plus the original chunk and sends that to the embedder. The output is a numeric vector of fixed dimension.

Index and query embeddings must belong to compatible model spaces. Same dimensions alone are not sufficient. The lab stores an embedding_model ID with each record and rejects a mixture of model versions or mismatched query model. Changing embedding models normally requires re-embedding the corpus and rebuilding the index.

The default HashingEmbedder is deterministic **lexical feature hashing**; it is NOT a neural semantic embedding model and is not suitable for demonstrating paraphrase or multilingual semantic retrieval.

To explicitly opt in to a real API, configure EMBEDDING_MODEL and a compatible endpoint/key using the single workspace-root .env. EMBEDDING_BASE_URL and EMBEDDING_API_KEY override their LLM counterparts when present.

~~~bash
.venv/bin/python 11-vector-search/agent.py --backend exact \
  --embedding api \
  --question "How can an administrator restore account access?"
~~~

This mode makes actual external calls and may incur cost. The root .env is never copied into a lab folder or printed. Existing SQLite data indexed under a different model must be rebuilt using --reindex.

## 4. Three similarity metrics

Cosine compares directions: dot(a,b) / (norm(a)*norm(b)).

Dot product depends on both direction and magnitude.

Euclidean (L2) measures geometric distance; smaller means closer. For a uniform higher-is-better score contract, this teaching lab represents L2 as NEGATIVE SQUARED Euclidean distance. The displayed score is not a calibrated probability.

When all vectors including query are normalized to unit length, the identity below applies:

~~~text
squared_l2(a,b) = 2 - 2 * dot(a,b)
~~~

Thus cosine, dot and L2 yield the same ranking in that setting. When unnormalized they can disagree.

Try a query Q=(1,0) with A=(0.8,0.6), B=(0.9,0.1), C=(2,0). Cosine and dot prefer C; L2 prefers B. See --geometry for numeric scores and ordering.

## 5. Exact KNN: the reference

Exact KNN evaluates distance/similarity against every authorized candidate, then returns Top-K. Complexity of brute-force distance calculations is O(N*d) for N vectors with d dimensions. This is an excellent ground truth for the quality of an ANN index.

Pure Python exact.py emphasizes clarity. Optional NumPy exact is a real vectorized matrix operation, more suitable for benchmarking numerical work:

~~~bash
# Install all lab dependencies once: python -m pip install -r requirements.txt
.venv/bin/python 11-vector-search/agent.py \
  --backend numpy --metric cosine \
  --question "vLLM CUDA memory"
~~~

Exact KNN guarantees nearest neighbors under a chosen metric and permitted filter, not necessarily documents relevant to the user's natural language question.

## 6. HNSW: graph construction, layers, navigation

HNSW stands for Hierarchical Navigable Small World. It assigns vectors to probabilistic layers. Upper layers are sparse and help navigate toward a region. Lower layers are denser and support more detailed candidate exploration.

During insertion, the algorithm navigates existing layers, selects nearest candidate neighbors and creates/prunes graph connections. During query, it starts at an entry point, greedily navigates upper layers and searches the bottom layer using a bounded beam of candidates.

| Parameter       | Phase | Trade-off                                       |
| --------------- | ----- | ----------------------------------------------- |
| M               | Build | Neighborhood size and graph memory/connectivity |
| ef_construction | Build | Quality of neighbor selection vs indexing cost  |
| ef_search       | Query | Candidate exploration and recall vs latency     |
| top_k           | Query | Maximum returned results                        |

The pure-Python hnsw.py is a genuine layered graph implementation **for teaching**, not a faithful, optimized reproduction of every design choice in production HNSW. It uses a simple neighbor selection heuristic, may be poorly connected on difficult datasets and does not persist or dynamically mutate the graph. Its index is reconstructed from SQLite records on each process run.

To use a tested external HNSW engine, explicitly install hnswlib (and NumPy):

~~~bash
# hnswlib is declared in root requirements.txt.
.venv/bin/python 11-vector-search/agent.py --backend hnswlib \
  --ef-search 32 --question "GPU out of memory in vLLM"

.venv/bin/python 11-vector-search/agent.py \
  --backend hnswlib --benchmark
~~~

This invokes the real hnswlib graph index. It is optional and is never silently substituted in offline toy mode.

## 7. Metadata filters and zero results

Each record contains canonical metadata:

~~~text
tenant_id | company_id | period | section | source
~~~

Both ingestion and query use the same canonicalization logic:

~~~text
"FPT Corporation" -> "FPT"
"Q2/2025"         -> "2025-Q2"
~~~

User-query filters can only target company_id, period or section. tenant_id must come from trusted application context (the CLI flag simulates this context solely for training). The code refuses unknown filter fields and never broadens authorization on zero results.

In this lab the distinction is explicit:

**post:** ANN obtains the global Top-K first, then filters results. It can return fewer than K even when more valid vectors exist. This is a demonstration of a recall hazard, not a recommended tenant-isolation implementation for a multi-tenant service.

**exact-filter:** Filter by scope first and then perform Exact KNN on eligible vectors. It returns correct nearest neighbors but is **not** pre-filtered HNSW; the fallback is clearly labeled.

The optional Qdrant adapter passes tenant and metadata conditions to its query_filter API rather than performing client-side Top-K post-filtering.

## 8. Durable vector records and CRUD

SQLite stores original text, embedding vector JSON, embedding model ID, dimension and metadata. Upserts/deletes are transactional. Unlike a production Vector DB, this SQLite table is a source-record store; it does NOT persist the HNSW ANN graph itself.

~~~bash
# Persist the sample embeddings to disk.
.venv/bin/python 11-vector-search/agent.py \
  --db 11-vector-search/data/vectors.sqlite3 --index-only

# Query them from another process.
.venv/bin/python 11-vector-search/agent.py \
  --db 11-vector-search/data/vectors.sqlite3 --backend exact \
  --question "password recovery expiration"

# Practice insert, update with re-embedding, delete (ephemeral DB).
.venv/bin/python 11-vector-search/agent.py --crud-demo

# Recompute/upsert sample vectors when source or model changes.
.venv/bin/python 11-vector-search/agent.py \
  --db 11-vector-search/data/vectors.sqlite3 --reindex --index-only
~~~

Source text changes require updated embeddings; a stale vector can mislead retrieval even if payload text is correct. Production systems additionally need versioned dense index synchronization with their source metadata and lexical search indexes.

## 9. Qdrant service with Docker Compose

Qdrant is a separate database service. To connect to a real service, this lab owns a development Docker Compose definition at 11-vector-search/compose.yaml. The container runs REST on 6333 and gRPC on 6334, binds both ports to host loopback only, and stores data in a persistent named Docker volume.

**Prerequisites:** Docker Desktop / Docker Engine with Compose plugin, and the Python client installed from the shared root requirements.txt.

### Step 1 — Start the database

From the repository root:

~~~bash
cd ~/lab/ai-agent-learning
docker compose -f 11-vector-search/compose.yaml config
docker compose -f 11-vector-search/compose.yaml up -d
docker compose -f 11-vector-search/compose.yaml ps
~~~

The Compose image is pinned to qdrant/qdrant:v1.20.0 instead of latest. If ports 6333/6334 are already in use, stop the conflicting service or edit the Compose host-side ports and matching QDRANT_URL.

### Step 2 — Verify readiness

~~~bash
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:6333/healthz
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:6333/collections
docker compose -f 11-vector-search/compose.yaml logs --tail=80 qdrant
~~~

Qdrant dashboard: http://127.0.0.1:6333/dashboard

### Step 3 — Configure the client

Keep all shared environment values in the repository-root .env, never a per-lab .env file. The root .env.example includes:

~~~dotenv
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
# Only if your machine uses an HTTP proxy:
NO_PROXY=localhost,127.0.0.1
~~~

If an HTTP proxy is configured on your machine, the NO_PROXY setting is necessary for Python clients to reach the locally published service directly. Use curl --noproxy '*' when checking health.

The provided Compose instance has **no authentication or TLS**, strictly for local development. Ports are bound to 127.0.0.1 to avoid public exposure. Do not publish it to the internet or connect it to a public reverse proxy. For remote/team setups enable authentication, TLS and proper network restrictions. A protected Qdrant deployment can use QDRANT_API_KEY from the untracked root .env. You can also override the server URL with --qdrant-url.

### Step 4 — Install the client and build the collection

~~~bash
.venv/bin/python -m pip install -r requirements.txt

# Creates or updates only the lab11_documents collection.
.venv/bin/python 11-vector-search/agent.py \
  --backend qdrant --index-only
~~~

The explicit --index-only command creates/upserts the lab collection. An ordinary query does not auto-create a remote collection. Qdrant persists vectors and canonical payloads; the teaching CLI separately uses a small local SQLite source snapshot to verify source/model compatibility.

### Step 5 — Query with metadata filters

~~~bash
.venv/bin/python 11-vector-search/agent.py \
  --backend qdrant \
  --question "How much did FPT international software revenue grow in Q2 2025?" \
  --company "FPT Corporation" --period "Q2/2025" \
  --json

.venv/bin/python 11-vector-search/agent.py \
  --backend qdrant --qdrant-url http://127.0.0.1:6333 \
  --question "ERR_AUTH_403 admin access"
~~~

The tenant filter is applied INSIDE the Qdrant query, rather than through global Top-K post-filtering in Python. The CLI --tenant-id flag only simulates trusted application context. In a real service derive tenant scope from authentication/authorization.

When the source documents or embedding model change, deliberately reindex:

~~~bash
# DESTRUCTIVE reset of the dedicated lab11_documents collection only.
.venv/bin/python 11-vector-search/agent.py \
  --backend qdrant --reindex --index-only
~~~

The data uses reproducible lexical-hash vectors by default, NOT true neural semantic embeddings. For a real embedding model, set EMBEDDING_MODEL and its endpoint/key in the root .env, and use --embedding api for indexing AND querying. Production ingestion needs versioned embeddings, synchronization and migration checks.

### Step 6 — Stop, restart or reset

~~~bash
# Stop but retain the named persistent volume:
docker compose -f 11-vector-search/compose.yaml down

# Restart with the existing collections:
docker compose -f 11-vector-search/compose.yaml up -d

# DESTRUCTIVE: remove the Qdrant database volume and all its data:
docker compose -f 11-vector-search/compose.yaml down --volumes
~~~

This Compose file is for a single-node development database, not a hardened production cluster.

### Alternative — Qdrant SDK local mode without Docker

The previous in-process SDK backend remains available explicitly as qdrant-local:

~~~bash
.venv/bin/python 11-vector-search/agent.py \
  --backend qdrant-local --question "FPT software revenue Q2"

.venv/bin/python 11-vector-search/agent.py \
  --backend qdrant-local \
  --qdrant-path 11-vector-search/data/qdrant-local \
  --question "FPT software revenue Q2"
~~~

Qdrant local mode uses the real SDK but does NOT prove Qdrant server-side HNSW performance. To benchmark a real server, use the Docker service with more data and appropriate index configuration.

### External service conventions for future labs

Every self-hosted external service should have a lab-local Compose file, a health-check/smoke command, environment variables documented in the shared root .env.example, persistence configuration, and start/stop/reset instructions in that lab's README.

External managed APIs (such as hosted LLM/embedding endpoints) instead need an endpoint/configuration guide, credential handling, connectivity verification and cost/security notes. They cannot necessarily be self-hosted via Compose.

Pure-Python libraries, hnswlib, NumPy, SQLite and Qdrant in-process local mode are not separate background services and do not need Compose.

## 10. Evaluation: ANN Recall vs semantic relevance

~~~bash
.venv/bin/python 11-vector-search/agent.py --benchmark
.venv/bin/python 11-vector-search/agent.py \
  --benchmark --benchmark-size 100 --benchmark-dim 12 --top-k 3
.venv/bin/python 11-vector-search/agent.py \
  --backend hnswlib --benchmark
~~~

The benchmark generates seeded, clustered numerical vectors, builds exact and approximate indexes, and measures:

- ANN Recall@K against exact KNN IDs.
- Index build time, median search latency across a fixed query set.
- Candidate/distance evaluations where the implementation exposes them.
- Impact of ef_search values on the selected numerical corpus.

If exact top-5 is A,B,C,D,E and ANN top-5 is A,B,F,D,G, ANN Recall@5 is 3/5=0.6.

**Do not confuse this with RAG Retrieval Recall@K.** ANN Recall is about matching an exact vector-search result. RAG retrieval recall uses human-labeled relevant documents as the gold standard. Perfect ANN recall can still yield wrong answers when embeddings, corpus, chunking or retrieval policy are poor.

These tiny synthetic workloads are NOT reliable production throughput or latency estimates. A pure-Python HNSW can be slower than vectorized Exact KNN.

## 11. Tests and optional dependencies

~~~bash
cd ~/lab/ai-agent-learning
.venv/bin/python -m py_compile 11-vector-search/*.py
.venv/bin/python -m unittest discover -s 11-vector-search/tests -v

# Explicit real Qdrant Docker integration test (requires running service):
RUN_QDRANT_INTEGRATION=1 NO_PROXY=localhost,127.0.0.1 \
  .venv/bin/python -m unittest discover -s 11-vector-search/tests -v

# Regression:
.venv/bin/python -m unittest discover -s 10-hybrid-retrieval/tests -q
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -q
.venv/bin/python -m unittest discover -s 8-rag-fundamentals/tests -q
~~~

Optional library tests execute only when NumPy, hnswlib and qdrant-client are actually installed; these packages are now declared in the shared root requirements.txt. No test uses a remote LLM by default.

The lab checks similarity arithmetic, normalized-vector equivalence, exact KNN correctness, indexing/query schema compatibility, tenant scoping, post-filter false negatives, HNSW layer construction, persistent CRUD, deterministic ANN comparisons and CLI behavior.

## 12. Practice questions

1. Why does cosine give a different nearest neighbor from L2 for unnormalized vectors?
2. Why does exact vector search cost O(N*d), and why can NumPy be faster without changing the algorithm?
3. How do M, ef_construction and ef_search affect HNSW recall, build cost and memory?
4. Can a database have five semantically relevant documents but return fewer than five under Top-K=5?
5. Why are ANN Recall@K and RAG Retrieval Recall@K different?
6. Why must you re-embed documents when changing embedding models?
7. What is different between Qdrant local-mode payload filtering and client-side post-filtering?

Next: Cross-Encoder reranking, candidate recall and context construction for answer generation.