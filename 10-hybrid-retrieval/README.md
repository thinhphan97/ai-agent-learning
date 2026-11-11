# Lab 10 — BM25, SQLite FTS5, Hybrid Retrieval and Reranking

This standalone Python lab teaches lexical retrieval and its combination with dense retrieval, RRF and reranking. It does NOT generate answers or measure end-to-end RAG accuracy. All data is synthetic.

## Learning objectives

1. Search a full natural-language question without an LLM keyword extractor.
2. Trace analyzer tokens, TF, DF/IDF, document length normalization and BM25 scores.
3. Use a pure-Python inverted index and a persistent SQLite FTS5 database.
4. Use one canonical metadata schema for indexing and query filters.
5. Compare BM25, SQLite FTS5, hashed lexical pseudo-dense retrieval and RRF fusion.
6. Inspect a deterministic lexical reranker or opt into a genuine local Cross-Encoder.
7. Evaluate Recall@K, MRR@K and nDCG@K with independent gold labels.

## Architecture

~~~text
10-hybrid-retrieval/
├── agent.py              # CLI, diagnostics, benchmark
├── analyzer.py           # Tokenization, stopwords, limits
├── contracts.py          # Chunk, Hit, EvalCase
├── metadata.py           # Canonical metadata, filter allowlist
├── bm25.py               # Okapi BM25 and inverted index
├── storage.py            # Persistent SQLite FTS5 index
├── dense.py              # Explicit toy hashing / real API embeddings
├── fusion.py             # Reciprocal Rank Fusion
├── reranker.py           # Toy lexical / optional local Cross-Encoder
├── pipeline.py           # Scoped retrieval orchestration
├── evaluation.py         # Fixed-label retrieval metrics
├── fixtures.py           # Corpus and evaluation labels
├── data/corpus.json      # Synthetic multi-tenant fixtures
└── tests/                # Independent deterministic tests
~~~

Dependencies are shared at repository root, including the declared optional neural reranker packages. Default offline code requires only the Python standard library and SQLite compiled with FTS5. Each lab remains import-independent.

## Quick start

~~~bash
cd ~/lab/ai-agent-learning

# Inspect query tokens plus document frequency and IDF.
.venv/bin/python 10-hybrid-retrieval/agent.py \
  --method bm25 \
  --question "What causes ERR_AUTH_403 on the admin dashboard?" --explain

# SQLite FTS5 with canonical filters:
.venv/bin/python 10-hybrid-retrieval/agent.py \
  --method sqlite --question "FPT software revenue Q2" \
  --company "FPT Corporation" --period "Q2/2025"

# Hybrid SQLite FTS5 + toy hashed vectors + RRF + toy reranker.
.venv/bin/python 10-hybrid-retrieval/agent.py \
  --method hybrid --question "CUDA out of memory in vLLM" \
  --reranker lexical

# Fixed-label benchmark; no LLM or network.
.venv/bin/python 10-hybrid-retrieval/agent.py --benchmark
.venv/bin/python 10-hybrid-retrieval/agent.py --benchmark --json
~~~

### Why the full query works

BM25 is a ranking formula, not a keyword-selection LLM. The analyzer splits the question into terms, normalizes case and removes a small English stopword list. A query such as:

~~~text
How can I fix ERR_AUTH_403 when accessing the admin page?
~~~

is analyzed into terms such as:

~~~text
fix, err_auth_403, accessing, admin, page
~~~

The tokenizer preserves underscore-delimited technical identifiers. For Vietnamese production data, choose a Vietnamese-aware analyzer and test proper names, dates and identifiers. Avoid dropping negations such as "not" and "không".

### How BM25 works

The index stores an inverted posting list for each term and term frequency per document. The scorer visits only query-term postings rather than scoring every corpus document:

\[
\operatorname{IDF}(t)=\ln\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right)
\]

\[
\operatorname{BM25}(q,d)=\sum_{t\in q}\operatorname{IDF}(t)\,
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b|d|/avgdl)}
\]

The default educational parameters are k1=1.2 and b=0.75. The implementation deduplicates query terms, uses positive IDF and returns ranked scores. More occurrences help but TF gain saturates; rare terms receive higher IDF; document length is normalized.

The output BM25 score is NOT a probability. The algorithm has no learned semantic understanding and does not implement negation or exact date logic.

## SQLite FTS5 persistence

The database contains a structured chunks table (including original text and canonical metadata), plus a separate FTS5 inverted index over contextualized text. Updates/deletes to both occur within the same transaction.

~~~bash
# Run in one process to build a durable index.
.venv/bin/python 10-hybrid-retrieval/agent.py \
  --db 10-hybrid-retrieval/data/example.sqlite3 --index-only

# Run in another process and search the existing index.
.venv/bin/python 10-hybrid-retrieval/agent.py \
  --db 10-hybrid-retrieval/data/example.sqlite3 \
  --method sqlite --question "ERR_AUTH_403 admin"

# Explicitly update current fixture rows in an existing file.
.venv/bin/python 10-hybrid-retrieval/agent.py \
  --db 10-hybrid-retrieval/data/example.sqlite3 --reindex --index-only
~~~

SQLite FTS5 uses its own unicode61 tokenizer and BM25 implementation. Its BM25 scores generally are negative and LOWER means BETTER. The wrapper negates them to rank descending, but their magnitude is not comparable to Python BM25 or cosine. User text is analyzed and each term safely quoted into an OR-based FTS5 MATCH expression, which is then bound as a parameter; never build SQL identifiers or unvalidated MATCH syntax from user text.

Use chunk IDs consistently when updating separate lexical and dense indexes. Generated SQLite files are local artifacts and should not be committed.

## Canonical metadata and security

Both ingestion and query paths share metadata.py. Required indexing fields: tenant_id, company_id, period, segment, document_type, source. Query filters are restricted to company_id, period, segment, document_type. Alias examples:

~~~text
FPT Corporation, Công ty FPT -> FPT
Q2/2025                   -> 2025-Q2
Software Services, Phần mềm -> software
~~~

The tenant_id is NOT a user-query filter. The backend must derive it from authenticated authorization context and enforce it in BOTH retrieval branches, before fusion or reranking. The CLI tenant_id flag is only a simulation for a trusted application scope; do not expose it as a client-controlled field in a service.

Unknown filter fields fail closed. Zero retrieval results do NOT trigger automatic filter relaxation. This small lab accepts new company codes without a master registry; production systems need authoritative canonical IDs, migrations, ACLs, logging and index consistency checks.

There is no LLM Query Understanding step in this lab. Filters come from explicit CLI arguments. An LLM-based extractor may propose filters in a later lab, but backend validation and authorization must remain independent.

## Hybrid retrieval and RRF

~~~text
User question
  |                               |
SQLite FTS5 BM25             Embedding model
  |                               |
Filtered lexical Top-K       Filtered dense Top-K
  |                               |
  +---------- RRF fusion ---------+
                 |
         Optional reranking
                 |
              Top-N hits
~~~

Dense mode by default uses deterministic lexical feature hashing, held IN MEMORY. Its numerical vectors allow us to demonstrate indexing, cosine similarity and rank fusion, but they are NOT semantic neural embeddings and cannot validate semantic recall or cross-lingual similarity.

Hybrid mode combines SQLite FTS5 and the dense branch with RRF:

\[
\operatorname{RRF}(d)=\sum_{r}\frac{1}{60+\operatorname{rank}_r(d)}
\]

Raw BM25 and cosine scores are NOT added. Each chunk is merged by the same stable ID. A reranker only reorders the candidate pool; it cannot recover a missing chunk.

The --candidate-k flag controls retrieval candidates; --top-k controls final output.

## Optional genuine models (explicit opt-in)

A real OpenAI-compatible embedding endpoint can be enabled with the shared root .env setting EMBEDDING_MODEL, plus EMBEDDING_BASE_URL/EMBEDDING_API_KEY or compatible LLM endpoint variables:

~~~bash
.venv/bin/python 10-hybrid-retrieval/agent.py \
  --method hybrid --dense-provider api \
  --question "How can administrators recover account access?"
~~~

This opt-in makes network calls and may cost money. The small teaching CLI re-embeds the corpus on each run, unlike a production persistent dense vector index.

A real Cross-Encoder requires the optional sentence-transformers package and an ALREADY DOWNLOADED local model directory:

~~~bash
# Already declared in the shared root requirements.txt.
# Install from repository root: python -m pip install -r requirements.txt
.venv/bin/python 10-hybrid-retrieval/agent.py \
  --method hybrid --reranker cross-encoder \
  --cross-encoder-path /path/to/local/model \
  --question "When do password reset links expire?"
~~~

The default --reranker lexical is only a deterministic term-overlap scoring demonstration; it is NOT a neural Cross-Encoder. Real models depend on the language, domain, tokenizer and provider. Cross-Encoder scores may be logits rather than probabilities.

## Evaluation and interpretation

~~~bash
.venv/bin/python 10-hybrid-retrieval/agent.py --benchmark
.venv/bin/python 10-hybrid-retrieval/agent.py --benchmark --json
~~~

The fixed suite includes queries about company/period/segment, technical codes, GPU errors, refunds and access policies. Gold chunk IDs are specified separately from retrieval and never used to tune the index. Reports include Recall@K, MRR@K and nDCG@K for Python BM25, SQLite FTS5, toy dense, hybrid RRF and hybrid plus toy reranker.

This is a small POSITIVE-ONLY synthetic retrieval benchmark with one designated gold chunk per question. It measures neither answer correctness nor generalizable neural retrieval accuracy. To evaluate production RAG use held-out labels, paraphrases, exact identifiers, negative/no-answer queries, multi-document evidence, ACL failures, latency and cost.

## Tests

~~~bash
cd ~/lab/ai-agent-learning
.venv/bin/python -m py_compile 10-hybrid-retrieval/*.py
.venv/bin/python -m unittest discover -s 10-hybrid-retrieval/tests -v

# Backward compatibility:
.venv/bin/python -m unittest discover -s 8-rag-fundamentals/tests -v
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
~~~

Unit tests cover BM25 IDF/TF saturation, inverted postings, canonicalization aliases, tenant ACL simulation, bad filters and budgets, SQLite FTS5 persistence and transactions, injection-safe query parsing, vector dimensions, fusion rank math, candidate-limited reranking, metrics and CLI smoke behavior.

The essential distinction: Analyzer decides searchable terms; BM25 ranks lexical matches; metadata filters enforce structured constraints; embedding search provides a different retrieval signal; RRF combines rank positions; reranking prioritizes already-retrieved candidates.