# Lab 9 — RAG Chunking Strategies: From Fixed-size to RAPTOR

**Goal:** Build and compare **12 chunking/representation strategy families** with 13 runnable modes (sentence and paragraph are separately selectable), understand what each preserves, where it fails, and how to design an honest evaluation.

This is a **standalone** lesson, not an import of Lab 8. All code, tests and documentation are English. The normal mode is **offline, standard-library only**. The existing root `.env` and root `requirements.txt` are reused for explicit API-based features; no per-lab credentials or dependencies.

## 1. Key mental model

There are at least three distinct design questions:

1. **Where to cut the source?** Fixed-size, Recursive, Sentence/Paragraph, Markdown Structure, Semantic, Proposition, Tables and Code AST.
2. **What retrieval unit should be indexed?** The small child, an enriched passage, a table row, a code method, or both leaves and summaries.
3. **What context should reach the LLM?** A matched chunk, its larger parent, nearby sentences, or an evidence-traceable summary.

Parent–Child, Contextual Retrieval, Late Chunking and RAPTOR are not interchangeable "ways to split text". They solve **context retrieval, representation or synthesis** problems in addition to, or instead of, placing chunk boundaries.

A good chunk preserves the **minimum sufficient evidence** for the intended question without filling the retrieval context with noise. There is no universally best chunk size.

## 2. Individual README for each strategy

Every strategy has its own readable explanation, sample, exact runnable command, applicable use cases, algorithm, failure modes and evaluation criteria:

| #   | Strategy and guide                                                      | Code             | Best starting use case      |
| --- | ----------------------------------------------------------------------- | ---------------- | --------------------------- |
| 01  | [Fixed-size Sliding Window](guides/01-fixed-size/README.md)             | `basic.py`       | Low-cost control baseline   |
| 02  | [Recursive Separator Chunking](guides/02-recursive/README.md)           | `basic.py`       | General prose and documents |
| 03  | [Sentence & Paragraph Chunking](guides/03-sentence-paragraph/README.md) | `basic.py`       | FAQs and short facts        |
| 04  | [Structure-aware Sections](guides/04-structure-aware/README.md)         | `structure.py`   | Markdown, policies, SOP     |
| 05  | [Semantic Breakpoint Chunking](guides/05-semantic/README.md)            | `semantic.py`    | Transcript/topic shifts     |
| 06  | [Proposition / Atomic-fact Candidates](guides/06-proposition/README.md) | `proposition.py` | Factual questions           |
| 07  | [Parent–Child Hierarchy](guides/07-parent-child/README.md)              | `hierarchy.py`   | Clauses and exceptions      |
| 08  | [Contextual Enrichment](guides/08-contextual/README.md)                 | `contextual.py`  | Entity/time disambiguation  |
| 09  | [Layout / Table-aware](guides/09-table-layout/README.md)                | `layout.py`      | Markdown financial tables   |
| 10  | [Code-aware AST](guides/10-code-aware/README.md)                        | `code.py`        | Python repository search    |
| 11  | [Late Chunking](guides/11-late-chunking/README.md)                      | `late.py`        | Contextual word vectors     |
| 12  | [RAPTOR-style Multi-level Summaries](guides/12-raptor/README.md)        | `raptor.py`      | Multi-section questions     |

## 3. Repository layout

~~~text
9-rag-chunking/
├── README.md
├── agent.py                 # CLI: one/all/benchmark and JSON view
├── contracts.py             # Document, Chunk, ChunkSet with source links
├── shared.py                # Sentence/word spans, lexical helpers
├── basic.py                 # Fixed / recursive / sentence / paragraph
├── structure.py             # Heading paths, section parents
├── semantic.py              # Embedding-distance breakpoints
├── proposition.py           # Source-linked proposition candidates
├── hierarchy.py             # Child index + parent expansion
├── contextual.py            # Metadata prefix on indexed text
├── layout.py                # Markdown table rows with repeated headers
├── code.py                  # Python AST and class/function nodes
├── late.py                  # Contextual encode -> range pooling
├── raptor.py                # Greedy clusters, extractive/LLM summaries
├── providers.py             # Explicit toy, API and local-model adapters
├── runner.py                # Factoryless dispatch and per-strategy fixtures
├── benchmark.py             # Small lexical evidence-coverage checks
├── data/
│   ├── policies.md
│   ├── financial_report.md
│   ├── transcript.txt
│   └── order_service.py
├── guides/
│   ├── 01-fixed-size/README.md
│   ├── ...  (12 separate strategy READMEs)
│   └── 12-raptor/README.md
└── tests/
    ├── test_strategies.py
    ├── test_cli.py
    └── test_benchmark.py
~~~

The sample `financial_report.md` is synthetic; all entities and figures in it are examples.

## 4. Quick start — all strategies, no network

~~~bash
cd ~/lab/ai-agent-learning

# Execute every strategy against its appropriate synthetic fixture.
.venv/bin/python 9-rag-chunking/agent.py --strategy all --limit 1

# Inspect individual results.
.venv/bin/python 9-rag-chunking/agent.py --strategy fixed
.venv/bin/python 9-rag-chunking/agent.py --strategy recursive
.venv/bin/python 9-rag-chunking/agent.py --strategy sentence
.venv/bin/python 9-rag-chunking/agent.py --strategy paragraph
.venv/bin/python 9-rag-chunking/agent.py --strategy structure
.venv/bin/python 9-rag-chunking/agent.py --strategy semantic
.venv/bin/python 9-rag-chunking/agent.py --strategy proposition
.venv/bin/python 9-rag-chunking/agent.py --strategy parent-child
.venv/bin/python 9-rag-chunking/agent.py --strategy contextual
.venv/bin/python 9-rag-chunking/agent.py --strategy table
.venv/bin/python 9-rag-chunking/agent.py --strategy code
.venv/bin/python 9-rag-chunking/agent.py --strategy late
.venv/bin/python 9-rag-chunking/agent.py --strategy raptor
~~~

Add `--json` to inspect full `Chunk` metadata, parent node mappings and pooled vectors. If the JSON output is too long, use `--limit 2` without JSON.

## 5. Inputs and options

| Option                   | Meaning                                                        |
| ------------------------ | -------------------------------------------------------------- |
| `--strategy`             | One strategy or `all` (default)                                |
| `--file PATH`            | Explicit UTF-8 `.txt`, `.md` or `.py`; max 200 KB              |
| `--size 35`              | Word window/cap for most strategies; **line cap for Code AST** |
| `--overlap 5`            | Word overlap for Fixed, Recursive, Parent–Child and Late       |
| `--group-size 2`         | Count of sentences, paragraphs or table rows per chunk         |
| `--threshold 0.55`       | Cosine **distance** threshold for Semantic                     |
| `--provider toy`         | Offline, deterministic, does **not** use learned semantics     |
| `--provider api`         | **Explicit remote calls** for Semantic, Proposition or RAPTOR  |
| `--local-model-dir PATH` | Local HF Transformer for genuinely contextual Late Chunking    |
| `--benchmark`            | Tiny common-document lexical evidence coverage                 |
| `--json`                 | Export result as printed JSON                                  |

**Important:** `--size` is counted in **words** for text methods, not BPE tokens. The structure and table methods preserve source spans where meaningful. Generated representations (propositions, summary nodes, contextual prefixes) may not match a literal source substring; their provenance metadata must remain attached.

**Invalid combinations** fail explicitly instead of silently falling back.

## 6. Which methods are genuine, approximated or intentionally narrow?

| Strategy                             | Teaching implementation                                                | What is *not* claimed                                    |
| ------------------------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------- |
| Fixed, Recursive, Sentence/Paragraph | Actual word/separator chunkers                                         | Tokenizer-aware production NLP                           |
| Structure                            | Actual Markdown heading parser with recursive fallback                 | Full PDF/HTML layout parsing                             |
| Semantic + default provider          | Actual cosine-breakpoint algorithm using **toy lexical features**      | Neural semantic embedding accuracy                       |
| Semantic + `--provider api`          | Actual embedding-model distances                                       | Universal embedding endpoint availability                |
| Proposition + default                | Source-preserved sentence-level proposition **candidates**             | True atomicity and independent factuality                |
| Proposition + `--provider api`       | LLM-created independent candidates, source-linked                      | Automatically verified factual entailment                |
| Parent–Child                         | Actual section/child hierarchy and deduplicated parent expansion       | Cross-section reasoning without retrieval                |
| Contextual                           | Deterministic metadata/heading prefix                                  | LLM-generated contextual retrieval                       |
| Layout/Table                         | Real Markdown pipe-table structure, header/row links                   | PDF OCR, merged cells, multi-level headers               |
| Code-aware                           | Real Python AST class/function boundaries                              | Multi-language semantics or complete call graph          |
| Late default                         | Correct encode-full-before-pool **with toy context-mixing encoder**    | True pretrained Transformer representations              |
| Late + local HF                      | Real local Transformer word states then chunk pooling                  | Document length beyond encoder context                   |
| RAPTOR default                       | Greedy lexical pairing and **extractive** summaries in multiple levels | Original RAPTOR's semantic clustering/abstractive recipe |
| RAPTOR + `--provider api`            | LLM-generated summaries, still greedy lexical grouping                 | Factual correctness of LLM summaries                     |

This separation matters for interviews: implementing the shape of an algorithm does **not** establish the effectiveness of a learned model or a full reference architecture.

## 7. Real-model options (explicit opt-in)

Only the existing root `.env` is read. No secrets are printed.

**Semantic using a real embedding model:**

~~~bash
# Root .env should contain EMBEDDING_MODEL, and a compatible endpoint/key.
.venv/bin/python 9-rag-chunking/agent.py \
  --strategy semantic --provider api --threshold 0.55
~~~

**LLM proposition extraction or RAPTOR summaries:**

~~~bash
.venv/bin/python 9-rag-chunking/agent.py --strategy proposition --provider api
.venv/bin/python 9-rag-chunking/agent.py --strategy raptor --provider api
~~~

Proposition mode may make a separate LLM call **per sentence** and RAPTOR may make multiple summarization calls. Use only when you want to incur provider costs. Neither mode uses labels from the benchmark to rewrite source text.

**Real local-transformer Late Chunking:**

~~~bash
# OPTIONAL: install transformers and torch in the shared root environment.
# Requires a model directory already present on disk; NO auto-download.
.venv/bin/python 9-rag-chunking/agent.py \
  --strategy late --local-model-dir /path/to/local/hf-encoder
~~~

A standard remote embedding API that returns only a single vector per input is insufficient to implement true token-level Late Chunking. The local adapter requires a fast tokenizer, does not trust remote model code and refuses inputs beyond model positional length.

## 8. Compare chunking methods on a common document

~~~bash
.venv/bin/python 9-rag-chunking/agent.py --benchmark
.venv/bin/python 9-rag-chunking/agent.py --benchmark --json
~~~

The benchmark uses the same `data/policies.md`, four questions with exact required quotes, **lexical cosine retrieval**, top-K=3 and a 150-word context budget. It reports:

- Evidence coverage: did the selected passages include the exact required quote?
- Indexed chunk count and mean words per chunk.
- Which chunk IDs and context size were selected (JSON).

**Interpretation limits:** This is a smoke test of **retrieval mechanics and evidence preservation**, NOT a valid comparison of model quality. Fixed-size can beat more complex methods on this tiny dataset. Parent–Child is evaluated after parent expansion, RAPTOR includes indexable summary nodes, and toy semantic vectors do not model true semantics. A proper research benchmark must contain independent human-labeled retrieval cases of multiple types and evaluate answer faithfulness and cost.

For fairer comparisons, hold the embedder, retriever, reranker, corpus, top-K and/or context-token budget fixed; separately report both `Recall@K` and `Evidence Coverage` along with latency and index cost. Use a held-out question set to avoid tuning to the fixture.

## 9. Tests

~~~bash
cd ~/lab/ai-agent-learning
.venv/bin/python -m py_compile 9-rag-chunking/*.py
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
~~~

The tests cover boundaries, invalid sizes, source offsets, heading and parent links, table header mapping, Python AST symbols, fake embedding breakpoints, generated-proposition verification flags, true encode-before-pool order with a fake contextual encoder, RAPTOR leaf ancestry, benchmark constraints and offline CLI execution. None of the tests call a remote provider.

## 10. Recommended study sequence

First, compare **Fixed vs Recursive vs Structure-aware** with the same policies. Then examine how **Parent–Child** restores missing context, and how **Contextual** adds document/section identity. Compare **Sentence** with **Semantic** on the transcript, and understand why toy lexical features are not genuine semantics. Continue with Table/Code-aware, then analyze Proposition, Late Chunking and RAPTOR as advanced representations with distinct verification requirements.

For the next exercise, add a real, held-out corpus of policy/FAQ questions and implement **Recall@K + claim-level grounding**. Keep the source chunk mapping and budgets stable; choose a strategy from measured evidence rather than complexity alone.

**Safety:** The lab only reads explicit small local text/code files in offline mode. Remote APIs or local Transformers run only after selecting the corresponding explicit option. Documents and model-generated content are data, not instructions to execute.