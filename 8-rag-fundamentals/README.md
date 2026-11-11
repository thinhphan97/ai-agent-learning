# Lab 8 — RAG Fundamentals: Chunking, Retrieval and Grounded Answers

**Goal:** Understand and build a complete Retrieval-Augmented Generation pipeline, with separate modules and a reproducible, offline-first demo. Interview focus: Junior–Middle AI/LLM Engineer.

The lab is independent of Labs 0–7. It uses the shared root requirements.txt and .env, and all source code, tests and documentation are in English.

## 1. What is RAG?

RAG = **Retrieval-Augmented Generation**. At question time, the application searches external documents for relevant passages, then provides those passages to an LLM as additional context for its answer.

Example question:

> How many days can I return unopened hardware?

Our returns policy says customers can return unopened hardware **within 30 days after delivery**. The RAG pipeline searches the policy and generates a source-linked answer, rather than expecting the LLM to have memorized the company's policies.

RAG is different from fine-tuning: RAG injects knowledge **at inference time**; fine-tuning changes model parameters. RAG makes documents easier to update, but it does not eliminate hallucinations or guarantee correct retrieval.

## 2. Two separate phases

~~~mermaid
flowchart TD
  subgraph Indexing
    A[TXT documents] --> B[Load and clean]
    B --> C[Overlapping chunks]
    C --> D[TF-IDF vectors or dense embeddings]
    D --> E[(In-memory index)]
  end
  subgraph Query
    Q[User question] --> V[Query vector]
    V --> S[Cosine similarity and Top-K]
    E --> S
    S --> F[Relevant chunks]
    F --> G[Context budget]
    G --> H[Mock extraction or real LLM]
    H --> I[Citation validation]
    I --> J[Answer or Abstention]
  end
~~~

**Indexing** happens when the document collection is built or updated. **Query-time retrieval + generation** happens for every user question. This small teaching CLI rebuilds its index every run; a production system normally caches or persists it.

## 3. File layout

~~~text
8-rag-fundamentals/
├── README.md
├── agent.py                    # Minimal CLI, orchestration only
├── documents.py                # Bounded .txt loader and word chunking
├── retrieval.py                # TF-IDF, cosine, optional dense embeddings
├── generation.py               # Extractive offline generator / real LLM
├── pipeline.py                 # Retrieve, budget, generate, validate citations
├── evaluation.py               # 8 independent fixture questions
├── data/documents/
│   ├── billing.txt
│   ├── returns.txt
│   ├── security.txt
│   ├── shipping.txt
│   └── support.txt
└── tests/
    ├── test_documents.py
    ├── test_retrieval.py
    ├── test_pipeline.py
    └── test_cli.py
~~~

No local per-lab .env, no additional packages for offline mode, and no dependency on earlier lab code.

## 4. Exercise A — Run the full pipeline offline

~~~bash
cd ~/lab/ai-agent-learning
.venv/bin/python 8-rag-fundamentals/agent.py --mock
~~~

The demo loads five policy documents, creates overlapping chunks, builds a **TF-IDF vector index**, retrieves relevant chunks and extracts an answer from the retrieved text.

Expected answer includes **30 days** and a citation such as **[returns:0]**. The CLI prints the retrieved chunk ID, cosine score, title and word positions. Scores may differ if you change the corpus or chunk parameters.

**Important:** The offline generator is extractive and deterministic. It is NOT a mock semantic LLM. Separating retrieval from LLM generation makes it easier to debug each layer.

Try an unsupported question:

~~~bash
.venv/bin/python 8-rag-fundamentals/agent.py --mock \
  --question "What is the employee cafeteria menu?"
~~~

No relevant document exists, so the pipeline should abstain instead of inventing a cafeteria menu.

## 5. Exercise B — Chunking and overlap

Large documents exceed context limits and mix unrelated topics. **Chunking** divides them into smaller passages that the retriever can match.

This lab uses a sliding **word** window, not true model tokens. For chunk_size=10 and overlap=3:

~~~text
Chunk 0: word_0 ... word_7 word_8 word_9
Chunk 1:             word_7 word_8 word_9 word_10 ... word_16
Chunk 2:                                     word_14 ...
~~~

The overlap helps preserve information at chunk boundaries, at the cost of duplicated text and indexing work.

~~~bash
.venv/bin/python 8-rag-fundamentals/agent.py --mock \
  --chunk-size 12 --overlap 3
~~~

Compare the number of chunks and top-ranked results with the default (40 words, overlap 10).

**Interview tradeoffs:** Small chunks can break a complete fact into fragments. Large chunks introduce noisy, irrelevant content. Excess overlap creates duplicates. Production systems may use sentence-aware or token-aware chunking instead of this simple word-level baseline.

## 6. Exercise C — TF-IDF versus embeddings

Both backends represent text as vectors, but they are **not equivalent**.

| Dimension | TF-IDF baseline | Dense embedding model |
| --- | --- | --- |
| Meaning of vector | Word frequency weighted by rarity | Learned text representation |
| Matching | Primarily lexical word overlap | Can capture paraphrase and semantic similarity |
| Model inference | Not required | Required during indexing and query |
| External API | None | An embedding endpoint |
| Storage in this lesson | In-memory Python index | In-memory Python index |

TF-IDF computes weighted terms, normalizes each vector and compares them with cosine similarity:

\[
\cos(q,d)=\frac{q \cdot d}{\|q\|\|d\|}
\]

A similarity score is **not the probability that a passage is correct**. Top-K retrieves the highest ranked chunks subject to a threshold; a high-ranking chunk can still be irrelevant.

The default uses TF-IDF to keep every step observable. An optional genuine dense-embedding backend is available when your provider supports the Embeddings API.

Add an embedding model name to the **existing root .env**:

~~~dotenv
# Keep existing LLM_BASE_URL, LLM_MODEL, LLM_API_KEY.
EMBEDDING_MODEL=<your-provider-supported-embedding-model>
~~~

Run dense retrieval while keeping offline generation:

~~~bash
.venv/bin/python 8-rag-fundamentals/agent.py --mock \
  --retriever dense \
  --question "How much does fast delivery cost?"
~~~

**Note:** A vLLM deployment that only serves Chat Completions may not expose an embedding endpoint. Do not set EMBEDDING_MODEL to an ordinary chat model and assume it can generate embeddings. This lab uses the same base URL and API key for both APIs; separate embedding services would need an additional explicit adapter.

## 7. Exercise D — Use the real LLM to generate answers

Live generation reuses root .env and the existing OpenAI-compatible SDK:

~~~bash
.venv/bin/python 8-rag-fundamentals/agent.py --live \
  --question "How much does expedited shipping cost?"
~~~

This command uses **TF-IDF retrieval** and a **real chat LLM** for generation.

If the provider supports embeddings, both stages can be real:

~~~bash
.venv/bin/python 8-rag-fundamentals/agent.py \
  --live --retriever dense \
  --question "How much does expedited shipping cost?"
~~~

The generator prompt says to answer from retrieved excerpts, treat the excerpts as untrusted reference data, and cite only supplied source IDs. The runtime checks whether cited IDs exist among the retrieved chunks. It refuses answers with missing or invented citation IDs.

**Caution:** This check validates **citation IDs**, not whether the cited passage truly entails the claim. Real-world RAG needs a separate faithfulness evaluator and possibly claim verification.

## 8. Exercise E — Measure retrieval before judging the answer

The offline evaluation suite contains seven supported questions and one question that should be unanswered:

~~~bash
.venv/bin/python 8-rag-fundamentals/agent.py --mock --eval
~~~

It reports three teaching metrics:

- **Source Recall@K:** Did the relevant document appear in the retrieved top-K?
- **Answer check rate:** Did the extractive answer include the expected phrase and cite the expected document?
- **Negative abstention accuracy:** Did the pipeline abstain when none of the documents supported the question?

Tune retrieval:

~~~bash
.venv/bin/python 8-rag-fundamentals/agent.py \
  --mock --eval --top-k 1 --min-score 0.3

.venv/bin/python 8-rag-fundamentals/agent.py \
  --mock --eval --chunk-size 12 --overlap 2
~~~

A wrong answer with **low Recall@K** is most likely a retrieval/indexing problem. A wrong answer despite correct retrieved context is more likely a generation, interpretation, or citation problem. This separation is essential for practical RAG debugging.

The 8 cases are a tiny didactic dataset, not a representative benchmark. Phrase checks are weak proxies for semantic accuracy.

## 9. Debugging order for interviews

When asked "The answer is in the PDF, but RAG answers incorrectly," inspect these boundaries in order:

1. **Loading:** Was the file ingested, and is its extracted text complete?
2. **Chunking:** Does one chunk still contain the complete fact?
3. **Representation:** Are the embeddings appropriate for the language and domain? Would keyword matching outperform them?
4. **Retrieval:** Does the relevant chunk appear in top-K? Check filters, scores and thresholds.
5. **Context construction:** Was the chunk dropped due to a context/token budget?
6. **Generation:** Did the model use the context or hallucinate?
7. **Grounding:** Do citations truly support each claim?

Common improvements: hybrid BM25 + vector search, metadata filtering, reranking, query rewriting and better document parsing. These are follow-up lessons, not prerequisites to operate this baseline.

## 10. Quick interview answers

**Q: Why RAG instead of fine-tuning?**
RAG is usually preferable when facts change frequently or are in private documents; it retrieves current evidence at inference time. Fine-tuning changes model parameters and is more appropriate for behavior, patterns or domain adaptation. The approaches can be combined.

**Q: What is the impact of chunk size?**
Small chunks can lose context, while large chunks lower precision and increase token cost. Tune chunking based on Recall@K and actual downstream answer quality.

**Q: What is a vector database?**
A system that stores embeddings with metadata and supports similarity search, often via an approximate nearest-neighbor index. This lab uses an in-memory index for teaching the same retrieval concepts; it is not a persistent vector DB.

**Q: What is reranking?**
A second, often more expensive model scores the relevance of candidates retrieved by a cheaper first-stage search and reorders them.

**Q: Does a citation guarantee correctness?**
No. A citation can point to a real passage that does not support the generated claim. Evaluate **retrieval relevance**, **faithfulness**, and **answer correctness** separately.

## 11. Testing

~~~bash
cd ~/lab/ai-agent-learning
.venv/bin/python -m unittest discover \
  -s 8-rag-fundamentals/tests -v
~~~

Unit tests cover deterministic loading/chunking, overlap and boundaries, lexical ranking, fake dense embeddings and vector validation, citation rejection, context budget behavior, unsupported questions, evaluation contracts, and CLI behavior without remote calls.

## 12. Deliberate limitations and next lessons

This lesson only accepts small UTF-8 .txt files; it does not parse PDFs, handle OCR, enforce tenant document permissions, persist an ANN index, cache embeddings, rerank candidates, measure token cost or guarantee semantic answer faithfulness. The chunker uses words, not model tokenizer IDs. Real dense embeddings require an appropriately configured provider and may incur API costs.

**Next:** RAG Lesson 02 — Embeddings, Vector DB, Hybrid Search and Reranking; then RAG Lesson 03 — RAG Evaluation & Agentic RAG. Before moving on, you should be able to explain every stage and identify which stage caused an incorrect answer.
