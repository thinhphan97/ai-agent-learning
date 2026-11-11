# RAPTOR-style Multilevel Retrieval

**Family:** raptor | **Implementation:** `raptor.py / raptor()`

[Return to Lab 9 overview](../../README.md)

## Why use this method?

Use when questions ask for conclusions spread across multiple document sections: long reports, books, research surveys, multi-topic policy analysis, and complex background questions. A single small leaf chunk rarely contains all the required supporting information.

## Algorithm

Create fixed-size leaf chunks, greedily pair lexical-nearest chunks and create summaries, then recursively group summaries into higher levels. Index both leaf chunks and summary nodes. Every summary stores child IDs and ultimate leaf IDs to support evidence backtracking. The offline summarizer extracts leading text from passages; it is intentionally lossy and **not genuine abstractive RAPTOR**.

**Illustrative example:** Leaf chunks describe revenue, customers, new markets, profit and risks. A level-1 summary condenses a pair; a level-2 summary condenses multiple related summaries. A broad strategic question may retrieve a high-level summary, while a specific numeric question should retrieve and cite original leaf evidence.

## Local run

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy raptor --size 20 --json
```

Example source: `data/policies.md`. Examine the printed chunk texts, source spans, parent links, metadata and any derived vectors with `--json`.

## Extension toward real applications

To use LLM-generated abstractive summaries, explicitly opt in:

```bash
.venv/bin/python 9-rag-chunking/agent.py \
  --strategy raptor --provider api
```

This requires the root LLM configuration and makes paid API calls. Generated summary nodes are marked requires_source_verification=true. The clustering here is deterministic greedy **lexical** pairing, not the original RAPTOR paper's semantic embedding/clustering recipe. A full RAPTOR system would replace this component with embedding clustering and benchmark summary loss versus source-traceable relevance.

## Weaknesses and boundaries

Each summary can omit critical exceptions, distort qualifiers or propagate earlier hallucinations; incremental document updates require re-summarizing affected ancestors. Index size and LLM preprocessing cost increase, and a summary is a navigation aid, not original evidence.

## How to evaluate it

Use multi-section synthesis questions, multi-hop evidence coverage, answer correctness, citation backtracking to source leaves, summary information-loss tests, latency and indexing costs. Compare against flat retrieval and ordinary Parent–Child expansion under equal context budgets.

**Production note:** This lab deliberately prioritizes transparent mechanics. The offline output is deterministic and fully testable, but does not establish production-grade parsing, embedding quality or LLM claim fidelity.

## Interview question

**Question:** Why is RAPTOR-style Multilevel Retrieval a better fit than simple fixed-size chunking for your chosen source?

**Answer pattern:** Describe which information relation must be preserved (row/column, syntax scope, prior context or document-level synthesis); how the implementation does so; what metadata is needed; the risk of introducing incorrect derived text; and which metric will demonstrate an actual improvement.

## Hands-on exercise

Change `--size` or `--group-size` where applicable. Compare output and trace at least one source fact to the original input. Try the local offline benchmark to understand its limitations:

```bash
.venv/bin/python 9-rag-chunking/agent.py --benchmark
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
```