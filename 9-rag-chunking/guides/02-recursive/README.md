# Recursive Separator-based Chunking

**Family:** recursive | **Implementation:** `basic.py / recursive()` | **Mode:** offline, deterministic

[Return to Lab 9 overview](../../README.md)

## Why this strategy exists

Use as the preferred low-cost general-purpose baseline for prose, documentation, exported HTML text, articles and knowledge bases. It avoids many arbitrary mid-sentence cuts of fixed-size windows.

## Exact mechanism

First split oversized spans by paragraph breaks; oversized pieces are recursively split by lines, sentence-like delimiters, then whitespace, with a word-window fallback. The resulting subspans are greedily merged into bounded chunks. This uses **syntax-like separators**, not learned semantics.

**Worked example:** A section with two paragraphs is split at the blank line when it exceeds the word budget. If a paragraph remains oversized, it is split by lines/sentences before falling back to words. The implementation keeps source offsets after merging.

## Run it

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy recursive --size 35 --overlap 5
```

Source fixture: `data/policies.md`. To inspect the complete structured output, add `--json`. For this strategy, the implementation creates `Chunk` objects with `document_id`, `id`, `text`, optional source offsets and metadata.

## Parameters to tune

--size 35 --overlap 5. Inspect real max chunk lengths instead of assuming overlap does not affect them. For nested documents, prefer Structure-aware to keep heading paths.

## Failure modes / when NOT to choose it

Small sections might still be merged across logical topic boundaries. Sentence separator regex is simplistic for abbreviations and decimal numbers. In this teaching variant, optional look-back overlap can produce up to O additional words beyond the nominal size.

## Evaluate it

Compare fixed vs recursive on exact same corpus/retriever; examine incomplete sentences, boundary-fact misses, chunk-length distribution and Recall@K under equal context budget.

**Implementation boundary:** No language-aware sentence tokenizer or page/table understanding is included.

## Interview prompt

**Question:** Why choose Recursive Separator-based Chunking over arbitrary fixed-size windows?

**Answer guideline:** Explain the evidence unit selected by the algorithm, how this helps or harms retrieval, whether source boundaries are preserved, and what measurements would justify choosing it over the simpler baseline. Always distinguish retrieval accuracy from downstream grounded answer quality.

## Practical exercise

Run the command above, change the chunk size or grouping, inspect the first three chunk IDs and offsets, then run:

```bash
.venv/bin/python 9-rag-chunking/agent.py --benchmark
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
```

The benchmark is a tiny **lexical evidence-coverage proxy** on policy text. It is not a representative semantic retrieval benchmark, and the results for different document types cannot be compared solely by chunk counts.
