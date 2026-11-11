# Sentence / Paragraph Chunking

**Family:** sentence | **Implementation:** `basic.py / sentence_or_paragraph()` | **Mode:** offline, deterministic

[Return to Lab 9 overview](../../README.md)

## Why this strategy exists

Good for short factual FAQ passages, support tickets, comments, reviews, transcripts with pre-separated turns and text where sentences are the natural evidence units.

## Exact mechanism

For sentence mode, identify sentence-like units and group a fixed number of complete units; for paragraph mode, split on blank lines and group complete paragraphs. Avoid cutting a unit midway. The grouping parameter controls how many units form a chunk.

**Worked example:** "Annual leave requires approval. Employees receive twelve days. Sick leave requires documentation." with group_size=2 becomes ["Annual leave requires approval. Employees receive twelve days.", "Sick leave requires documentation."]. Paragraph mode instead uses blank-line units.

## Run it

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy sentence --size 35 --overlap 5 --group-size 2
```

Source fixture: `data/policies.md`. To inspect the complete structured output, add `--json`. For this strategy, the implementation creates `Chunk` objects with `document_id`, `id`, `text`, optional source offsets and metadata.

## Parameters to tune

--strategy sentence --group-size 2 or --strategy paragraph --group-size 1. A production Sentence Window variant should index a small sentence and expand +/- neighboring sentences after retrieval, rather than increasing grouping globally.

## Failure modes / when NOT to choose it

A pronoun ("it") may require preceding sentences. Important exceptions can be separated from the main rule. A single paragraph can still be huge. Sentence regex is intentionally simplistic and treats Markdown headings as separate text units in some cases.

## Evaluate it

Ask fact questions and conditional/exception questions separately. Track Recall@K, answer completeness and how often a required condition lies outside the retrieved sentence group.

**Implementation boundary:** Sentence Window is explained but not implemented by this exact module; it can be built with source offsets and parent expansion like the hierarchy demo.

## Interview prompt

**Question:** Why choose Sentence / Paragraph Chunking over arbitrary fixed-size windows?

**Answer guideline:** Explain the evidence unit selected by the algorithm, how this helps or harms retrieval, whether source boundaries are preserved, and what measurements would justify choosing it over the simpler baseline. Always distinguish retrieval accuracy from downstream grounded answer quality.

## Practical exercise

Run the command above, change the chunk size or grouping, inspect the first three chunk IDs and offsets, then run:

```bash
.venv/bin/python 9-rag-chunking/agent.py --benchmark
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
```

The benchmark is a tiny **lexical evidence-coverage proxy** on policy text. It is not a representative semantic retrieval benchmark, and the results for different document types cannot be compared solely by chunk counts.
