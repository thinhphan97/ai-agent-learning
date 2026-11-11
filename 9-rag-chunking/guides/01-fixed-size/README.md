# Fixed-size Sliding Window

**Family:** fixed | **Implementation:** `basic.py / fixed()` | **Mode:** offline, deterministic

[Return to Lab 9 overview](../../README.md)

## Why this strategy exists

Use as the cheapest reproducible baseline, for homogeneous notes and text without meaningful headings. It is also useful as a control group when evaluating more sophisticated methods.

## Exact mechanism

The input is tokenized here by whitespace into **words**, not model tokens. For chunk size C and overlap O, the next window starts at stride C-O. Each output chunk retains its original character offsets and its [start_word, end_word) index range. Fixed boundaries can split sentences, conditions, units or table structure.

**Worked example:** For C=5, O=2, a 12-word document produces ranges [0,5), [3,8), [6,11), and [9,12). Each later chunk shares two words with the preceding chunk. There is no empty trailing chunk.

## Run it

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy fixed --size 35 --overlap 5
```

Source fixture: `data/policies.md`. To inspect the complete structured output, add `--json`. For this strategy, the implementation creates `Chunk` objects with `document_id`, `id`, `text`, optional source offsets and metadata.

## Parameters to tune

--size 35 --overlap 5 (word counts). Require 2 <= size <= 4096 and 0 <= overlap < size. For actual embedding-model limits, replace word counting with a tokenizer-aware window.

## Failure modes / when NOT to choose it

A qualification in the next sentence may be excluded from a retrieved chunk. Increasing overlap can help but creates duplicate index vectors and repetitive top-K hits. It cannot reconstruct a corrupted PDF table.

## Evaluate it

Measure evidence coverage for facts near boundaries, duplicate fraction (indexed words / source words), recall at a fixed context-word budget, and average chunk length. Repeat with several sizes; do not choose one by intuition.

**Implementation boundary:** Whitespace words differ from BPE tokens. Keeping character offsets helps link retrieved text back to the source; this is not sentence-preserving chunking.

## Interview prompt

**Question:** Why choose Fixed-size Sliding Window over arbitrary fixed-size windows?

**Answer guideline:** Explain the evidence unit selected by the algorithm, how this helps or harms retrieval, whether source boundaries are preserved, and what measurements would justify choosing it over the simpler baseline. Always distinguish retrieval accuracy from downstream grounded answer quality.

## Practical exercise

Run the command above, change the chunk size or grouping, inspect the first three chunk IDs and offsets, then run:

```bash
.venv/bin/python 9-rag-chunking/agent.py --benchmark
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
```

The benchmark is a tiny **lexical evidence-coverage proxy** on policy text. It is not a representative semantic retrieval benchmark, and the results for different document types cannot be compared solely by chunk counts.
