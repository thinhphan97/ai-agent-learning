# Semantic Breakpoint Chunking

**Family:** semantic | **Implementation:** `semantic.py / semantic()`

[Return to Lab 9 overview](../../README.md)

## When to choose this strategy

Use this when topic shifts are not reliably marked by headings or blank lines: long conversations, research notes, meeting transcripts and free-form essays. Instead of fixed boundaries, the algorithm asks whether adjacent sentence embeddings represent sufficiently different content.

## Algorithm and data flow

Sentence-like spans are extracted and embedded in a **single batch**. The algorithm calculates `distance = 1 - cosine(vector[i-1], vector[i])`. When distance crosses `--threshold` and `min_words` is satisfied, it flushes the current chunk; `max_words` prevents unbounded sections. The output includes source offsets and the boundary distance for diagnosis.

**Worked example:** The first three speaker turns discuss leave requests; later turns discuss account security. A semantic model should place a breakpoint near the change from Leave Policy to Security.

## Run a local demonstration

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy semantic --threshold 0.55 --size 35
```

Sample source: `data/transcript.txt`. Inspect `--json` for `Chunk` fields, including `parent_id`, `section_path`, `start/end` offsets and metadata. Inspect the actual local output rather than assuming all generated chunks are relevant to arbitrary questions.

## Real-model or retrieval-specific extension

```bash
# Genuine dense embeddings, if the provider supports /embeddings:
.venv/bin/python 9-rag-chunking/agent.py \
  --strategy semantic --provider api --threshold 0.55
```
Requires `EMBEDDING_MODEL` and a compatible endpoint in the root `.env`. The default `TopicLexicalEmbedder` uses deterministic **hashed lexical features**, NOT learned semantic embeddings; it is an algorithm demonstration only.

## Trade-offs and failure modes

Distances depend strongly on the model and domain; nearby opposing rules can have similar embeddings, while stylistic changes can create false breaks. A fixed threshold may over-split one corpus and under-split another. Very short or very long sentences require special handling.

## What to benchmark

Label real topic change points and compare breakpoint precision/recall, chunk-length distribution, evidence Recall@K and end-to-end answer correctness against Recursive and Structure-aware baselines. Keep the same embedding and retrieval model across comparisons.

The existing `--benchmark` command is only a small lexical evidence-coverage sanity check. It does not validate semantic segmentation, factual entailment, claim-level accuracy or live-model generalization.

## Interview answer template

**Why this method, rather than making all chunks larger?** Explain which **specific failure** the method addresses (topic mixing, missing source entity, missing exception, or fact dilution). Identify the trusted original evidence, expected extra cost, a measurable quality metric and a simpler baseline to beat.

## Exercise

Try different `--size` values (and thresholds where applicable). Examine a retrieved chunk and determine whether a hypothetical answer still needs another clause. Mark every invented or unverified representation distinctly from source facts.
