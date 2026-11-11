# Proposition / Atomic-fact Chunking

**Family:** proposition | **Implementation:** `proposition.py / propositions()`

[Return to Lab 9 overview](../../README.md)

## When to choose this strategy

Good for factual QA, business rules, retrieval of independent claims, fact checking and high-precision answer lookup. A single dense paragraph may contain several independent statements; splitting facts can make candidate retrieval more targeted.

## Algorithm and data flow

Offline mode emits unchanged sentences as **proposition candidates** and attaches source-sentence offsets; it does not claim that compound sentences have been atomized. API mode calls an LLM-based proposition extractor on each sentence and stores generated candidates with source sentence and `requires_source_verification=true`. Generated propositions deliberately have no invented character offsets.

**Worked example:** "Company A reported 150 billion VND revenue in Q2 and grew 25% QoQ" could be rewritten into two self-contained claims, each retaining Company A, Q2 and the measurement unit. A valid extractor must also preserve qualifications such as "might" or "only if".

## Run a local demonstration

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy proposition
```

Sample source: `data/policies.md`. Inspect `--json` for `Chunk` fields, including `parent_id`, `section_path`, `start/end` offsets and metadata. Inspect the actual local output rather than assuming all generated chunks are relevant to arbitrary questions.

## Real-model or retrieval-specific extension

```bash
# Makes one or more PAID LLM calls per source sentence.
.venv/bin/python 9-rag-chunking/agent.py \
  --strategy proposition --provider api
```
Requires `LLM_MODEL` and other root `.env` values. Generated candidates are not automatically verified facts.

## Trade-offs and failure modes

Extractive restructuring can silently drop negations, units, conditions, time qualifiers or uncertainty. Propagation of that distortion into embeddings and answers is a critical reliability risk. Very granular indexes also need more vectors.

## What to benchmark

Use a labeled fact-level corpus: check atomicity, referential self-containment, predicate/subject/quantity preservation, source entailment and fact-level Recall@K. A model-generated proposition is only a retrieval aid until the original source supports it.

The existing `--benchmark` command is only a small lexical evidence-coverage sanity check. It does not validate semantic segmentation, factual entailment, claim-level accuracy or live-model generalization.

## Interview answer template

**Why this method, rather than making all chunks larger?** Explain which **specific failure** the method addresses (topic mixing, missing source entity, missing exception, or fact dilution). Identify the trusted original evidence, expected extra cost, a measurable quality metric and a simpler baseline to beat.

## Exercise

Try different `--size` values (and thresholds where applicable). Examine a retrieved chunk and determine whether a hypothetical answer still needs another clause. Mark every invented or unverified representation distinctly from source facts.
