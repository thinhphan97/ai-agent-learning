# Contextual Enrichment / Contextual Retrieval

**Family:** contextual | **Implementation:** `contextual.py / contextual()`

[Return to Lab 9 overview](../../README.md)

## When to choose this strategy

Use when short chunks lose their entity, timeframe, topic or document identity: financial reports, policy excerpts, research papers and documents containing "this result", "the company" or similar references.

## Algorithm and data flow

Run Structure-aware first. For each child, prepend a **deterministic prefix** with document ID and heading path; keep original text in metadata and preserve the original source offset. This enriches the text that would be embedded or indexed without inventing extra facts.

**Worked example:** "Revenue increased 25%" becomes "Document: companyA_q2_2025 / Section: International Software Revenue / Revenue increased 25%" so the query's entity and time signals have matching terms.

## Run a local demonstration

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy contextual --size 35
```

Sample source: `data/policies.md`. Inspect `--json` for `Chunk` fields, including `parent_id`, `section_path`, `start/end` offsets and metadata. Inspect the actual local output rather than assuming all generated chunks are relevant to arbitrary questions.

## Real-model or retrieval-specific extension

`Contextual Retrieval` in the literature can also use an LLM to generate a short document-specific explanation per chunk. This lab implements only deterministic metadata enrichment: it does **not** call an LLM or claim Anthropic-style LLM contextualization. To extend it, preserve original text and verify every enriched statement against source metadata.

## Trade-offs and failure modes

Wrong titles, wrong dates or incorrect metadata can increase retrieval of the wrong document. Context prefixes increase indexed text and can bias retrieval toward repeated headings. Never substitute generated contextual text for source evidence in the final answer.

## What to benchmark

Hold underlying chunk boundaries fixed; compare bare vs metadata-prefixed embeddings/BM25. Use entity- and period-sensitive questions plus mismatched-document negatives. Measure Recall@K, wrong-company/wrong-quarter rates, index size and cost.

The existing `--benchmark` command is only a small lexical evidence-coverage sanity check. It does not validate semantic segmentation, factual entailment, claim-level accuracy or live-model generalization.

## Interview answer template

**Why this method, rather than making all chunks larger?** Explain which **specific failure** the method addresses (topic mixing, missing source entity, missing exception, or fact dilution). Identify the trusted original evidence, expected extra cost, a measurable quality metric and a simpler baseline to beat.

## Exercise

Try different `--size` values (and thresholds where applicable). Examine a retrieved chunk and determine whether a hypothetical answer still needs another clause. Mark every invented or unverified representation distinctly from source facts.
