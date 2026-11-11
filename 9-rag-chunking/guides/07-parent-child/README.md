# Parent–Child / Hierarchical Retrieval

**Family:** parent-child | **Implementation:** `hierarchy.py / parent_child(), expand_to_parents()`

[Return to Lab 9 overview](../../README.md)

## When to choose this strategy

Use when users ask for narrow details but answers require nearby conditions, exceptions, definitions or procedural steps: contracts, SOP, legal policies, technical manuals and long handbooks.

## Algorithm and data flow

Split the document into Markdown section **parents** with section paths. Sliding-window word children are made from each parent's actual source text. Every child includes `parent_id`; after matching a child, `expand_to_parents` resolves the unique full source parent with a strict character budget. Only children are normally indexed.

**Worked example:** A child matches "Eligible employees receive twelve days". Expanding to the Annual Leave section returns the eligibility period and part-time exception needed for a correct answer. Two matched children under the same parent yield ONE parent in context.

## Run a local demonstration

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy parent-child --size 20 --overlap 3
```

Sample source: `data/policies.md`. Inspect `--json` for `Chunk` fields, including `parent_id`, `section_path`, `start/end` offsets and metadata. Inspect the actual local output rather than assuming all generated chunks are relevant to arbitrary questions.

## Real-model or retrieval-specific extension

```python
from hierarchy import parent_child, expand_to_parents
collection = parent_child(document, child_words=20, overlap=3)
matched_children = collection.chunks[:2]  # Substitute real retriever hits
context = expand_to_parents(matched_children, collection, max_chars=3000)
```
Retrieve small; send only deduplicated, budgeted parent sections.

## Trade-offs and failure modes

Huge parents can fill context; matching unrelated sections creates noisy expansions. Parent IDs must remain version-consistent after reindexing. Expanding the parent is a retrieval-context strategy, not new knowledge or factual verification.

## What to benchmark

Measure child Recall@K **and** parent-level evidence coverage under the same context-token budget as flat retrieval. Count duplicated tokens and cases where retrieved child alone missed a required exception.

The existing `--benchmark` command is only a small lexical evidence-coverage sanity check. It does not validate semantic segmentation, factual entailment, claim-level accuracy or live-model generalization.

## Interview answer template

**Why this method, rather than making all chunks larger?** Explain which **specific failure** the method addresses (topic mixing, missing source entity, missing exception, or fact dilution). Identify the trusted original evidence, expected extra cost, a measurable quality metric and a simpler baseline to beat.

## Exercise

Try different `--size` values (and thresholds where applicable). Examine a retrieved chunk and determine whether a hypothetical answer still needs another clause. Mark every invented or unverified representation distinctly from source facts.
