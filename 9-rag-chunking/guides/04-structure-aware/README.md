# Structure-aware Markdown Sections

**Family:** structure | **Implementation:** `structure.py / markdown_sections()` | **Mode:** offline, deterministic

[Return to Lab 9 overview](../../README.md)

## Why this strategy exists

Prefer for manuals, policy documentation, SOPs, contracts converted to structured Markdown, internal wikis and documentation sites with trustworthy headings. Document structure is usually more meaningful than fixed-length boundaries.

## Exact mechanism

Parse Markdown heading levels (# through ######) to build section paths. Emit a source-linked section parent and index the section body. Oversized sections fall back to recursive splitting, with child offsets adjusted to the original document.

**Worked example:** "Company Handbook > Annual Leave" becomes a section path, so the indexed text "Full-time employees receive twelve days" can be disambiguated without searching the entire handbook. Every child points to its original section parent.

## Run it

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy structure --size 35 --overlap 5
```

Source fixture: `data/policies.md`. To inspect the complete structured output, add `--json`. For this strategy, the implementation creates `Chunk` objects with `document_id`, `id`, `text`, optional source offsets and metadata.

## Parameters to tune

--size 35 sets maximum section-body word budget for fallback splitting. In production use heading hierarchies, document IDs, version IDs and source paths as metadata; optionally prepend section path in the indexing text.

## Failure modes / when NOT to choose it

A poor PDF/HTML parser can misidentify headings. Large headings must still be split; empty sections produce no indexable chunk. Markdown lists and tables need more specialized element handling. Heading metadata is available separately here rather than prepended to raw child text.

## Evaluate it

Inspect whether the correct policy section is retrieved, whether conditions/exceptions remain associated, source line integrity, and how often a heading path is missing or incorrect.

**Implementation boundary:** This is a Markdown-heading parser, not a PDF layout parser; use the Table-aware guide for structured table elements.

## Interview prompt

**Question:** Why choose Structure-aware Markdown Sections over arbitrary fixed-size windows?

**Answer guideline:** Explain the evidence unit selected by the algorithm, how this helps or harms retrieval, whether source boundaries are preserved, and what measurements would justify choosing it over the simpler baseline. Always distinguish retrieval accuracy from downstream grounded answer quality.

## Practical exercise

Run the command above, change the chunk size or grouping, inspect the first three chunk IDs and offsets, then run:

```bash
.venv/bin/python 9-rag-chunking/agent.py --benchmark
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
```

The benchmark is a tiny **lexical evidence-coverage proxy** on policy text. It is not a representative semantic retrieval benchmark, and the results for different document types cannot be compared solely by chunk counts.
