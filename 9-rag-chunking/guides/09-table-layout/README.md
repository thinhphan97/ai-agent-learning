# Layout / Table-aware Chunking

**Family:** table | **Implementation:** `layout.py / table_aware()`

[Return to Lab 9 overview](../../README.md)

## Why use this method?

Tables, invoices, financial statements and technical specification grids have relational structure. A numeric cell is meaningless without its row, column, unit and report period. For real PDF documents, extraction and layout parsing are prerequisites to any useful table chunker.

## Algorithm

The lab accepts Markdown pipe tables, locates header and separator rows, validates column counts and creates a source table parent. Child chunks contain one or more complete data rows in a key=value representation, **repeating the table title and column names in every indexed child**. Original row offsets, row numbers, column values and table ID are preserved. This is not a PDF OCR/layout implementation.

**Illustrative example:** From a table with columns Segment, Q1 2025, Q2 2025, Unit and row Software | 120 | 150 | billion VND, the child contains "Segment=Software | Q1 2025=120 | Q2 2025=150 | Unit=billion VND" rather than "Software 120 150".

## Local run

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy table --group-size 1 --json
```

Example source: `data/financial_report.md`. Examine the printed chunk texts, source spans, parent links, metadata and any derived vectors with `--json`.

## Extension toward real applications

For PDF ingestion, first use a document-layout parser to preserve table grids and cell relations. Then convert the parsed table to a structured object and apply the same header-preserving row grouping. Avoid flattening multi-level headers or omitting notes/footnotes.

## Weaknesses and boundaries

Malformed Markdown cells, merged cells, cross-page tables, OCR errors, footnotes, numeric formatting and header hierarchies require more sophisticated parsing. The current parser is intentionally strict about row/header column counts and does not parse genuine PDFs.

## How to evaluate it

Label question-to-cell references and benchmark correct row, correct period, unit preservation, page citations, multi-level header handling and numeric consistency. For exact arithmetic over thousands of rows, use SQL/analytics after extraction rather than free-form LLM calculation.

**Production note:** This lab deliberately prioritizes transparent mechanics. The offline output is deterministic and fully testable, but does not establish production-grade parsing, embedding quality or LLM claim fidelity.

## Interview question

**Question:** Why is Layout / Table-aware Chunking a better fit than simple fixed-size chunking for your chosen source?

**Answer pattern:** Describe which information relation must be preserved (row/column, syntax scope, prior context or document-level synthesis); how the implementation does so; what metadata is needed; the risk of introducing incorrect derived text; and which metric will demonstrate an actual improvement.

## Hands-on exercise

Change `--size` or `--group-size` where applicable. Compare output and trace at least one source fact to the original input. Try the local offline benchmark to understand its limitations:

```bash
.venv/bin/python 9-rag-chunking/agent.py --benchmark
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
```
