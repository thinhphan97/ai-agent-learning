# Code-aware / Python AST Chunking

**Family:** code | **Implementation:** `code.py / python_ast()`

[Return to Lab 9 overview](../../README.md)

## Why use this method?

Use source-code syntax as a natural boundary for repository Q&A, function discovery, bug investigation, API understanding, and programming assistants. Raw character windows can cut off a function condition, its return statement or the surrounding class name.

## Algorithm

Parse a .py file with Python's ast module and inspect ClassDef, FunctionDef and AsyncFunctionDef nodes. Classes are retained as source parents; methods are emitted as searchable child functions with symbol path, class, file source span and line range. Oversized nodes are split by line count and tagged partial=true.

**Illustrative example:** "OrderService.calculate_revenue" is emitted as a function chunk under parent "OrderService"; "format_currency" is an independent top-level function chunk. A query about paid-order revenue should be able to retrieve the method without ingesting the whole repository into one vector.

## Local run

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy code --size 20 --json
```

Example source: `data/order_service.py`. Examine the printed chunk texts, source spans, parent links, metadata and any derived vectors with `--json`.

## Extension toward real applications

A production code assistant should parse more node kinds, enrich with function signatures, import paths, docstrings, symbol cross-references and a call graph. Tree-sitter is useful for multi-language repositories. Preserve commit hashes and path/line metadata for citations.

## Weaknesses and boundaries

A function can rely on imported helpers and external classes; an AST node is not a self-contained unit of business logic. Giant methods remain too long, decorators may modify behavior, generated code may be unusual, and line splitting inside a long node can break syntax. This lab intentionally supports Python only.

## How to evaluate it

Evaluate function/symbol Recall@K, definition-site accuracy, ability to retrieve called dependencies and whether generated explanations correctly include control-flow and error-handling context. Compare with simple line-window and recursive baselines.

**Production note:** This lab deliberately prioritizes transparent mechanics. The offline output is deterministic and fully testable, but does not establish production-grade parsing, embedding quality or LLM claim fidelity.

## Interview question

**Question:** Why is Code-aware / Python AST Chunking a better fit than simple fixed-size chunking for your chosen source?

**Answer pattern:** Describe which information relation must be preserved (row/column, syntax scope, prior context or document-level synthesis); how the implementation does so; what metadata is needed; the risk of introducing incorrect derived text; and which metric will demonstrate an actual improvement.

## Hands-on exercise

Change `--size` or `--group-size` where applicable. Compare output and trace at least one source fact to the original input. Try the local offline benchmark to understand its limitations:

```bash
.venv/bin/python 9-rag-chunking/agent.py --benchmark
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
```
