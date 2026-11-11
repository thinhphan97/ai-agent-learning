"""Python AST-aware chunking with class/function signatures and line positions."""

from __future__ import annotations

import ast

from contracts import ChunkSet, Document, make_chunk


def python_ast(doc: Document, *, max_lines: int = 80) -> ChunkSet:
    if type(max_lines) is not int or not 3 <= max_lines <= 2000:
        raise ValueError("max_lines must be in [3,2000]")
    try:
        tree = ast.parse(doc.text)
    except SyntaxError as exc:
        raise ValueError("Invalid Python source") from exc
    lines = doc.text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    chunks, parents = [], []

    def emit(node: ast.AST, *, scope: tuple[str, ...]) -> None:
        if not hasattr(node, "end_lineno") or node.end_lineno is None:
            return
        start_line = node.lineno
        end_line = node.end_lineno
        span_a, span_b = offsets[start_line - 1], offsets[end_line]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
            name = node.name
        elif isinstance(node, ast.ClassDef):
            kind = "class"
            name = node.name
        else:
            return
        full_scope = (*scope, name)
        # Classes carry a parent context; methods are searchable separately.
        if isinstance(node, ast.ClassDef):
            parent = make_chunk(
                doc,
                "code-parent",
                len(parents),
                doc.text[span_a:span_b],
                kind="class",
                start=span_a,
                end=span_b,
                section_path=full_scope,
                metadata={"line_range": [start_line, end_line]},
            )
            parents.append(parent)
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                for method in methods:
                    emit(method, scope=full_scope)
                return

        parent_id = parents[-1].id if scope and parents else None
        for part_start in range(start_line, end_line + 1, max_lines):
            part_end = min(part_start + max_lines - 1, end_line)
            a, b = offsets[part_start - 1], offsets[part_end]
            text = doc.text[a:b].strip()
            if text:
                chunks.append(
                    make_chunk(
                        doc,
                        "code",
                        len(chunks),
                        text,
                        kind=kind,
                        start=a,
                        end=b,
                        parent_id=parent_id,
                        section_path=full_scope,
                        metadata={
                            "line_range": [part_start, part_end],
                            "symbol": ".".join(full_scope),
                            "partial": part_start != start_line or part_end != end_line,
                        },
                    )
                )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            emit(node, scope=())
    if not chunks:
        raise ValueError("No Python functions or classes found")
    return ChunkSet("code", tuple(chunks), tuple(parents))
