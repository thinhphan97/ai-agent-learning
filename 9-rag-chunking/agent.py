"""Lab 9 CLI: explore 12 chunking families with 13 runnable modes.

No secret credentials or external calls are used without --provider api.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from benchmark import assess
from runner import STRATEGIES, run, sample_file
from shared import cap_chars, load_document


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 9 — RAG Chunking Strategies")
    parser.add_argument(
        "--strategy",
        choices=("all", *STRATEGIES),
        default="all",
    )
    parser.add_argument("--file", type=Path, help="Optional custom UTF-8 document")
    parser.add_argument(
        "--size", type=int, default=35, help="Word limit/child size; code uses lines"
    )
    parser.add_argument(
        "--overlap", type=int, default=5, help="Overlap in words where supported"
    )
    parser.add_argument(
        "--group-size", type=int, default=2, help="Sentence groups or table rows"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55, help="Semantic distance breakpoint"
    )
    parser.add_argument(
        "--provider",
        choices=("toy", "api"),
        default="toy",
        help="Explicitly opt in to remote embedding/LLM calls",
    )
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        help="Optional already-downloaded HuggingFace transformer for genuine late pooling",
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="Preview at most N indexed chunks"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Evaluate text methods using a shared source and fixed lexical retrieval",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON")
    args = parser.parse_args()

    if args.provider == "api" and (args.strategy == "all" or args.benchmark):
        parser.error("--provider api requires one explicit strategy")
    if args.local_model_dir and args.strategy != "late":
        parser.error("--local-model-dir requires --strategy late")
    if args.limit < 0 or args.limit > 50:
        parser.error("--limit must be between 0 and 50")

    try:
        if args.benchmark:
            methods = [
                "fixed",
                "recursive",
                "sentence",
                "paragraph",
                "structure",
                "semantic",
                "proposition",
                "parent-child",
                "contextual",
                "late",
                "raptor",
            ]
            doc = load_document(args.file or sample_file("fixed"))
            records = [
                assess(
                    run(
                        method,
                        doc,
                        size=args.size,
                        overlap=args.overlap,
                        group_size=args.group_size,
                        threshold=args.threshold,
                    )
                )
                for method in methods
            ]
            if args.json:
                print(json.dumps(records, indent=2, ensure_ascii=False))
            else:
                print(
                    f"Evidence benchmark — document={doc.id}, top_k=3, context_budget=150 words"
                )
                print("Strategy             Coverage   Indexed   Avg words")
                for record in records:
                    print(
                        f"{record['strategy']:<20} "
                        f"{record['evidence_coverage_at_3']:.0%} "
                        f"{record['indexed_chunks']:>9} "
                        f"{record['avg_index_words']:>11}"
                    )
                print("Teaching proxy only; NOT semantic relevance or answer accuracy.")
            return 0

        methods = STRATEGIES if args.strategy == "all" else (args.strategy,)
        all_results = []
        for method in methods:
            doc = load_document(args.file or sample_file(method))
            output = run(
                method,
                doc,
                size=args.size,
                overlap=args.overlap,
                group_size=args.group_size,
                threshold=args.threshold,
                provider=args.provider,
                local_model_dir=args.local_model_dir,
            )
            if not output.chunks:
                raise ValueError(f"No chunks created for {method}")
            if args.json:
                all_results.append(
                    {
                        "method": method,
                        "document": doc.id,
                        "chunks": [asdict(item) for item in output.chunks],
                        "parents": [asdict(item) for item in output.parents],
                        "vectors": output.vectors,
                    }
                )
            else:
                print(
                    f"\n[{method}] doc={doc.id} "
                    f"indexed={len(output.chunks)} parents={len(output.parents)} "
                    f"vectors={len(output.vectors)}"
                )
                for item in output.chunks[: args.limit]:
                    print(
                        f"  {item.id} ({item.kind})"
                        f" parent={item.parent_id or '-'}"
                        f" section={'/'.join(item.section_path) or '-'}"
                    )
                    print(f"    {cap_chars(item.text)}")
        if args.json:
            print(json.dumps(all_results, indent=2, ensure_ascii=False))
    except (ValueError, ImportError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
