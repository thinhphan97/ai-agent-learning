# Late Chunking (Post-contextual Pooling)

**Family:** late | **Implementation:** `late.py / late(); providers.py`

[Return to Lab 9 overview](../../README.md)

## Why use this method?

Use when a phrase or pronoun in a small chunk only makes sense with preceding/following text, and you have a contextual word/token encoder able to process the larger passage. Late Chunking changes when text is **embedded and pooled**, not necessarily where chunk boundaries are drawn.

## Algorithm

Split source into whitespace words to determine intended ranges. Call encoder.encode_words(words) ONCE on the entire document to obtain contextual vectors (one per word), then mean-pool token/word states for each chunk range. The result is a ChunkSet with indexed chunks and an aligned numeric vector for each chunk.

**Illustrative example:** "The city is very old" becomes more informative when the contextual encoder saw the preceding sentence "Rome is the capital of Italy". The vector of "the city" can carry document context before chunk pooling.

## Local run

```bash
cd ~/lab/ai-agent-learning
.venv/bin/python 9-rag-chunking/agent.py --strategy late --size 20 --overlap 3 --json
```

Example source: `data/policies.md`. Examine the printed chunk texts, source spans, parent links, metadata and any derived vectors with `--json`.

## Extension toward real applications

The default WordContextToyEncoder simply mixes deterministic lexical features with neighbor features; it is **not a pretrained Transformer**. To execute true token-contextual pooling, install transformers + torch separately and supply an existing local model path:

```bash
.venv/bin/python 9-rag-chunking/agent.py \
  --strategy late --local-model-dir /path/to/local/hf-encoder
```

The optional adapter calls a local AutoModel on pretokenized words, averages subword embeddings per word, then the late chunker pools word ranges. No weights are downloaded automatically. Real models have context windows; this adapter intentionally fails if input exceeds the model's positional limit.

## Weaknesses and boundaries

A standard /embeddings endpoint that returns one pooled vector per text **cannot perform true Late Chunking** unless it exposes per-token contextual states. Very long documents need macro windows. Word mapping through subword tokenization can fail with unsupported tokenizers; raw pretrained token states are not necessarily optimized as retrieval embeddings.

## How to evaluate it

Compare genuinely contextual late-pooled vectors with independently encoded chunk vectors from the SAME encoder architecture, and evaluate coreference-intensive questions, Recall@K, token window length, memory footprint and latency. Do not call the toy output a semantic retrieval benchmark.

**Production note:** This lab deliberately prioritizes transparent mechanics. The offline output is deterministic and fully testable, but does not establish production-grade parsing, embedding quality or LLM claim fidelity.

## Interview question

**Question:** Why is Late Chunking (Post-contextual Pooling) a better fit than simple fixed-size chunking for your chosen source?

**Answer pattern:** Describe which information relation must be preserved (row/column, syntax scope, prior context or document-level synthesis); how the implementation does so; what metadata is needed; the risk of introducing incorrect derived text; and which metric will demonstrate an actual improvement.

## Hands-on exercise

Change `--size` or `--group-size` where applicable. Compare output and trace at least one source fact to the original input. Try the local offline benchmark to understand its limitations:

```bash
.venv/bin/python 9-rag-chunking/agent.py --benchmark
.venv/bin/python -m unittest discover -s 9-rag-chunking/tests -v
```