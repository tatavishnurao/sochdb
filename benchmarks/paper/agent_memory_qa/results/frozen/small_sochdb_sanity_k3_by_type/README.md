# Small SochDB Agent Memory QA sanity benchmark

Dataset: `small_memory_qa.jsonl`  
Questions: 16  
k: 3  
Systems:
- `no_memory`
- `recent_history`
- `vector_rag` placeholder
- `sochdb` placeholder

This is a synthetic sanity-check benchmark for validating the Agent Memory QA harness.
It is not a final SochDB system result because `vector_rag` and `sochdb` are placeholder retrievers.

## Aggregate result at k=3

| System | EM/F1 | Cited Recall | Evidence Hit | Avg Tokens |
|---|---:|---:|---:|---:|
| no_memory | 0.000 | 0.000 | 0.000 | 0.0 |
| recent_history | 0.188 | 0.198 | 0.250 | 91.0 |
| vector_rag | 0.875 | 0.792 | 0.875 | 102.2 |
| sochdb | 1.000 | 0.885 | 1.000 | 99.9 |

## Per-type finding

At k=3, the vector-style baseline fails on:
- `knowledge_update`
- `multi_fact`

The SochDB-style retriever succeeds on all question types.

## Failure analysis

The vector-style baseline misses:

1. `q4`, a knowledge-update question:
   - It retrieves semantically adjacent benchmark-planning context.
   - It misses the actual priority-update record where Modular Baseline vs SochDB becomes P0.1.

2. `q12`, a multi-fact benchmark-spec question:
   - It retrieves the general Token Budget vs Answer Quality topic.
   - It misses the exact strategy-list record containing top-k, BM25, hybrid, planner, TOON, and planner+TOON.

## Interpretation

The harness distinguishes no-memory, recent-history, topic-style retrieval, and SochDB-style retrieval.

At k=3, the SochDB-style retriever reaches full answer accuracy with roughly the same context budget as the vector-style retriever.

This supports the benchmark thesis that agent memory systems need current, specific, provenance-grounded memory retrieval rather than only semantically similar context.

## Safe wording

This should be described as a synthetic sanity benchmark, not as a final SochDB system result.

Suggested wording:

> In a synthetic SochDB-specific Agent Memory QA sanity suite, the benchmark harness distinguishes no-memory, recent-history, topic-based retrieval, and SochDB-style retrieval. At k=3, the SochDB-style retriever reaches full answer accuracy, while the vector-style baseline misses a priority-update case and an exact benchmark-spec case. These preliminary results validate the benchmark harness and motivate replacing the placeholder retriever with real SochDB `CONTEXT SELECT` execution.
