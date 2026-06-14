# Small SochDB Agent Memory QA k-sweep summary

## Setup

Dataset: `small_memory_qa.jsonl`  
Questions: 16  
Systems: `no_memory`, `recent_history`, `vector_rag` placeholder, `sochdb` placeholder  
k values: 1, 3, 5, 10  

This is a synthetic sanity-check benchmark for the Agent Memory QA harness. It should not be reported as a final SochDB system result because `vector_rag` and `sochdb` are still placeholder retrievers.

## Main result

The SochDB-style retriever reaches full answer accuracy at k=3, while the vector-style retriever does not reach full accuracy even at k=10.

At k=3:

| System | EM/F1 | Cited Recall | Evidence Hit | Avg Tokens |
|---|---:|---:|---:|---:|
| recent_history | 0.188 | 0.198 | 0.250 | 91.0 |
| vector_rag | 0.875 | 0.792 | 0.875 | 102.2 |
| sochdb | 1.000 | 0.885 | 1.000 | 99.9 |

## k-sweep

| k | System | EM/F1 | Cited Recall | Evidence Hit | Avg Tokens |
|---:|---|---:|---:|---:|---:|
| 1 | recent_history | 0.000 | 0.083 | 0.188 | 25.0 |
| 1 | vector_rag | 0.625 | 0.583 | 0.750 | 32.3 |
| 1 | sochdb | 0.750 | 0.677 | 0.813 | 34.1 |
| 3 | recent_history | 0.188 | 0.198 | 0.250 | 91.0 |
| 3 | vector_rag | 0.875 | 0.792 | 0.875 | 102.2 |
| 3 | sochdb | 1.000 | 0.885 | 1.000 | 99.9 |
| 5 | recent_history | 0.250 | 0.260 | 0.313 | 172.0 |
| 5 | vector_rag | 0.938 | 0.885 | 0.938 | 165.6 |
| 5 | sochdb | 1.000 | 0.917 | 1.000 | 167.2 |
| 10 | recent_history | 0.563 | 0.573 | 0.625 | 326.0 |
| 10 | vector_rag | 0.938 | 0.938 | 0.938 | 325.2 |
| 10 | sochdb | 1.000 | 1.000 | 1.000 | 324.9 |

## Interpretation

Recent-history retrieval remains weak even as k increases, showing that long-term memory QA cannot be solved by recent context alone.

Vector-style retrieval performs well on broad semantic matches but remains brittle on temporal updates and exact benchmark-spec retrieval. It needs larger context budgets to improve and still does not reach full accuracy.

The SochDB-style retriever reaches full answer accuracy at k=3, showing that structured memory signals, importance, recency, and benchmark-specific metadata can recover better evidence under tighter context budgets.

## Paper-safe wording

This result should be described as a sanity-check benchmark, not a final SochDB result.

Safe wording:

> In a synthetic SochDB-specific Agent Memory QA sanity suite, the benchmark harness distinguishes no-memory, recent-history, topic-based retrieval, and SochDB-style retrieval. The SochDB-style retriever reaches full answer accuracy at k=3, while the vector-style baseline remains below full accuracy even at k=10. These preliminary results validate the benchmark harness and motivate replacing the placeholder retriever with real SochDB `CONTEXT SELECT` execution.

