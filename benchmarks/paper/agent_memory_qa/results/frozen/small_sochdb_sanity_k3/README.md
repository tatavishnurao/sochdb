# Small SochDB Agent Memory QA sanity result

Dataset: small_memory_qa.jsonl
Questions: 16
k: 3
Systems: no_memory, recent_history, vector_rag placeholder, sochdb placeholder

This is a synthetic sanity-check benchmark for validating the Agent Memory QA harness.
It is not a final SochDB system result because vector_rag and sochdb are placeholder retrievers.

At k=3:
- no_memory: EM/F1 0.000, evidence_hit_rate 0.000
- recent_history: EM/F1 0.188, evidence_hit_rate 0.250
- vector_rag: EM/F1 0.875, evidence_hit_rate 0.875
- sochdb: EM/F1 1.000, evidence_hit_rate 1.000

Interpretation:
The harness can distinguish memory retrieval strategies. The SochDB-style retriever reaches full answer accuracy with a similar context budget to vector-style retrieval, while vector-style retrieval misses temporal-update and exact benchmark-spec questions.
