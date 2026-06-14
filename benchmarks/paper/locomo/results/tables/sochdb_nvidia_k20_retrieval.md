# LoCoMo Retrieval-Only Benchmark: SochDB + NVIDIA Nemotron

## Configuration

- Dataset: LoCoMo converted QA split
- Questions: 1,986
- Sparse retriever: BM25
- Dense retriever: SochDB gRPC vector backend
- Embedding model: `nvidia/llama-nemotron-embed-1b-v2`
- Fusion: RRF
- k: 20
- candidate_k: 100
- bm25_weight: 1.5
- vector_weight: 0.75

## Result

| System | n | Evidence Hit@20 | Evidence Recall@20 | Avg context tokens | Avg latency |
|---|---:|---:|---:|---:|---:|
| SochDB + BM25 + NVIDIA Nemotron + RRF | 1,986 | 0.7056 | 0.6522 | 657.98 | 200.81 ms |

## Category Breakdown

| Category | n | Hit@20 | Recall@20 |
|---|---:|---:|---:|
| adversarial | 446 | 0.7534 | 0.7455 |
| multi_hop | 96 | 0.4157 | 0.3058 |
| open_domain | 841 | 0.7337 | 0.7210 |
| single_hop | 282 | 0.5801 | 0.3271 |
| temporal | 321 | 0.7563 | 0.7232 |

## Notes

This is a retrieval-only evidence recovery benchmark, not end-to-end LoCoMo answer accuracy.

The current latency reflects remote unary SochDB gRPC calls. Retrieval quality is the primary artifact here; serving latency should be improved separately using concurrent search or SearchBatch RPC.
