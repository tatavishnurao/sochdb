# Memory Benchmark Retrieval-Only Report

## Dataset

- Dataset name:
- Dataset version/source:
- Raw data path:
- Normalized memories path:
- Normalized questions path:
- Number of samples:
- Number of memories:
- Number of questions:
- Number of scored questions:
- Number of unscored questions:
- Evidence label availability:

## Retrieval Configuration

- Retrieval-only run:
- Query mode:
- Memory render mode:
- K:
- Candidate K:
- BM25 weight:
- Vector weight:
- RRF K:
- Reranker provider:
- Embedding provider:
- Embedding model:
- Embedding dimension:
- Embedding cache:
- Vector backend:
- SochDB host/port:
- Collection prefix:

## Overall Retrieval Metrics

| K | Scored Questions | Hit@K | Recall@K | Full-Hit Rows | Partial-Hit Rows | Zero-Hit Rows |
|---|---:|---:|---:|---:|---:|---:|
| 20 |  |  |  |  |  |  |
| 50 |  |  |  |  |  |  |
| 100 |  |  |  |  |  |  |

Only include K=150 or K=200 when retrieved list length diagnostics confirm those depths are present.

## Category Metrics

| Category | Questions | Hit@K | Recall@K |
|---|---:|---:|---:|
|  |  |  |  |

## Sample-Wise Metrics

| Sample | Questions | Hit@K | Recall@K |
|---|---:|---:|---:|
|  |  |  |  |

## Evidence-Count Buckets

| Evidence Count Bucket | Questions | Hit@K | Recall@K |
|---|---:|---:|---:|
| 1 |  |  |  |
| 2 |  |  |  |
| 3-5 |  |  |  |
| 6+ |  |  |  |

## Retrieved Length Diagnostics

- Requested K:
- Minimum retrieved list length:
- Maximum retrieved list length:
- Average retrieved list length:
- Rows below requested K:
- K values withheld because retrieved depth was unavailable:

## Failure Analysis

- Full-hit rows:
- Partial-hit rows:
- Zero-hit rows:
- Common missing evidence IDs:
- Representative missing evidence examples:
- Evidence span mapping failures:
- Other unscored reasons:

## Limitations

This report covers retrieval-only evidence recovery. It is not final benchmark answer accuracy.

Answer generation, judge/evaluator scoring, and official end-to-end benchmark comparison are out of scope unless those pipelines are implemented and run separately.

## Notes

- Retrieval-only evidence recovery:
- Answer generation:
- Judge/evaluator score:
- Dataset-specific caveats:
