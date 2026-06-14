# SochDB Vector Search: Distance Metric Opacity Issue

## Context

This document captures a conversation exploring a fundamental gap in the SochDB gRPC API: `SearchResult.distance` is opaque — callers receive a float with no indication of which distance metric produced it.

---

## The Problem

When a client calls the VectorIndexService `Search` RPC, it receives back:

```protobuf
message SearchResult {
  uint64 id = 1;
  float distance = 2;
}
```

The `distance` field tells you "how close" the returned vector is, but not **how that closeness was measured**. This makes the value meaningless for score-based fusion or cross-configuration comparison — it only works as a rank-order signal.

---

## Metrics in SochDB

SochDB supports three distance metrics, defined in `sochdb-index/src/hnsw.rs:366-370`:

```rust
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}
```

The dispatch happens at `hnsw.rs:4754-4759`:

```rust
fn distance_raw(&self, a: &[f32], b: &[f32]) -> f32 {
    match self.config.metric {
        DistanceMetric::Cosine    => simd_distance::cosine_distance_fast(a, b),
        DistanceMetric::Euclidean => simd_distance::l2_distance_fast(a, b),
        DistanceMetric::DotProduct => -simd_distance::dot_product_fast(a, b),  // NEGATED
    }
}
```

### 1. Cosine — angular similarity

- **Range**: `[0, 2]` (lower = more similar)
- **Implementation**: `sochdb-index/src/simd_distance.rs:685-696`
  - For normalized vectors: `1 - dot_product(a, b)`
  - For non-normalized: full cosine distance with norm computation
- **Auto-normalization**: SochDB normalizes vectors at ingest when using Cosine metric (`hnsw.rs:2719-2724`)
- **SIMD paths**: AVX2, AVX512, SSE41, NEON (`simd_distance.rs:181-200`)

### 2. Euclidean (L2) — straight-line geometric distance

- **Range**: `[0, ∞)` (lower = more similar)
- **Implementation**: `sochdb-index/src/simd_distance.rs:670-672`
  ```rust
  pub fn l2_distance_fast(a: &[f32], b: &[f32]) -> f32 {
      get_kernel().l2_squared(a, b).sqrt()
  }
  ```
- **SIMD-accelerated**: threshold-aware early abort variants (`simd_distance.rs:702-715`)
- **SIMD paths**: AVX2 with early abort, scalar fallback

### 3. DotProduct — raw inner product

- **Range**: Raw dot product is `(-∞, ∞)` but is **negated before return**, so API sees `[−∞, 0]` (lower = better)
- **Implementation**: `sochdb-index/src/simd_distance.rs:647-648` + negation at `hnsw.rs:4758`
  ```rust
  DistanceMetric::DotProduct => -simd_distance::dot_product_fast(a, b),
  ```
- **Critical**: The negation means you cannot recover the raw dot product value from the API's `distance` field without external knowledge that `DotProduct` was the metric

---

## The API Contract

The gRPC API in `proto/sochdb.proto`:

```protobuf
service VectorIndexService {
  rpc Search(SearchRequest) returns (SearchResponse);
  rpc SearchBatch(SearchBatchRequest) returns (SearchBatchResponse);
}

message SearchRequest {
  string index_name = 1;
  repeated float query = 2;
  uint32 k = 3;
  uint32 ef = 4;
}

message SearchResponse {
  repeated SearchResult results = 1;
  uint64 duration_us = 3;
  string error = 4;
}

message SearchResult {
  uint64 id = 1;
  float distance = 2;   // ← only (id, distance), no metric field
}
```

**Absence**: No `SearchResult.metric` field. No `DistanceMetric` enum in the proto. No metadata indicating which metric the index was configured with.

---

## Why It Is So Critical

### Cross-configuration comparison is broken

A `distance: 0.2` can mean:
- **Cosine**: `0.2 = 1 - cos(78.8°)` — moderately similar direction
- **Euclidean**: `0.2` geometric units — possibly very close depending on the space's scale
- **DotProduct**: `0.2 = -(dot = -0.2)` — the raw alignment was -0.2, which is mediocre and unbounded below

You cannot compare distances across indexes or queries without knowing the metric.

### DotProduct is doubly opaque

Even if you knew the metric was DotProduct, the API returns the negated value, so you cannot recover the original dot product signal. A "close" match with `distance: -50.0` under DotProduct actually has a raw dot product of `50.0` — a very strong alignment — but the negative sign convention inverts the intuitive ordering.

### Score-based fusion is blocked

The LoCoMo runner's `rrf_fuse_with_scores` (`benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py:188-205`) uses **rank-based** RRF only — it discards the distance value entirely and only uses position:

```python
def rrf_fuse_with_scores(
    bm25_ranked: List[int],
    vector_ranked: List[int],
    final_k: int,
    rrf_k: int,
    bm25_weight: float,
    vector_weight: float,
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}

    for rank, mid in enumerate(bm25_ranked, start=1):
        scores[mid] = scores.get(mid, 0.0) + bm25_weight / (rrf_k + rank)

    for rank, mid in enumerate(vector_ranked, start=1):
        scores[mid] = scores.get(mid, 0.0) + vector_weight / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:final_k]
```

The `distance` float returned by `search_sochdb()` is retrieved and dropped. The vector distance value is never used — only the rank position matters.

The `search_sochdb()` call returns `List[Tuple[int, float]] = [(record_id, distance), ...]` but the distance is immediately discarded and only the IDs are passed into `rrf_fuse_with_scores`.

If someone wanted to do score-based fusion (e.g., "weight by confidence"), they cannot because the scale is uninterpretable.

---

## Architectural Proof

### Proof 1 — The gRPC Vector API is flat KNN

The proto has no concept of source/parent, no grouping semantics, no view types:

```
SearchGroupedRequest    : (absent)
SearchGroupedResponse   : (absent)
source_memory_id        : (absent)
parent_id               : (absent)
view_type               : (absent)
group_by_source         : (absent)
source_candidate_k      : (absent)
view_overfetch_k        : (absent)
return_view_breakdown   : (absent)
```

The API can only say "give me K nearest vector IDs with distances." It cannot ask "give me K unique source memories after grouping child views."

### Proof 2 — The Rust server calls flat index search

The server implementation (`sochdb-grpc/src/server.rs:312-357`) is a direct passthrough:

```rust
async fn search(
    &self,
    request: Request<SearchRequest>,
) -> Result<Response<SearchResponse>, Status> {
    let req = request.into_inner();
    let (index, dimension) = self.get_index_with_dim(&req.index_name)?;
    let k = req.k.max(1) as usize;

    let results = match index.search(&req.query, k) {  // flat HNSW call
        Ok(r) => r,
        Err(e) => { return Ok(Response::new(SearchResponse { results: vec![], ... })); }
    };

    Ok(Response::new(SearchResponse {
        results: results
            .into_iter()
            .map(|(id, distance)| SearchResult {  // straight passthrough
                id: id as u64,
                distance,
            })
            .collect(),
        duration_us,
        error: String::new(),
    }))
}
```

The chain is: gRPC `SearchRequest` → `index.search(query, k)` → `Vec<(u128, f32)>` → `SearchResponse`. No grouping step exists.

### Proof 3 — The Index/Kernel abstraction is point retrieval

Every index implementation returns the same contract:

```rust
// sochdb-kernel/src/plugin.rs:213
fn nearest(&self, _query: &[u8], _k: usize) -> KernelResult<Vec<(RowId, f32)>>

// sochdb-index/src/vector.rs:246
pub fn search(&self, query: &Embedding, k: usize) -> Result<Vec<(u128, f32)>, String>

// sochdb-index/src/unified_search.rs:183
pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u128, f32)>

// sochdb-index/src/lockfree_hnsw.rs:750
pub fn search(&self, query: &QuantizedVector, k: usize) -> Vec<SearchCandidate>

// sochdb-index/src/vamana.rs:529
pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u128, f32)>, String>

// sochdb-vector/src/async_lsm.rs:696
pub fn search(&self, query: &[f32], k: usize) -> Vec<(VectorKey, f32)>
```

All return `Vec<(ID, score)>` — sorted by distance ascending. None return `SourceGroup`, `ViewHit` metadata, or grouped scores. The index has no concept of parent memory IDs or view types.

### Proof 4 — The runner does source mapping client-side

The Python runner (`run_hybrid_locomo_retrieval.py`) handles all multiview semantics above the API:

1. **Build search records with source mappings** (`build_memory_search_records`, lines 693-758):
   - Each record gets `source_memory_id` attached in Python before being sent to SochDB
   - Turn, event, entity, neighbor_window views are separate records with the same `source_memory_id`

2. **Compute view overfetch** (`compute_view_overfetch`, lines 679-690):
   - Scales up `candidate_k` by view count to account for child views
   - With `candidate_k=400` and 4 view types, fetches `view_candidate_k=1600` raw IDs

3. **Deduplicate view hits to sources** (`dedup_view_hits_to_source_ids`, lines 818-845):
   - After SochDB returns flat KNN results, maps `record_id → source_memory_id`, drops duplicates, preserves rank

4. **Multiview diagnostics** (`compute_multiview_diagnostics`, lines 761-815):
   - Tracks `view_type_counts_before_dedup`, `duplicate_view_candidate_count`, `sources_with_multiple_view_hits_count`

The architectural picture:

```
Client (run_hybrid_locomo_retrieval.py)
  1. build_memory_search_records()  → attaches source_memory_id to every record
  2. embed & send flat record IDs to SochDB
  3. SochDB gRPC API: SearchRequest { query, k, ef } → SearchResponse { [id, distance] }
  4. Server: index.search(query, k) → Vec<(u128, f32)>  ← flat point retrieval
  5. dedup_view_hits_to_source_ids()  → maps raw IDs back to source_memory_id
  6. scores against gold evidence

SochDB core (gRPC + index) is completely unaware of the parent/child view hierarchy.
All source-level semantics are client-side concerns.
```

---

## Proposed Enhancements

### Enhancement 1 — Fix the API contract (fastest impact)

Add metric to `SearchResult` in the proto:

```protobuf
message SearchResult {
  uint64 id = 1;
  float distance = 2;
  string metric = 3;  // "cosine" | "euclidean" | "dot_product"
}
```

The server (`sochdb-grpc/src/server.rs`) maps from `self.config.metric` at the response construction point. Python SDK exposes `metric` on returned objects.

Or expose it at the index level via `IndexInfo`:

```protobuf
message IndexInfo {
  string name = 1;
  uint64 dimension = 2;
  string metric = 3;  // NEW: "cosine" | "euclidean" | "dot_product"
  ...
}
```

This lets callers introspect an index's metric before issuing queries.

### Enhancement 2 — Score-based fusion in the runner

Once metric is available in the response, the runner could do **weighted score fusion** instead of pure rank-based RRF:

```python
v_vector_ranked = search_sochdb(...)  # returns [(id, distance, metric), ...]

for rank, (mid, dist, metric) in enumerate(vector_results):
    if metric == "dot_product":
        confidence = 1.0 / (1.0 + abs(dist))  # interpret negated dot
    elif metric == "cosine":
        confidence = 1.0 - (dist / 2.0)         # normalize to [0,1]
    else:  # euclidean
        confidence = 1.0 / (1.0 + dist)

    scores[mid] += vector_weight * confidence / (rrf_k + rank)
```

### Enhancement 3 — DotProduct de-negate for diagnostics

The `hnsw.rs:4758` negation means callers cannot recover raw dot product. The server could also return `raw_score: float` alongside `distance: float`, where for DotProduct `raw_score = -distance`, for others `raw_score = distance`. This gives callers the full signal for diagnostics and adaptive weighting.

---

## GitHub Issue

**Issue**: https://github.com/sochdb/sochdb/issues/62

**Labels**: `enhancement`, `api`

---

## Ablation Ladder Results (Related Context)

During the same session, a multi-hop retrieval ablation ladder was run to identify which changes improve multi-hop Hit/Recall. The champion baseline (`multihop_multiview_metadata_k200_overfetch_fixed`) had K200 Recall of 0.8281.

Key finding: removing `entity` and `neighbor_window` views (using only `turn,event`) improved K200 Recall from 0.8281 to 0.8437 (+0.016). This worked because the index was returning more child-view IDs than the rank budget could cover for unique source memories — reducing view noise at the index level had a bigger effect than tuning candidate-k, entity_multi, neighbor expansion, or coverage selection.

The ablation confirmed:
- Higher `candidate_k` hurt (more low-quality candidates diluted rankings)
- `entity_multi` significantly hurt recall
- `local_neighbor_expansion` hurt recall
- Coverage selection was neutral
- The best config: `turn,event` views with `candidate_k=400`, `query_mode=single`, rank selection