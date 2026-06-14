# Issue: SearchResult.distance is opaque — no metric field in gRPC API

## Summary

The gRPC Vector API's `SearchResult` returns only `(id, distance)` pairs with no metadata about which distance metric was used. This makes the `distance` field pretty confusing for any cross-configuration comparison or score-based fusion, only usable as a rank-order signal.

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

## Why it is so critical

###  Cross-configuration comparison is broken

A `distance: 0.2` can mean:
- **Cosine**: `0.2 = 1 - cos(78.8°)` — moderately similar direction
- **Euclidean**: `0.2` geometric units — possibly very close depending on the space's scale
- **DotProduct**: `0.2 = -(dot = -0.2)` — the raw alignment was -0.2, which is mediocre and unbounded below

We cannot compare distances across indexes or queries without knowing the metric.
 
###  DotProduct is doubly opaque

Even if you knew the metric was DotProduct, the API returns the negated value, so you cannot recover the original dot product signal. A "close" match with `distance: -50.0` under DotProduct actually has a raw dot product of `50.0` — a very strong alignment — but the negative sign convention inverts the intuitive ordering.

---

## Proposed fix

```protobuf
message SearchResult {
  uint64 id = 1;
  float distance = 2;
  string metric = 3;  // "cosine" | "euclidean" | "dot_product"
}
```

or expose it at the index-level in 'IndexInfo' so callers know which metric an index has used before issuing queries.
