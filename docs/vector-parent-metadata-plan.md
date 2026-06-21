# Vector Parent Metadata Implementation Plan

## 1. Current API Shape (Implemented)

### gRPC VectorIndexService RPCs

The `VectorIndexService` in `proto/sochdb.proto` currently defines six vector RPCs:

| RPC | Request | Response | Status |
|-----|---------|----------|--------|
| `CreateIndex` | name, dimension, config, metric | success, error, IndexInfo | ✅ |
| `DropIndex` | name | success, error | ✅ |
| `InsertBatch` | index_name, ids (uint64), vectors (flat f32) | inserted_count, error, duration_us | ✅ |
| `InsertStream` | (stream) index_name, id, vector | total_inserted, errors, duration_us | ✅ |
| `Search` | index_name, query, k, ef | results (SearchResult[]), duration_us, error, metric | ✅ |
| `SearchBatch` | index_name, queries, num_queries, k, ef | results (QueryResults[]), duration_us, metric | ✅ |

### Current Proto Messages (relevant fields)

```
message InsertBatchRequest {
  string index_name = 1;       // existing
  repeated uint64 ids = 2;     // existing
  repeated float vectors = 3;  // existing
}

message InsertStreamRequest {
  string index_name = 1;       // existing
  uint64 id = 2;               // existing
  repeated float vector = 3;   // existing
}

message SearchResult {
  uint64 id = 1;               // existing
  float distance = 2;          // existing
  string metric = 3;           // existing
}
```

### Current Internal Metadata Storage (in `sochdb-index/src/hnsw.rs`)

Already implemented at the index level:

```
HnswIndex {
    // Per-node metadata for filtered search, indexed by dense_index.
    // Stored as key-value string pairs for O(1) filter matching.
    pub(crate) metadata_store: Arc<RwLock<Vec<Option<Vec<(String, String)>>>>>,
    ...
}
```

Existing methods:
- `set_metadata(&self, node_id: u128, metadata: Vec<(String, String)>)` — `hnsw.rs:6398`
- `set_metadata_batch(&self, entries: &[(u128, Vec<(String, String)>)])` — `hnsw.rs:6413`
- `search_filtered(&self, query, k, ef, filter)` — uses `metadata_store` internally for post-filter matching — `hnsw.rs:6438`
- Python bindings exist: `HnswIndex.set_metadata()`, `HnswIndex.set_metadata_batch()`, `HnswIndex.search_filtered()` — `sochdb-python/src/lib.rs:644-738`

**Gap**: The gRPC API layer (`server.rs`) does NOT accept or return metadata. Metadata is only accessible through the Rust API or Python bindings, not through gRPC.

---

## 2. Proto Changes: Exact Fields to Add

### 2.1 New `VectorMetadata` Message

```protobuf
message VectorMetadata {
  optional uint64 parent_id = 1;   // optional parent vector ID (for trace lineage)
  optional string view_type = 2;   // optional view type label
}
```

Field numbers:
- `parent_id = 1` (optional uint64)
- `view_type = 2` (optional string)

### 2.2 Fields on `InsertBatchRequest`

```
message InsertBatchRequest {
  string index_name = 1;                     // existing
  repeated uint64 ids = 2;                   // existing
  repeated float vectors = 3;                // existing
  repeated VectorMetadata metadata = 4;      // NEW: one per vector (or empty array)
}
```

### 2.3 Fields on `InsertStreamRequest`

```
message InsertStreamRequest {
  string index_name = 1;                     // existing
  uint64 id = 2;                             // existing
  repeated float vector = 3;                 // existing
  VectorMetadata metadata = 4;               // NEW: metadata for this vector
}
```

### 2.4 Fields on `SearchResult`

```
message SearchResult {
  uint64 id = 1;                             // existing
  float distance = 2;                        // existing
  string metric = 3;                         // existing
  optional uint64 parent_id = 4;             // NEW: parent_id from metadata
  optional string view_type = 5;             // NEW: view_type from metadata
}
```

Field numbers chosen to avoid collision with existing fields (1, 2, 3 on SearchResult; 1, 2, 3 on InsertBatch/InsertStream).

### 2.5 Validation Rules

- `metadata` in `InsertBatchRequest` must be either empty (legacy/no metadata) or have the same length as `ids`.
- When `metadata.len() == ids.len()`, each entry's `parent_id` and `view_type` are stored in the index's `metadata_store`.
- When `metadata` is empty, vectors are inserted without metadata (backward-compatible).
- On `Search`/`SearchBatch`, the server reads metadata from `metadata_store` by dense_index and populates `SearchResult.parent_id` and `SearchResult.view_type` only when present.

---

## 3. Storage Plan

### 3.1 In-Memory Store (Already Exists)

The `metadata_store` field already exists on `HnswIndex`:

```rust
pub(crate) metadata_store: Arc<RwLock<Vec<Option<Vec<(String, String)>>>>>,
```

Indexed by `dense_index`. Each slot is:
- `None` — slot not yet allocated or node has no metadata
- `Some(vec![])` — node exists but has empty metadata
- `Some(vec![("key", "value"), ...])` — node has metadata pairs

### 3.2 New `get_metadata` Method to Add

```rust
pub fn get_metadata(&self, node_id: u128) -> Option<Vec<(String, String)>>
```

Resolves `node_id -> dense_index`, reads `metadata_store`, and returns a clone of
the metadata entries if present, or `None` if the node is not found or has no metadata.

### 3.3 gRPC Integration

On insert:
- Server converts proto `VectorMetadata` into `Vec<(String, String)>` pairs:
  - `parent_id` -> key `"parent_id"` with stringified value
  - `view_type` -> key `"view_type"` with string value
- Calls `index.set_metadata_batch()` with the converted pairs.

On search:
- After getting results, server calls `index.get_metadata()` per result ID.
- If metadata is present and contains `parent_id`/`view_type` keys, populates
  `SearchResult.parent_id` and `SearchResult.view_type`.
- If absent, fields are left unset (absent in proto3 = not transmitted on wire).

---

## 4. Persistence Plan

### 4.1 Trailer Magic

A fixed 8-byte trailer magic `SCMETA01` is appended after the bincode serialized
`IndexSnapshot` in the on-disk file. During load, the reader seeks to `file_length - 8`,
reads the magic bytes, and if they match, reads the `MetadataTrailer` structure
just before the magic.

### 4.2 MetadataTrailer Struct

```rust
/// Trailer appended after the IndexSnapshot for vector metadata persistence.
#[derive(Serialize, Deserialize)]
pub struct MetadataTrailer {
    /// Version for forward compatibility
    pub version: u32,
    /// Number of entries in the metadata table
    pub num_entries: u32,
    /// Serialized Vec<Option<Vec<(String, String)>>>
    pub entries: Vec<u8>,
}
```

### 4.3 Write Path: `save_to_disk`

Updated flow:
1. Write bincode serialized `IndexSnapshot` to file (existing).
2. Serialize `metadata_store` (the full `Vec<Option<Vec<(String, String)>>>`) via bincode into `MetadataTrailer.entries`.
3. Write bincode serialized `MetadataTrailer` to file.
4. Write 8-byte magic `SCMETA01`.

File layout after changes:

```
┌──────────────────────────────────────┐
│  IndexSnapshot (bincode)             │  ← existing
├──────────────────────────────────────┤
│  MetadataTrailer (bincode)           │  ← NEW
├──────────────────────────────────────┤
│  "SCMETA01" (8-byte magic)           │  ← NEW
└──────────────────────────────────────┘
```

### 4.4 Read Path: `load_from_disk`

Updated flow:
1. Open file, seek to `file_len - 8`, read magic bytes.
2. If magic == `SCMETA01`:
   - Read 4 bytes (trailer length), seek backward to trailer position.
   - Deserialize `MetadataTrailer`.
   - Deserialize `entries` into `Vec<Option<Vec<(String, String)>>>`.
   - Set `index.metadata_store = Arc::new(RwLock::new(decoded_metadata))`.
3. If magic != `SCMETA01` (legacy file, no trailer):
   - `metadata_store` remains empty (all nodes have no metadata).
4. Proceed with normal index deserialization (existing logic).

### 4.5 Compatibility

- **Forward**: New reader can read old files (trailer absent -> empty metadata).
- **Backward**: Old reader ignores trailing data after `IndexSnapshot` (bincode
  `deserialize_from` stops at end of stream; extra bytes are ignored). The old
  reader won't see metadata, but won't error.

### 4.6 Also Apply to Compressed Paths

Both `save_to_disk_compressed` and `load_from_disk_compressed` get the same
trailer treatment, appended after the gzip stream.

---

## 5. Test Coverage

Five tests to be implemented:

### 5.1 `legacy_insert_without_metadata_returns_absent_metadata`
- **File**: `sochdb-grpc/src/server.rs`
- **Label**: `LEGACY_INSERT_COMPAT`
- **What**: Insert vectors via gRPC with NO metadata field. Search and verify SearchResult has NO parent_id/view_type set (absent in proto3). Confirms backward compatibility.
- **Also covers**: `PARENT_ZERO` (parent_id=0 treated as absent)

### 5.2 `batch_insert_with_mixed_metadata_is_returned_by_search`
- **File**: `sochdb-grpc/src/server.rs`
- **Labels**: `MISSING_METADATA`, `SEARCH_METADATA`
- **What**: Insert 3 vectors where vector 0 has parent_id=100, view_type="episode"; vector 1 has no metadata; vector 2 has parent_id=200, no view_type. Search for nearby vector. Verify mixed metadata is correctly populated in results.

### 5.3 `search_batch_returns_metadata_for_mixed_presence`
- **File**: `sochdb-grpc/src/server.rs`
- **Label**: `BATCH_SEARCH_METADATA`
- **What**: Same mixed-data setup, search via SearchBatch, verify all results carry correct metadata across batch queries.

### 5.4 `test_save_and_load_preserves_metadata_trailer`
- **File**: `sochdb-index/src/persistence.rs`
- **Label**: `SNAPSHOT_ROUNDTRIP`
- **What**: Create index, insert vectors with metadata, save to disk, load from disk, verify metadata is preserved after roundtrip.

### 5.5 `test_index_isolation_metadata_not_leaked`
- **File**: `sochdb-index/src/hnsw.rs` (or `persistence.rs` if persistence-focused)
- **Label**: `INDEX_ISOLATION`
- **What**: Two separate `HnswIndex` instances. Insert metadata on index A. Verify index B has no metadata. Verify sharing does not leak.
- **Bonus sub-tests**: Malformed trailer (truncated SCMETA01 tail), truncated trailer (partial data), empty metadata file all handled gracefully.

---

## 6. Implementation Order

1. **Proto changes** (`proto/sochdb.proto`): Add `VectorMetadata`, update `InsertBatchRequest`, `InsertStreamRequest`, `SearchResult`.
2. **Regenerate code** from proto (build.rs already handles this).
3. **Add `get_metadata()`** to `HnswIndex` in `hnsw.rs`.
4. **Update gRPC server** (`server.rs`): Convert proto metadata to internal pairs on insert; resolve metadata on search.
5. **Add persistence trailer** (`persistence.rs`): `MetadataTrailer` struct, `SCMETA01` magic, updated `save_to_disk`/`load_from_disk`/compressed variants.
6. **Write tests** in both crates.
