# SochDB Architectural Observations

> **Note:** This is a structured summary of the current OpenCode architectural run (2026-05-29), not a verbatim transcript. It captures the analysis performed across a multi-phase inspection of the SochDB repository.

---

## 1. Purpose of This Run

The goal was to map SochDB against a generic DBMS architecture and produce a critical review suitable for:

- Understanding the repo
- PR creation
- Research-paper writing
- Investor communication

The baseline mental model used:

```text
Client
→ Auth/API layer
→ Query handler/parser/planner/optimizer
→ Execution engine
→ Transaction/lock/recovery/concurrency managers
→ Buffer/index/storage/replication managers
→ OS/file/network layer
```

Read-only analysis mode: no edits, no patches, no destructive commands.

---

## 2. Repo Inventory

### Top-Level Directories

| Directory | Contents |
|-----------|----------|
| `sochdb-client/` | Embedded API: `DurableConnection`, `SochConnection` (deprecated stubs), `ContextQuery`, `VectorCollection`, `GraphOverlay`, `SemanticCache`, `TokenBudget` |
| `sochdb-core/` | Types, traits, catalog, schema, TOON codec, version chain, concurrency primitives, columnar types |
| `sochdb-storage/` | **Storage engine**: `DurableStorage`, `MvccManager`, `TxnWal`, `LscsStorage`, ARIES recovery, `ssi.rs`, group commit |
| `sochdb-query/` | SQL-92 parser (`sql.rs`), SOCH-QL executor (`soch_ql_executor.rs`), context query builder, cost optimizer scaffold |
| `sochdb-index/` | HNSW (~8K LOC), Vamana, PQ, CSR graph, SIMD distance, `lockfree_hnsw.rs`, `aosoa_tiles.rs`, embedding providers |
| `sochdb-grpc/` | Tonic gRPC server with 11 services; service implementations exist but use in-memory DashMap, not `DurableStorage` |
| `sochdb-mcp/` | MCP JSON-RPC server over stdio |
| `sochdb-fusion/` | `hybrid_retrieval.rs`, `unified_fusion.rs`, `bm25_filtered.rs` — RAG fusion logic |
| `sochdb-bench/` | Criterion microbenchmarks (vs SQLite), `bench_context_query.rs` |
| `sochdb-python/` | PyO3 FFI bindings (uninspected in depth) |
| `sochdb-wasm/` | WASM bindings (uninspected in depth) |
| `proto/` | `sochdb.proto` — 37K chars, 11 services |
| `docs/` | Markdown documentation |
| `paper-results/` | External benchmark artifacts |
| `benches/` | Additional benchmark harnesses |

### Rust Crates

16-crate workspace (confirmed from `Cargo.toml` workspace members).

### Main Entry Points

| Entry Point | File | Role |
|-------------|------|------|
| Library API | `sochdb-client/src/lib.rs` | Primary Rust SDK |
| gRPC server | `sochdb-grpc/src/main.rs` | Tonic server wiring 11 services |
| MCP server | `sochdb-mcp/src/main.rs` | JSON-RPC over stdio |
| Benchmarks | `sochdb-bench/benches/micro.rs` | Criterion microbenchmarks |
| Proto definitions | `proto/sochdb.proto` | 11 gRPC service definitions |

---

## 3. Actual SochDB Architecture

SochDB is best characterized as **an embedded database engine optimized for AI agent memory**, combining:

- **KV-oriented storage** with MVCC and WAL (`DurableStorage`)
- **Vector search** (HNSW + Vamana + PQ) with scale-aware routing
- **Graph overlay** on top of KV paths (not a native graph engine)
- **Context query / token-budget assembly** (the core AI-agent value prop)
- **SQL-92 parser + SOCH-QL executor** (parser real, executor not yet wired to storage)
- **Columnar storage engine** (`Lscs`, `ColumnarTable`) — real but not connected to SQL
- **gRPC server** with real RPC handlers but separate in-memory storage

**Confirmed facts vs inference:**

- ✅ `DurableStorage` is a real storage engine with WAL, MVCC, SSI, crash recovery
- ✅ HNSW implementation is ~8K LOC with SIMD, PQ, CSR — production depth
- ✅ Columnar storage (`lscs.rs`, `columnar.rs`) has real column buffers, SSTables, learned sparse index
- ✅ gRPC service handlers exist and process real RPCs
- ❓ SQL executor returns empty results — storage integration pending
- ❓ gRPC services use their own `DashMap`, not `DurableStorage`
- ❓ LSCS columnar engine is not connected to SQL or KV paths

---

## 4. Generic DBMS Mapping

| Generic DBMS Layer | SochDB Equivalent | Key Files | Key Structs/Functions | Status | Notes |
|--------------------|-------------------|-----------|----------------------|--------|-------|
| **Client/API layer** | `DurableConnection` (real); deprecated `SochConnection` (stubs) | `sochdb-client/src/connection.rs` | `DurableConnection`, `Connection::open()` | ✅ Functional | Use `DurableConnection`; old `SochConnection` has no durability |
| **Authentication/security layer** | Not found / not implemented | — | — | ❌ Missing | No auth in gRPC handlers |
| **Request routing layer** | `Database` kernel routes between storage, query, index | `sochdb-storage/src/database.rs` | `Database::new()`, `Database::commit()` | ⚠️ Partial | gRPC has its own routing in `main.rs` |
| **Query parser/query handler** | SQL-92 parser; SOCH-QL parser; SQL tokenizer | `sochdb-query/src/sql.rs`, `sochdb-query/src/soch_ql.rs` | `Lexer`, `Parser`, `SochQlParser` | ✅ Functional | Both parsers are real and tested |
| **Query planner** | `QueryPlan` enum with basic heuristics | `sochdb-query/src/soch_ql_executor.rs` | `QueryPlan::TableScan`, `QueryPlan::Filter` | ⚠️ Scaffold | Planner exists but relies on catalog metadata |
| **Query optimizer** | `CostOptimizer` with `ExecutionPlan` | `sochdb-query/src/cost_optimizer.rs` | `CostOptimizer`, `ExecutionPlan`, `PlanCache` | ⚠️ Scaffold | Module exists but depth unverified |
| **Execution/retrieval engine** | SOCH-QL executor; vectorized executor scaffold | `sochdb-query/src/soch_ql_executor.rs` | `SochQlExecutor::execute_plan()` | ⚠️ Partial | `execute_plan()` returns empty rows for TableScan — storage integration pending |
| **Embedding/vector integration** | Multi-provider embedding system with fallback | `sochdb-index/src/embedding/` | `EmbeddingProvider`, `FastEmbedProvider`, `LocalEmbeddingProvider`, `EmbeddingRegistry` | ✅ Functional | Feature-gated: local, fastembed (ONNX), LLM providers |
| **Transaction manager** | `DurableStorage::commit()` with group commit option | `sochdb-storage/src/durable_storage.rs` | `DurableStorage::commit()`, `DurableStorage::begin_transaction()` | ✅ Functional | Real commit path with WAL flush + MVCC commit |
| **Lock manager** | Hierarchical lock manager + advisory file lock | `sochdb-core/src/concurrency.rs`, `sochdb-storage/src/lock.rs` | `LockManager`, `ShardedLockTable` (256 shards), `IntentLock` | ✅ Functional | Intent locks (IS/IX/S/X), row-level sharded locks, optimistic versions for HNSW |
| **Concurrency control** | MVCC with `VersionChain` + SSI validation on commit | `sochdb-storage/src/durable_storage.rs` | `MvccManager`, `MvccTransaction`, `validate_ssi()` | ✅ Functional | 4-level SSI fast path (read-only bypass → single-key bypass → bloom disjoint → full exact check) |
| **Recovery/WAL/durability** | ARIES-style WAL with three-phase recovery | `sochdb-storage/src/txn_wal.rs`, `sochdb-storage/src/aries_recovery.rs` | `TxnWal`, `AriesRecoveryManager`, `crash_recovery()` | ✅ Functional | Analysis → Redo (idempotent via page LSN) → Undo (with CLRs). Fuzzy checkpoints. Tests present. |
| **Buffer/cache manager** | Memtable + epoch-based dirty list for GC | `sochdb-storage/src/durable_storage.rs` | `MvccMemTable`, `EpochDirtyList` | ✅ Functional | DashMap-based memtable with deferred sorted index; epoch-based GC |
| **Index manager** | HNSW primary; Vamana+PQ for scale; BM25 + learned sparse index | `sochdb-index/src/hnsw.rs`, `sochdb-index/src/lockfree_hnsw.rs`, `sochdb-storage/src/lscs.rs` | `HnswIndex`, `LockFreeHnsw`, `LearnedSparseIndex` | ✅ Functional | HNSW ~8K LOC; scale-aware routing (HNSW <100K, Vamana 100K+); PQ compression |
| **Storage manager** | LSM-like KV (`DurableStorage`) + LSCS columnar (`Lscs`) | `sochdb-storage/src/durable_storage.rs`, `sochdb-storage/src/lscs.rs` | `DurableStorage`, `Lscs`, `ColumnarMemtable`, `ColumnGroup` | ⚠️ Two engines | KV engine and columnar engine are separate and unconnected |
| **Replication/distributed layer** | Not found / not implemented | — | — | ❌ Missing | Single-node embedded only |
| **gRPC/network layer** | Tonic gRPC server with 11 services | `sochdb-grpc/src/main.rs`, `proto/sochdb.proto` | `KvServer`, `VectorIndexServer`, etc. | ⚠️ Partial | Handlers are real but use in-memory DashMap, not `DurableStorage` |
| **Observability/logging/metrics** | `trace_server.rs`, latency tracking in WAL | `sochdb-grpc/src/trace_server.rs`, `sochdb-grpc/src/observability.rs` | `TraceService` | ⚠️ Partial | Trace service exists; full observability depth unverified |
| **Testing/benchmarking harness** | Criterion + custom agent-memory benchmarks | `sochdb-bench/benches/micro.rs`, `sochdb-client/tests/test_comprehensive.rs` | `bench_insert_txn_rollback`, `bench_kv_vs_sqlite` | ⚠️ Partial | Microbenchmarks present; no crash-recovery integration test; no sqllogictest |

---

## 5. Design Principles Inferred

| Principle | Evidence | Confidence |
|-----------|----------|------------|
| **Embedded DB design** | `DurableConnection`, `Database::open()`, advisory file lock, single-writer | High |
| **AI-agent memory/context construction** | `ContextQuery`, `TokenBudget`, `StreamingContextAssembler`, hybrid retrieval (vector + BM25 + RRF) | High |
| **Vector retrieval/candidate generation** | HNSW ~8K LOC, Vamana fallback, PQ, SIMD, CSR graph, `VectorCollection` scale-aware routing | High |
| **Token-aware context packing** | `TokenBudget`, `ContextQueryBuilder`, `estimate_token_reduction()` | High |
| **Graph/memory relationships** | `GraphOverlay` on KV paths (`_graph/{ns}/nodes/{id}`, `_graph/{ns}/edges/...`) | High |
| **Durability/transaction claims** | WAL with fsync, ARIES recovery, MVCC, SSI, group commit | High |
| **Local-first design** | Default local embedding, offline-capable, embedded mode | High |
| **gRPC remote usage** | Proto defines 11 services, `main.rs` wires them all, Docker build | High |
| **Benchmark-driven development** | Criterion benchmarks, MemoryAgentBench references, microbenchmarks vs SQLite | Medium |
| **Research/paper alignment** | TOON format, token reduction claims, vector search benchmarks, columnar storage theory | Medium |

---

## 6. Control-Flow Traces

### 6.1 Local Embedded/Library Usage Flow

```text
User code: conn.put("ns/key", value)
→ DurableConnection::put() [connection.rs]
→ DurableStorage::begin_transaction() [durable_storage.rs]
→ TxnWal::begin_transaction() [txn_wal.rs] (writes TxnBegin record)
→ DurableStorage::write_refs() [durable_storage.rs]
  → MvccManager::record_write() (tracks write set)
  → TxnWalBuffer::append() (buffers WAL writes in memory)
  → MemTableKind::write() → MvccMemTable::write() (adds uncommitted version)
→ DurableStorage::commit() [durable_storage.rs]
  → WAL: flush_buffer() (single lock acquisition for all buffered writes)
  → WAL: append TxnCommit record
  → MvccManager::commit() → validate_ssi() (SSI validation)
  → MemTableKind::commit() → VersionChain::commit() (makes versions visible)
  → WAL: flush() + sync() (fsync for durability)
→ Result: commit_ts
```

### 6.2 gRPC Server Usage Flow

```text
Client RPC → KvServer::get() [kv_server.rs]
→ DashMap<String, Arc<NamespaceKv>> lookup
→ NamespaceKv.entries.get(key)
→ Check TTL / return value

Note: gRPC server does NOT connect to DurableStorage.
It maintains its own in-memory DashMap storage.
```

### 6.3 Benchmark Retrieval Flow

```text
Criterion benchmark: bench_hnsw_search [micro.rs]
→ HnswIndex::search() [hnsw.rs]
→ PQ distance computation with SIMD [simd_distance.rs]
→ HNSW greedy beam search through CSR graph layers
→ Returns top-K candidates
→ Benchmark compares vs brute-force ground truth
```

### 6.4 Index Build/Search Flow

```text
VectorCollection::insert() [vectors.rs]
→ if collection_size < 100K: HnswIndex::insert()
→ else: migrate to VamanaIndex with PQ
→ EmbeddingProvider::embed() (local/ONNX/LLM)
→ normalize_l2_simd() [normalize.rs]
→ Index update (lockfree or epoch-based)
```

### 6.5 Agent-Memory Context Query Flow

```text
ContextQuery::execute() [context_query.rs]
→ TokenBudget::allocate() (determines how many tokens available)
→ Hybrid retrieval: vector search + BM25 + RRF [hybrid_retrieval.rs]
→ StreamingContextAssembler::assemble() [streaming_context.rs]
→ TOON serialization (token-efficient format)
→ Returns assembled context within token budget
```

---

## 7. Data Model

### 7.1 Core Model

SochDB uses a **path-based KV model** with namespaces:

- **Key**: `Vec<u8>` path (e.g., `"mycollection/doc123"`)
- **Value**: `Vec<u8>` bytes (typically TOON-encoded or JSON)
- **Namespace**: Implicit via key prefix or explicit via API
- **Collection**: Logical grouping via key prefixes
- **Metadata**: Stored as separate KV entries or in catalog

### 7.2 Vector Data

- **Embeddings**: `Vec<f32>` (typically 384-dim for local models)
- **Storage**: PQ-compressed in HNSW index; raw vectors in `VectorCollection`
- **IDs**: String/UUID identifiers mapped to embedding slots

### 7.3 Graph Overlay

- **Nodes**: Stored at `_graph/{namespace}/nodes/{id}` as JSON
- **Edges**: Stored at `_graph/{namespace}/edges/{from}/{type}/{to}`
- **Reverse index**: `_graph/{namespace}/index/{type}/{to}`
- **Implementation**: KV-backed, not native graph engine

### 7.4 Columnar Data (LSCS)

- **Schema**: `TableSchema` with `ColumnDef` (name, type, nullable)
- **Row ID**: `u64` globally unique within table
- **Columns**: `ColumnBuffer` with null bitmaps, offsets for variable-length
- **MVCC columns**: `__txn_start`, `__txn_end` added automatically
- **Storage**: Monolithic `.sst` files with TOON magic header

### 7.5 Comparison to Other DB Types

| DB Type | SochDB Overlap | SochDB Difference |
|---------|---------------|-------------------|
| Vector DB (Pinecone, Weaviate) | HNSW, PQ, semantic search | Also has KV, transactions, WAL, SQL parser |
| Document DB (MongoDB) | Path-based keys, JSON-like values | Embeddings are first-class, token budgets, graph overlay |
| Embedded DB (SQLite, RocksDB) | Embedded, advisory lock, WAL | Vector search, AI context assembly, TOON format |
| Relational DB | SQL-92 parser, columnar storage | Not yet end-to-end; executor returns empty results |
| Graph DB (Neo4j) | Graph overlay on KV | Not a native graph engine; no Cypher/GQL |

---

## 8. Storage/Index Internals

### 8.1 Storage Files

| File | Purpose | Location |
|------|---------|----------|
| `wal.log` | Write-ahead log with ARIES records | `storage_dir/wal.log` |
| `.clean_shutdown` | Marker for clean shutdown (removed on open) | `storage_dir/.clean_shutdown` |
| `L{level}_seq{seq}.sst` | LSCS columnar SSTable files | `storage_dir/` |
| `sochdb_data/` | Default data directory | project root |

### 8.2 WAL Format

- **Record types**: `TxnBegin`, `Data`, `PageUpdate`, `TxnCommit`, `TxnAbort`, `Checkpoint`, `CheckpointEnd`, `CompensationLogRecord`, `SchemaChange`, `Savepoint`, `RollbackToSavepoint`, `TxnEnd`, `Delete`
- **Checksum**: CRC32 per record
- **Durability**: `flush()` + `sync()` (fsync) on commit
- **Recovery**: `replay_for_recovery()` replays committed txns only

### 8.3 HNSW Implementation

- **File**: `sochdb-index/src/hnsw.rs` (~8K LOC)
- **Features**: SIMD distance (AVX-512), Product Quantization, CSR graph representation, parallel construction waves, lockfree variant
- **Key structs**: `HnswIndex`, `HnswBuilder`, `HnswSearcher`
- **Scale routing**: `VectorCollection` routes <100K to HNSW, 100K+ to Vamana+PQ

### 8.4 Serialization

- **TOON format**: Custom token-efficient serialization (`sochdb-core/src/soch_codec.rs`)
- **WAL records**: Binary with LittleEndian byte order
- **Columnar SSTables**: Binary format with column headers, null bitmaps, offsets, data sections
- **Checkpoint data**: Custom binary serialization

### 8.5 Memory Layout

- **Memtable**: DashMap<Vec<u8>, VersionChain> — lockfree per-key
- **Ordered index**: Optional SkipMap or deferred sorted index
- **Columnar**: Per-column contiguous buffers with separate validity bitmaps
- **Version chain**: Linked list of versions with `ts_start`, `ts_end`, `txn_id`

### 8.6 Performance-Sensitive Areas

| Area | Optimization |
|------|-------------|
| HNSW search | SIMD distance, PQ, CSR graph |
| WAL writes | BufWriter buffering, batch flush |
| SSI validation | 4-level fast path, bloom pre-filtering |
| Memtable writes | DashMap (lockfree), deferred sorting |
| Columnar reads | O(1) fixed-size access, learned sparse index |
| Context assembly | Streaming iteration, token budget short-circuit |

### 8.7 Likely Bottlenecks

1. **gRPC server in-memory storage**: No persistence, no ACID for remote clients
2. **SQL executor empty results**: No storage integration means SQL is unusable end-to-end
3. **Memtable only**: SSTable flushing for KV engine unverified in DurableStorage (LSCS has flush)
4. **Concurrent SSI stress**: Only unit-tested, no multi-threaded write-skew detection test
5. **Group commit latency**: `EventDrivenGroupCommit` exists but latency under load unverified

### 8.8 Correctness Risks

1. **gRPC dual storage**: gRPC uses its own DashMap — data is invisible to embedded API and vice versa
2. **SQL→columnar disconnect**: SQL executor cannot read columnar data; columnar data cannot be queried via SQL
3. **SSI false positives**: Conservative SSI check (in-conflict + out-conflict → abort) may over-abort
4. **WAL replay ordering**: Recovery rebuilds memtable but SSTable state may be inconsistent if memtable was flushed before crash

---

## 9. Transactions, Consistency, Durability

### 9.1 What Is Implemented

| Feature | Evidence | File |
|---------|----------|------|
| **Atomic writes** | WAL groups all writes under a txn_id; commit record makes them durable | `txn_wal.rs` |
| **Transaction boundaries** | `begin_transaction()` → `write()` → `commit()` / `abort()` | `durable_storage.rs` |
| **Rollback** | `abort()` removes uncommitted versions from `VersionChain` | `durable_storage.rs` |
| **WAL** | `TxnWal` with CRC32, fsync, ARIES record types | `txn_wal.rs` |
| **Recovery** | `crash_recovery()`, `replay_for_recovery()`, ARIES three-phase | `txn_wal.rs`, `aries_recovery.rs` |
| **Locking** | Advisory file lock + `LockManager` with intent/row locks | `lock.rs`, `concurrency.rs` |
| **Concurrency** | MVCC with `VersionChain`, snapshot reads, `DashMap` | `durable_storage.rs` |
| **Snapshot isolation** | `snapshot_ts` per transaction; reads at snapshot timestamp | `durable_storage.rs` |
| **SSI** | `validate_ssi()` with rw-antidependency detection and dangerous-structure abort | `durable_storage.rs` |
| **Group commit** | `EventDrivenGroupCommit` batches multiple commits into single fsync | `durable_storage.rs` |

### 9.2 What Is Partial

| Feature | State | Note |
|---------|-------|------|
| **Crash-recovery integration test** | Missing | No fork+SIGKILL test exists |
| **Concurrent SSI stress test** | Missing | Only single-threaded unit tests |
| **Statement-level rollback** | Unclear | Savepoint record types exist but usage unverified |
| **Multi-process concurrent mode** | Partial | `Database::open_concurrent()` exists but stress tests missing |

### 9.3 What Is Missing Before Claiming "Transactional Agent Memory DB"

1. **End-to-end SQL execution**: `execute_plan()` returns empty rows
2. **gRPC→storage integration**: gRPC servers must use `DurableStorage`, not DashMap
3. **Crash-recovery integration test**: Fork, write, SIGKILL, verify data survives
4. **Concurrent isolation stress test**: Multiple threads, write-skew, phantom read prevention
5. **Columnar→SQL integration**: LSCS data must be queryable via SQL
6. **Replication/distributed**: Single-node only; no HA story
7. **Formal correctness proof**: No TLA+, Jepsen, or model checking

---

## 10. Benchmarks and Research Alignment

### 10.1 Benchmarks Found

| Benchmark | File | What It Measures | Comparison |
|-----------|------|------------------|------------|
| KV insert/txn/rollback | `micro.rs` | Latency of basic operations | vs SQLite |
| KV bulk insert | `micro.rs` | Throughput | vs SQLite |
| HNSW search | `micro.rs` | Vector search latency | vs brute-force |
| Context query | `bench_context_query.rs` | Token assembly latency | None |

### 10.2 External Benchmarks Referenced

- **MemoryAgentBench**: Referenced in README; external results; not reproducible from repo alone

### 10.3 Benchmark Gaps for a Serious Paper

| Claim | Evidence Needed | Status |
|-------|---------------|--------|
| "67% token reduction" | In-repo TOON vs JSON benchmark | Missing |
| "98.99% HNSW recall" | In-repo recall@K curves | Missing (external only) |
| "Transactional correctness" | Concurrent stress tests, Jepsen-style | Missing |
| "Context construction quality" | End-to-end context query benchmark with token budget | Partial |
| "Scalability" | Throughput vs data size, concurrent writers | Missing |
| "Durability" | Crash-recovery integration test | Missing |
| "SQL-92 compliance" | sqllogictest or similar | Missing |

---

## 11. Architecture Diagrams

### 11.1 Actual Architecture (As Found in Repo)

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layers                                   │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │ Rust SDK     │  │ gRPC Server │  │ MCP Server   │  │ Python   │ │
│  │ (real)       │  │ (real RPCs, │  │ (JSON-RPC)   │  │ (PyO3)   │ │
│  │              │  │  own memory)│  │              │  │          │ │
│  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  └────┬─────┘ │
└─────────┼─────────────────┼────────────────┼───────────────┼───────┘
          │                 │                │               │
          ▼                 ▼                ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Query / Context Layer                        │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐               │
│  │ SQL-92       │  │ SOCH-QL     │  │ ContextQuery │               │
│  │ Parser       │  │ Executor    │  │ / TokenBudget│               │
│  │ (real)       │  │ (empty rows)│  │ (real)       │               │
│  └──────────────┘  └─────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Index / Vector Layer                         │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐               │
│  │ HNSW         │  │ Vamana + PQ │  │ BM25 / RRF   │               │
│  │ (~8K LOC)    │  │ (scale>100K)│  │ (hybrid)     │               │
│  └──────────────┘  └─────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Storage Engines (TWO, UNCONNECTED)            │
│  ┌────────────────────────────┐    ┌──────────────────────────────┐│
│  │ KV Engine (DurableStorage)│    │ Columnar Engine (LSCS)        ││
│  │ • WAL + fsync             │    │ • ColumnarMemtable            ││
│  │ • MVCC + SSI              │    │ • SSTable flush (real)        ││
│  │ • DashMap memtable         │    │ • Temperature-aware compaction││
│  │ • Advisory file lock       │    │ • Learned sparse index        ││
│  │ • Group commit option       │    │ • Arrow-compatible layout     ││
│  └────────────────────────────┘    └──────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OS / File Layer                              │
│  • WAL log file          • SSTable files          • Advisory locks │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.2 Idealized Architecture ("SQLite for AI Agent Memory")

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layers                                   │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │ Rust SDK     │  │ gRPC Server │  │ MCP Server   │  │ Python   │ │
│  │              │  │ (backed by  │  │              │  │          │ │
│  │              │  │  DurableStg)│  │              │  │          │ │
│  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  └────┬─────┘ │
└─────────┼─────────────────┼────────────────┼───────────────┼───────┘
          │                 │                │               │
          ▼                 ▼                ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Unified Query Layer                          │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐               │
│  │ SQL-92       │  │ SOCH-QL     │  │ ContextQuery │               │
│  │ Parser       │  │ Planner     │  │ / TokenBudget│               │
│  │              │  │ → Executor  │  │              │               │
│  │              │  │    (wired)   │  │              │               │
│  └──────────────┘  └─────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Unified Execution Engine                     │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐               │
│  │ Vectorized   │  │ Index       │  │ Hybrid       │               │
│  │ Executor     │  │ Manager     │  │ Retrieval    │               │
│  │ (columnar)   │  │ (HNSW/PQ)   │  │ (RRF)        │               │
│  └──────────────┘  └─────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Unified Storage Engine                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Transactional Storage                       │ │
│  │  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │ │
│  │  │ WAL +      │  │ MVCC     │  │ SSI      │  │ Lock       │ │ │
│  │  │ Recovery   │  │ Versions │  │ Validate │  │ Manager    │ │ │
│  │  └────────────┘  └──────────┘  └──────────┘  └────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │           Unified Memtable + Columnar SSTables          │ │ │
│  │  │     (KV paths + columnar tables in same engine)         │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. PR Opportunities

### PR 1: Wire SQL Executor to DurableStorage

- **Title**: `feat: connect SOCH-QL executor to DurableStorage for real query results`
- **Why it matters**: Without this, SQL is aspirational. The #1 gap.
- **Files**: `sochdb-query/src/soch_ql_executor.rs`, `sochdb-storage/src/durable_storage.rs`, `sochdb-client/src/connection.rs`
- **Difficulty**: Medium-Hard (2-3 weeks)
- **Founder/investor signal**: Very High — makes SQL claims defensible
- **Research-paper value**: Very High — proves SQL-92 story
- **Tests/benchmarks needed**: sqllogictest subset, SELECT/INSERT/UPDATE/DELETE/JOIN integration tests
- **Risks**: May expose memtable scalability limits for large table scans

### PR 2: Wire gRPC Services to DurableStorage

- **Title**: `feat: route gRPC KV and VectorIndex services through DurableStorage`
- **Why it matters**: gRPC has real handlers but uses in-memory DashMap. This makes remote API durable.
- **Files**: `sochdb-grpc/src/kv_server.rs`, `sochdb-grpc/src/vector_index_server.rs`, `sochdb-storage/src/durable_storage.rs`
- **Difficulty**: Medium (2 weeks)
- **Founder/investor signal**: Very High — makes Docker/gRPC story real
- **Research-paper value**: Low-Medium — shows production readiness
- **Tests/benchmarks needed**: gRPC integration tests against DurableStorage
- **Risks**: Performance regression from DashMap (in-memory) to DurableStorage (WAL+MVCC)

### PR 3: Add Crash-Recovery Integration Test

- **Title**: `test: fork+SIGKILL crash recovery integration test`
- **Why it matters**: No amount of WAL unit tests equals one crash test. Proves durability.
- **Files**: `sochdb-storage/tests/crash_recovery.rs` (new), `sochdb-storage/src/txn_wal.rs`
- **Difficulty**: Medium (3-5 days)
- **Founder/investor signal**: Very High — proves ACID durability claim
- **Research-paper value**: High — standard for any DB paper
- **Tests/benchmarks needed**: Fork process, write txn, SIGKILL, verify committed data survives, uncommitted data does not
- **Risks**: May fail and reveal recovery bugs

### PR 4: Add In-Repo HNSW Recall Benchmark

- **Title**: `bench: add HNSW recall@K vs brute-force benchmark`
- **Why it matters**: README claims 98.99% recall but proof is external.
- **Files**: `sochdb-bench/benches/hnsw_recall.rs` (new), `sochdb-index/src/hnsw.rs`
- **Difficulty**: Easy (2-3 days)
- **Founder/investor signal**: Medium — shows vector quality is measurable
- **Research-paper value**: Very High — recall curves are table-stakes for vector-search papers
- **Tests/benchmarks needed**: Generate 10K-100K vectors, brute-force k-NN ground truth, measure recall@K across ef_search values
- **Risks**: None

### PR 5: Add TOON vs JSON Token Comparison Benchmark

- **Title**: `bench: add TOON vs JSON token efficiency benchmark`
- **Why it matters**: The "67% token reduction" is the marquee claim.
- **Files**: `sochdb-bench/benches/toon_vs_json.rs` (new), `sochdb-core/src/soch_codec.rs`
- **Difficulty**: Easy (2-3 days)
- **Founder/investor signal**: High — proves core differentiator
- **Research-paper value**: Very High — central claim must be quantified
- **Tests/benchmarks needed**: Datasets of varying sizes, serialize to TOON and JSON, count tokens, report savings
- **Risks**: None

### PR 6: Add Concurrent SSI Stress Test

- **Title**: `test: concurrent SSI stress test with write-skew detection`
- **Why it matters**: SSI is real but only unit-tested. A concurrent stress test proves it works under load.
- **Files**: `sochdb-storage/tests/ssi_stress.rs` (new), `sochdb-storage/src/durable_storage.rs`
- **Difficulty**: Medium (1 week)
- **Founder/investor signal**: High — proves isolation claims under concurrency
- **Research-paper value**: Very High — standard DB papers include isolation stress tests
- **Tests/benchmarks needed**: Write-skew scenario (two txns read overlapping data, write non-overlapping keys); 10K iterations; verify 0 anomalies
- **Risks**: May expose SSI false positives under concurrency

### PR 7: Unify KV and Columnar Storage Engines

- **Title**: `feat: integrate LSCS columnar engine into DurableStorage unified memtable`
- **Why it matters**: Two unconnected storage engines is architectural debt. Unifying them enables SQL over columnar data.
- **Files**: `sochdb-storage/src/durable_storage.rs`, `sochdb-storage/src/lscs.rs`, `sochdb-core/src/columnar.rs`
- **Difficulty**: Hard (4-6 weeks)
- **Founder/investor signal**: Medium — reduces technical debt
- **Research-paper value**: Medium — enables "columnar storage" claim for SQL queries
- **Tests/benchmarks needed**: Columnar insert → SQL query → correct results
- **Risks**: High; touches core storage engine

### PR 8: Add sqllogictest Compatibility Suite

- **Title**: `test: add sqllogictest compatibility suite`
- **Why it matters**: "SQL-92" is a strong claim; sqllogictest is the industry standard.
- **Files**: `sochdb-query/tests/sqllogictest/` (new), `sochdb-query/src/sql.rs`
- **Difficulty**: Medium (1-2 weeks)
- **Founder/investor signal**: High — shows SQL is real
- **Research-paper value**: Medium — proves SQL-92 claims
- **Tests/benchmarks needed**: Port subset of SQLite's sqllogictest; cover SELECT/INSERT/UPDATE/DELETE/JOIN
- **Risks**: Will likely fail initially until executor is wired to storage

### PR 9: Add End-to-End Context Query Benchmark

- **Title**: `bench: end-to-end context query benchmark with token budgeting`
- **Why it matters**: SochDB's reason to exist is token-efficient context for AI agents. No end-to-end benchmark means the value prop is unsupported.
- **Files**: `sochdb-bench/benches/context_query.rs` (new), `sochdb-client/src/context_query.rs`
- **Difficulty**: Medium (1 week)
- **Founder/investor signal**: Very High — proves core AI-memory value prop
- **Research-paper value**: Very High — the paper's central claim
- **Tests/benchmarks needed**: Insert 1K documents with embeddings, run ContextQuery with 4K token budget, measure wall-clock time, token count, retrieved chunks, token utilization ratio
- **Risks**: None

### PR 10: Remove Deprecated Stubs from sochdb-client

- **Title**: `chore: remove deprecated WalStorageManager and TransactionManager stubs`
- **Why it matters**: These stubs are marked "NO durability" and "NO MVCC." A reviewer reading the code would find them and question ACID claims.
- **Files**: `sochdb-client/src/connection.rs` (lines 136-262)
- **Difficulty**: Easy (1 day)
- **Founder/investor signal**: Medium — shows engineering hygiene
- **Research-paper value**: Low but removes a credibility trap
- **Tests/benchmarks needed**: All existing tests should still pass
- **Risks**: None

---

## 13. Final Verdict

### What SochDB Is Today

SochDB is a **collection of real, well-implemented database subsystems** that need integration work to form a cohesive DBMS:

- ✅ **Real storage engine**: WAL + MVCC + SSI + ARIES recovery + group commit
- ✅ **Real vector search**: HNSW (~8K LOC), Vamana, PQ, SIMD
- ✅ **Real columnar storage**: Arrow-compatible layout, temperature-aware compaction, learned sparse index
- ✅ **Real lock manager**: Intent locks, sharded row locks, optimistic concurrency, epoch reclamation
- ✅ **Real embedding pipeline**: Multi-provider, fallback chains, hierarchical embedding
- ✅ **Real context assembly**: Token budgets, hybrid retrieval, streaming assembly
- ✅ **Real SQL parser**: SQL-92 lexer + parser, SOCH-QL planner scaffold
- ⚠️ **Partial SQL execution**: Executor exists but returns empty results — storage integration pending
- ⚠️ **Partial gRPC**: Real RPC handlers but use in-memory storage, not `DurableStorage`

### What SochDB Is Not Yet

- ❌ An end-to-end SQL database (parser real, executor not wired to storage)
- ❌ A remote-accessible transactional database (gRPC uses in-memory storage)
- ❌ A unified storage engine (KV and columnar engines are separate)
- ❌ A stress-tested transactional system (no concurrent SSI stress test, no crash-recovery integration test)
- ❌ A distributed/replicated database (single-node only)

### Strongest Technical Differentiators

1. **Token-efficient context assembly**: `TokenBudget` + `StreamingContextAssembler` + TOON format — genuinely novel for an embedded DB
2. **Scale-aware vector routing**: `VectorCollection` auto-selects HNSW (<100K) vs Vamana+PQ (100K+) — smart engineering
3. **Production SSI in embedded DB**: 4-level fast-path SSI with bloom pre-filtering — unusual depth for an embedded engine
4. **Multi-provider embedding with fallback**: Local/ONNX/LLM with graceful degradation

### Weakest Technical Gaps

1. **SQL execution not wired to storage**: The #1 blocker for "database" claims
2. **gRPC not wired to storage**: Makes remote API non-durable
3. **No crash-recovery integration test**: Durability claim is unverified under real crashes
4. **No concurrent SSI stress test**: Isolation claims are unit-tested only
5. **Two unconnected storage engines**: KV and columnar are separate worlds

### Files to Study First

| Priority | File | Why |
|----------|------|-----|
| 1 | `sochdb-storage/src/durable_storage.rs` | The heart of the engine — commit path, MVCC, SSI |
| 2 | `sochdb-storage/src/txn_wal.rs` | WAL implementation and recovery |
| 3 | `sochdb-index/src/hnsw.rs` | Deepest module, ~8K LOC of vector search |
| 4 | `sochdb-client/src/connection.rs` | Primary API surface |
| 5 | `sochdb-query/src/soch_ql_executor.rs` | SQL execution gap (returns empty rows) |
| 6 | `sochdb-grpc/src/kv_server.rs` | gRPC dual-storage problem |
| 7 | `sochdb-storage/src/lscs.rs` | Columnar engine (real but unconnected) |
| 8 | `sochdb-core/src/concurrency.rs` | Lock manager depth |
| 9 | `sochdb-storage/src/aries_recovery.rs` | ARIES recovery implementation |
| 10 | `sochdb-index/src/embedding/mod.rs` | Embedding provider architecture |

### Safe Wording for README/Paper

- ✅ "SochDB is an embedded database engine with WAL, MVCC, and snapshot isolation optimized for AI agent memory."
- ✅ "SochDB includes a SQL-92 parser and query planner; end-to-end SQL execution via the storage engine is in active development."
- ✅ "SochDB provides vector search with HNSW and product quantization, with scale-aware backend selection."
- ✅ "SochDB's context query system assembles token-budgeted context from hybrid retrieval."
- ❌ "SochDB is a fully transactional SQL database" — SQL execution is not yet wired to storage
- ❌ "SochDB supports remote durable access via gRPC" — gRPC uses in-memory storage
- ❌ "SochDB has been stress-tested for transactional correctness" — only unit tests exist

---

## 14. Key Corrections From Self-Critique

During this run, several initial claims were corrected after deeper inspection:

1. **SSI was initially rated "enum only"** — **WRONG**: `MvccManager::validate_ssi()` is a full 4-level SSI implementation running on every ReadWrite commit.
2. **"Two SSI implementations" was misleading** — There is one active SSI in `MvccManager`. The standalone `ssi.rs` is a richer evolution not yet wired.
3. **"Columnar is result format only" was partially wrong** — `lscs.rs` has real `ColumnarMemtable`, `ColumnGroup` SSTables, and `LearnedSparseIndex`. `columnar.rs` has Arrow-compatible `TypedColumn`.
4. **"Lock manager = advisory file lock + RwLocks" was incomplete** — `concurrency.rs` has a full hierarchical lock manager with intent locks and 256-shard row locks.
5. **"gRPC services might be stubs" was partially true** — Handlers are real but use in-memory DashMap, not `DurableStorage`.
6. **"Embedding providers are stubs" was wrong** — Multi-provider system with local, ONNX (FastEmbed), and LLM providers, plus fallback chains.
7. **"ARIES recovery unclear" was wrong** — Full three-phase ARIES with analysis, redo, undo, CLRs, and fuzzy checkpoints.

---

*End of architectural observations. Generated on 2026-05-29 from an OpenCode Go deep-analysis run.*
