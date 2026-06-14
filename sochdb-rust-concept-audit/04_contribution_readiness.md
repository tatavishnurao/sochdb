# 04 Contribution Readiness

What you can realistically contribute based on your current Rust level, with exact files and task types.

---

## Level 1: Beginner (Just Finished The Rust Book)

### What You Know
- Variables, ownership, borrowing, structs, enums, `match`, `Option`, `Result`, `Vec`, `HashMap`

### Where You Can Contribute
| Area | Exact Files | Task Examples |
|------|-------------|---------------|
| **Examples** | `examples/rust/01_basic_database.rs`, `02_transactions.rs`, `03_vector_search.rs`, `04_sql_queries.rs` | Update examples, add comments, fix typos |
| **Tools CLI** | `sochdb-tools/src/cli.rs`, `sochdb-tools/src/error.rs` | Add CLI flags, improve error messages |
| **Tests** | Any `#[cfg(test)]` module | Add unit tests for edge cases |
| **Docs** | Any `//!` or `///` doc comment | Fix stale documentation, add missing examples |

### What You Must NOT Touch Yet
- Any file with `unsafe` blocks
- `sochdb-index/src/simd_distance.rs`
- `sochdb-storage/src/ffi.rs`
- `sochdb-core/src/transaction_typestate.rs` (read-only)

---

## Level 2: Intermediate (Comfortable with Lifetimes, Traits, `Arc`)

### What You Know
- Lifetimes, generic structs, traits, `impl Trait for`, `Arc`, `Box`, `dyn Trait`, iterators, closures

### Where You Can Contribute
| Area | Exact Files | Task Examples |
|------|-------------|---------------|
| **Core Format** | `sochdb-core/src/soch.rs` | Add new `SochValue` variants, improve parser |
| **Core Schema** | `sochdb-core/src/memory_schema.rs`, `sochdb-core/src/schema_bridge.rs` | Add schema migration helpers |
| **Client SDK** | `sochdb-client/src/connection.rs`, `sochdb-client/src/crud.rs` | Add convenience methods, batching helpers |
| **Query Engine** | `sochdb-query/src/calc.rs`, `sochdb-query/src/soch_ql.rs` | Add SQL functions, SochQL operators |
| **Fusion** | `sochdb-fusion/src/query.rs`, `sochdb-fusion/src/pipeline.rs` | Add result fusion strategies |
| **Kernel (Safe Parts)** | `sochdb-kernel/src/plugin.rs`, `sochdb-kernel/src/plugin_manifest.rs` | Add plugin metadata, manifest validation |

### What You Must Be Cautious About
- `sochdb-core/src/transaction_typestate.rs` — can read and add safe methods, but do not change the type-state mechanics without deep review
- `sochdb-storage/src/database.rs` — understand MVCC before modifying

---

## Level 3: Advanced (Comfortable with `unsafe`, Atomics, Concurrency)

### What You Know
- `unsafe` blocks, raw pointers, atomics (`Ordering`), `crossbeam`, SIMD basics, FFI

### Where You Can Contribute
| Area | Exact Files | Task Examples |
|------|-------------|---------------|
| **Storage Engine** | `sochdb-storage/src/lscs.rs`, `sochdb-storage/src/durable_storage.rs`, `sochdb-storage/src/mvcc_new.rs` | Add compaction heuristics, improve MVCC snapshot logic |
| **Lock-Free Structures** | `sochdb-storage/src/lockfree_memtable.rs`, `sochdb-index/src/lockfree_hnsw.rs` | Optimize read paths, add hazard pointer tests |
| **SIMD Kernels** | `sochdb-vector/src/simd/`, `sochdb-index/src/simd_distance.rs`, `sochdb-index/src/simd_batch_distance.rs` | Add NEON paths, optimize prefetch distances |
| **FFI / Bindings** | `sochdb-storage/src/ffi.rs`, `sochdb-python/src/lib.rs` | Add C API functions, fix Python binding memory leaks |
| **WAL / Recovery** | `sochdb-storage/src/aries_recovery.rs`, `sochdb-storage/src/wal_integration.rs` | Add recovery tests, improve checkpoint logic |
| **Index Structures** | `sochdb-index/src/hnsw.rs`, `sochdb-index/src/vamana.rs`, `sochdb-index/src/hnsw_staged.rs` | Add graph construction optimizations |
| **Vector Search** | `sochdb-vector/src/query/`, `sochdb-vector/src/hybrid.rs` | Add reranking strategies, hybrid search tuning |

---

## Level 4: Expert (Can Design New Lock-Free Algorithms or Write Correct SIMD)

### What You Know
- Lock-free algorithm design, memory models, cache-line optimization, vector intrinsics on multiple architectures

### Where You Can Contribute
| Area | Exact Files | Task Examples |
|------|-------------|---------------|
| **Core Concurrency** | `sochdb-core/src/reclamation.rs`, `sochdb-core/src/epoch_gc.rs` | Design new reclamation strategies |
| **Storage Lock-Free** | `sochdb-storage/src/lockfree_epoch.rs`, `sochdb-storage/src/concurrent_art.rs` | Implement ART node types, epoch tracking improvements |
| **Index SIMD** | `sochdb-index/src/predicated_simd.rs`, `sochdb-index/src/simd_scan.rs` | Write AVX-512 kernels, SVE paths for aarch64 |
| **Vector Engine** | `sochdb-vector/src/portable_simd.rs`, `sochdb-vector/src/dispatch.rs` | Add runtime CPU detection, new kernel dispatch |
| **Kernel Runtime** | `sochdb-kernel/src/wasm_sandbox_runtime.rs`, `sochdb-kernel/src/wasm_host_abi.rs` | Implement real wasmtime integration |

---

## Universal Contribution Rules for This Repo

1. **Read `Cargo.toml` workspace root first.** It documents the feature-flag registry and runtime design decisions (sync-first storage, optional tokio).
2. **Do not add new dependencies without justification.** The workspace is carefully curated.
3. **Run `cargo clippy` and fix warnings.** The repo has a history of clippy-driven PRs (see `PRs/` folder).
4. **Add tests for any new logic.** There are extensive `#[cfg(test)]` modules; follow their style.
5. **Document public APIs with `///`.** Every `pub` item should have a doc comment.
6. **Feature-gate optional behavior.** Follow the pattern in `Cargo.toml` (e.g., `async = ["tokio"]`).

---

## Quick Self-Assessment

Can you read and explain the following snippet from `sochdb-core/src/transaction_typestate.rs:160-167`?

```rust
pub struct Transaction<State: TransactionState, Mode: TransactionMode = ReadWrite> {
    inner: TransactionInner,
    _state: PhantomData<State>,
    _mode: PhantomData<Mode>,
}
```

| Response | Your Level |
|----------|------------|
| "It's a generic struct with two type parameters." | Beginner |
| "The `PhantomData` fields make the type parameter part of the type without storing data, enabling type-state." | Intermediate |
| "This encodes a state machine in the type system; `commit(self)` consumes the `Active` state and returns `Committed`, making invalid transitions unrepresentable." | Advanced |
| "I could add a `Preparing` state with 2PC semantics by extending `TransactionState` and adding a sealed trait impl." | Expert |
