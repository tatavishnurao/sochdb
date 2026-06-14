# 00 Summary — SochDB Rust Concept Audit

**Repository:** https://github.com/sochdb/sochdb  
**Commit audited:** `c614ec37d88cc74965d1d06d52c435517aed9d56`  
**Date:** 2026-05-24  
**Edition:** 2024 (Rust 1.85+)

## What This Audit Is

A source-code-driven inventory of every Rust concept that appears in the SochDB codebase, with exact file paths, line counts, and concrete examples. No textbook advice — only what is actually used.

## The Codebase at a Glance

- **~15 Rust crates** in a Cargo workspace
- **~400+ modules** (774 `mod` declarations)
- **~2,078 `impl` blocks**
- **~1,789 `pub struct` definitions** (including prost-generated)
- **~442 `pub enum` definitions**
- **~80 `pub trait` definitions**
- **~1,070 `unsafe` occurrences** — this is an advanced codebase

## Major Crates

| Crate | Role | Complexity |
|-------|------|------------|
| `sochdb-core` | TOON format, transactions, VFS, schemas | Intermediate–Advanced |
| `sochdb-storage` | LSM/storage engine, WAL, MVCC, FFI | Advanced |
| `sochdb-index` | HNSW, Vamana, vector indices, SIMD | Advanced |
| `sochdb-vector` | ANN search engine, SIMD kernels, filtering | Advanced |
| `sochdb-grpc` | gRPC server (tonic), async services | Intermediate |
| `sochdb-client` | SDK, routing, tracing, recovery | Intermediate |
| `sochdb-query` | SQL/SochQL execution, optimizer | Intermediate–Advanced |
| `sochdb-kernel` | Plugin system, WASM runtime, page manager | Advanced |
| `sochdb-fusion` | Query result fusion, pipeline | Intermediate |
| `sochdb-mcp` | MCP protocol server (async) | Intermediate |
| `sochdb-tools` | CLI, bulk ingest, builder patterns | Beginner–Intermediate |
| `sochdb-wasm` | WASM HNSW core | Intermediate |
| `sochdb-python` | PyO3 bindings | Intermediate |

## Concept Heat Map (by Frequency & Depth)

| Concept | Frequency | Depth | Where It Matters |
|---------|-----------|-------|------------------|
| Structs + Enums | Ubiquitous | Basic | Every file |
| `impl` blocks | Ubiquitous | Basic | Every file |
| `derive` macros | 1,655 | Basic | Every file |
| `Vec<T>`, `HashMap<K,V>` | 3,975 / 554 | Basic | Every file |
| `Arc<T>` | 610 | Intermediate | Concurrency, storage backends |
| `unsafe` blocks | 1,070 | Advanced | SIMD, FFI, zero-copy |
| Lifetimes (`<'a>`) | 382 | Intermediate | Parsers, zero-copy, FFI |
| Traits (`pub trait`) | 80 | Intermediate | Storage backends, index abstractions |
| `dyn Trait` | 298 | Intermediate | Callbacks, storage, routing |
| `PhantomData` | 69 | Advanced | Type-state transactions |
| `where` clauses | 178 | Intermediate | Generic bounds |
| `Drop` impls | 44 | Intermediate | RAII guards, hazard pointers |
| `macro_rules!` | 5 | Intermediate | SIMD macros, DSLs |
| `async` / `await` | 209 / 220 | Intermediate | gRPC, MCP, servers |
| `extern "C"` / FFI | 105 | Advanced | C API, Python bindings |
| `#[repr(C)]` | ~10+ | Intermediate | FFI structs |
| `parking_lot::RwLock` | 308 (total RwLock) | Intermediate | Lock-free memtable, registries |
| Atomics | 503 | Advanced | Lock-free data structures |
| `crossbeam` | 31 | Advanced | Skip lists, channels |
| `Fn` / `FnOnce` / `FnMut` | 104 / 45 / 10 | Intermediate | Routing, callbacks |
| `move` closures | 135 | Intermediate | Async, threading |
| Custom `Iterator` impls | 44 | Intermediate | Scans, graph traversal |
| `const fn` | 51 | Intermediate | Constants, bit manipulation |
| `thiserror::Error` | 23 | Basic | Error enums |
| `serde` derive | 233 | Basic | Serialization |

## What's NOT in the Repo

The following concepts are either **not present** or **not clearly used**:

- `const generics` (e.g., `const N: usize`) — not found
- Generic Associated Types (GATs) — not clearly present
- `impl Trait` in return position (RPIT) — not found
- `Pin<>` / `pin!` — only 2 occurrences (almost absent)
- `no_std` — not used
- `Rc<>` — not used (only `Arc`)
- `LazyLock` — not used (`OnceLock` preferred)
- Native `async fn` in traits — uses `async-trait` crate optionally
- `IntoIterator` implementations — not found
- `rayon` — minimal (22 occurrences, mostly benchmarks)

## Bottom Line for a Learner

This is **not a beginner Rust codebase**. You can contribute to peripheral crates (`sochdb-tools`, examples) with intermediate knowledge, but to touch `sochdb-index`, `sochdb-vector`, `sochdb-storage`, or `sochdb-kernel`, you need solid understanding of:

1. Ownership, borrowing, lifetimes
2. Traits, trait objects (`dyn`), and generic bounds
3. `unsafe` Rust (at least enough to read it)
4. Concurrency primitives (`Arc`, `RwLock`, atomics, `crossbeam`)
5. The module system and workspace structure
