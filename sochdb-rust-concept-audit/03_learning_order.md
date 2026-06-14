# 03 Learning Order — What to Learn First to Contribute

This order is based on **frequency of appearance** and **prerequisite dependency** inside SochDB. Learn in this order; do not skip.

---

## Tier 0: Absolute Prerequisites (Learn These First)

Without these, you cannot read any file in the repo.

### 1. Ownership, Borrowing, `&` vs `&mut`
- **Why:** Every function signature uses them.
- **Study:** `sochdb-core/src/soch.rs` — `SochRow::parse()` takes `&str` and `&SochSchema`; `SochTable::format()` uses `&self`.
- **Key insight:** The entire TOON parser is a lesson in borrowing.

### 2. Structs and Enums
- **Why:** 2,000+ definitions.
- **Study:** `sochdb-core/src/soch.rs` — `SochValue`, `SochType`, `SochSchema`, `SochRow`, `SochTable`.
- **Key insight:** Learn how enums with data (`SochValue::Text(String)`) replace class hierarchies.

### 3. `impl` Blocks
- **Why:** 2,078 occurrences.
- **Study:** `sochdb-core/src/soch.rs` — `impl SochValue`, `impl SochSchema`, `impl SochRow`.
- **Key insight:** Methods are defined separately from data; this is not OOP.

### 4. `match` Expressions & Pattern Matching
- **Why:** Every enum variant is handled with `match`.
- **Study:** `sochdb-core/src/soch.rs:120-168` — `fmt::Display for SochValue` is one giant `match`.

### 5. `Option<T>` and `Result<T, E>`
- **Why:** Error handling backbone; `?` operator everywhere.
- **Study:** `sochdb-core/src/error.rs` — `pub type Result<T> = std::result::Result<T, SochDBError>;`
- **Study:** `sochdb-core/src/soch.rs:359` — `pub fn get(&self, index: usize) -> Option<&SochValue>`

### 6. `Vec<T>`, `HashMap<K, V>`, `String`
- **Why:** 3,975 `Vec<`, 554 `HashMap<`.
- **Study:** Every file.

### 7. `derive` Macros (`Debug`, `Clone`, `PartialEq`, `Serialize`)
- **Why:** 1,655 occurrences.
- **Study:** Any struct declaration in the repo.

---

## Tier 1: Intermediate Patterns (Needed for Non-Trivial Contributions)

### 8. Lifetimes (`<'a>`)
- **Why:** 382 occurrences. Zero-copy parsing, FFI, and guards all use them.
- **Study:** `sochdb-core/src/tbp.rs` — `NullBitmap<'a>`, `RowView<'a>`, `TbpReader<'a>`.
- **Exercise:** Understand why `TbpReader` cannot outlive its input byte slice.

### 9. Traits (`trait` / `impl Trait for`)
- **Why:** 80 `pub trait`, 2,078 impl blocks.
- **Study:** `sochdb-core/src/soch.rs:624` — `trait ColumnAccess` and its `write_value` method.
- **Study:** `sochdb-core/src/transaction_typestate.rs:196` — `trait TransactionStorage`.
- **Key insight:** Traits are how SochDB abstracts storage backends.

### 10. `Box<T>` and Basic Smart Pointers
- **Why:** Heap allocation, recursive types.
- **Study:** `sochdb-core/src/soch.rs:181` — `Array(Box<SochType>)` and `Optional(Box<SochType>)`.

### 11. Closures and `Iterator`
- **Why:** 44 custom iterators, 135 move closures.
- **Study:** `sochdb-core/src/soch.rs:655` — `impl<'a, C: ColumnAccess> Iterator for SochCursor<'a, C>`.
- **Study:** `sochdb-storage/src/ffi.rs:639` — `.map(Ok)` closure inside FFI.

### 12. `Arc<T>` and Shared Ownership
- **Why:** 610 occurrences. Multi-threaded sharing is everywhere.
- **Study:** `sochdb-core/src/reclamation.rs` — `hazard: Arc<HazardDomain>`.
- **Study:** `sochdb-kernel/src/wasm_runtime.rs` — `Arc<WasmPluginInstance>`.

### 13. Generic Structs and Functions (`<T>`, `<K, V>`)
- **Why:** Generics appear in almost every crate.
- **Study:** `sochdb-core/src/epoch_gc.rs` — `EpochGC<K, V>`.
- **Study:** `sochdb-core/src/transaction_typestate.rs:160` — `Transaction<State, Mode>`.

---

## Tier 2: Advanced Patterns (Needed for Core/Storage/Index/Vector)

### 14. `PhantomData` and the Type-State Pattern
- **Why:** 69 occurrences. The transaction API is built on this.
- **Study:** `sochdb-core/src/transaction_typestate.rs` — `Transaction<Active, ReadWrite>` uses `PhantomData` to prevent use-after-commit at compile time.
- **Key insight:** This is the most advanced "everyday" pattern in the repo.

### 15. `unsafe` Rust (Reading It, Not Writing It Yet)
- **Why:** 1,070 occurrences. You will read `unsafe` even if you don't write it.
- **Study:** `sochdb-storage/src/ffi.rs` — see how `unsafe` is used to cross the FFI boundary safely with null checks.
- **Study:** `sochdb-vector/src/simd/mod.rs` — see how `SimdBackend` trait abstracts `unsafe` intrinsics.
- **Rule:** Never write `unsafe` until you can explain every line of an existing `unsafe` block.

### 16. `dyn Trait` (Trait Objects)
- **Why:** 298 occurrences. Storage backends and callbacks use them.
- **Study:** `sochdb-core/src/transaction_typestate.rs:182` — `Arc<dyn TransactionStorage>`.
- **Study:** `sochdb-client/src/routing.rs` — `Arc<dyn Fn(...) + Send + Sync>`.

### 17. `Drop` and RAII Guards
- **Why:** 44 impls. Locks, hazard pointers, epochs.
- **Study:** `sochdb-core/src/reclamation.rs` — `impl<'a> Drop for HazardGuard<'a>`.
- **Study:** `sochdb-core/src/concurrency.rs` — `impl<'a> Drop for WriteGuard<'a>`.

### 18. Atomics and Memory Orderings
- **Why:** 503 occurrences. Lock-free data structures.
- **Study:** `sochdb-kernel/src/wasm_runtime.rs` — `AtomicU64` with `Ordering::Acquire`, `Release`, `AcqRel`, `Relaxed`.
- **Key insight:** Understanding the difference between `Relaxed` and `SeqCst` is required for `sochdb-index`.

### 19. `RwLock<T>` and `Mutex<T>` (including `parking_lot`)
- **Why:** 308 + 103 occurrences.
- **Study:** `sochdb-kernel/src/wasm_runtime.rs` — `RwLock<HashMap<String, Arc<WasmPluginInstance>>>`.
- **Note:** The codebase uses `parking_lot::RwLock` in hot paths for performance.

### 20. `where` Clauses and Trait Bounds
- **Why:** 178 occurrences.
- **Study:** `sochdb-core/src/transaction_typestate.rs:556` — `impl<State: TransactionState, Mode: TransactionMode> std::fmt::Debug for Transaction<State, Mode>`.

---

## Tier 3: Specialist Topics (Only for Specific Crates)

### 21. `macro_rules!` (Declarative Macros)
- **Why:** 5 occurrences. SIMD profiling, filter IR DSL.
- **Study:** `sochdb-index/src/profiling.rs` — `macro_rules! profile_section`.
- **Study:** `sochdb-query/src/filter_ir.rs` — `macro_rules! filter_ir`.

### 22. `extern "C"` and FFI
- **Why:** 105 occurrences. `sochdb-storage/src/ffi.rs` is 1,500+ lines of FFI.
- **Study:** `sochdb-storage/src/ffi.rs` — `#[unsafe(no_mangle)] pub unsafe extern "C" fn sochdb_open(...)`.
- **Needed for:** Python bindings, C API, any SDK work.

### 23. `async` / `await` and `tonic`
- **Why:** 209 occurrences, but only in gRPC/MCP/server crates.
- **Study:** `sochdb-grpc/src/lib.rs` — `tonic::include_proto!("sochdb.v1")`.
- **Study:** `sochdb-vector/src/async_rotation.rs` — `BoundedChannel<T>`.
- **Needed for:** Server-side work only. Core storage is explicitly sync-first.

### 24. `crossbeam` (Epoch Reclamation, SkipLists, Channels)
- **Why:** 31 occurrences.
- **Needed for:** `sochdb-storage/src/lockfree_memtable.rs`, `sochdb-index/src/lockfree_hnsw.rs`.

### 25. SIMD Intrinsics (`core::arch`)
- **Why:** Pure Rust SIMD in `sochdb-vector/src/simd/` and `sochdb-index/src/simd_distance.rs`.
- **Needed for:** Vector search performance work only.

---

## Tier 4: What You Can Skip for Now

These are either absent or used in very narrow areas:

- `const generics` — not present
- GATs — not clearly present
- `Pin<>` / `pin!` — almost absent (2 occurrences)
- `impl Trait` in return position — not found
- `no_std` — not used
- `Rc<>` — not used (only `Arc`)
- `LazyLock` — not used (`OnceLock` preferred)
- `rayon` — minimal (benchmarks only)
- `wasmtime` — only 5 occurrences (simulated, not fully integrated)

---

## Suggested Study Path (Week by Week)

| Week | Topic | File to Read |
|------|-------|--------------|
| 1 | Ownership, structs, enums, match, Option/Result | `sochdb-core/src/soch.rs` |
| 2 | impl blocks, traits, collections, iterators | `sochdb-core/src/soch.rs` + `sochdb-tools/src/builder.rs` |
| 3 | Lifetimes, `Box<T>`, basic generics | `sochdb-core/src/tbp.rs` |
| 4 | `Arc<T>`, shared ownership, `RwLock` | `sochdb-kernel/src/wasm_runtime.rs` |
| 5 | `PhantomData`, type-state pattern | `sochdb-core/src/transaction_typestate.rs` |
| 6 | `unsafe` reading, `dyn Trait`, `Drop` | `sochdb-storage/src/ffi.rs` + `sochdb-core/src/reclamation.rs` |
| 7 | Atomics, `where` clauses, closures | `sochdb-index/src/atomic_entry_point.rs` |
| 8 | `macro_rules!`, `async` (if server interest) | `sochdb-query/src/filter_ir.rs` + `sochdb-grpc/src/lib.rs` |
| 9+ | Specialist: SIMD, `crossbeam`, FFI deep dive | Pick a module from `sochdb-vector/src/simd/` or `sochdb-storage/src/ffi.rs` |
