# SochDB Rust Concept Audit — Raw Ripgrep Results

Generated from commit `c614ec37d88cc74965d1d06d52c435517aed9d56`.

## Core Language Constructs

| Pattern | Count | Notes |
|---------|-------|-------|
| `pub struct` | 1789 | Many generated (prost/tonic), but also many hand-written |
| `struct` (all) | 201 | Excluding pub |
| `pub enum` | 442 | Heavy use for errors, states, types |
| `enum` (all) | 21 | |
| `pub trait` | 80 | |
| `trait` (all) | 1 | |
| `impl` blocks | 2078 | |
| `impl<` (generic impl) | ~40+ sampled | Lifetime + type generic impls |

## Ownership, Borrowing, Lifetimes

| Pattern | Count | Notes |
|---------|-------|-------|
| `<'a` (lifetime annotations) | 382 | Common in parsers, zero-copy, FFI |
| `Arc<` | 610 | Shared ownership everywhere |
| `Box<` | 253 | Heap allocation, trait objects |
| `Rc<` | 0 | Not used; codebase prefers `Arc` |
| `Cell<` | 19 | Interior mutability (limited) |
| `RefCell<` | 8 | Interior mutability (limited) |

## Concurrency & Synchronization

| Pattern | Count | Notes |
|---------|-------|-------|
| `Mutex<` | 103 | Mostly `std::sync::Mutex` in FFI/tests |
| `RwLock<` | 308 | Heavy use; also `parking_lot::RwLock` |
| `atomic` (keyword) | 503 | Atomics heavily used in lock-free paths |
| `crossbeam` | 31 | `crossbeam-skiplist`, `crossbeam-channel` |
| `mpsc` | 27 | Standard library channels |
| `thread_local!` | 15 | |
| `OnceLock` | 29 | Lazy initialization |
| `LazyLock` | 0 | Not present (uses `OnceLock` instead) |

## Unsafe

| Pattern | Count | Notes |
|---------|-------|-------|
| `unsafe` (keyword) | 1070 | Very high; SIMD intrinsics, FFI, zero-copy |
| `#![allow(unsafe_op_in_unsafe_fn)]` | 1 | Present in `sochdb-index/src/lib.rs` |

## Generics & Traits

| Pattern | Count | Notes |
|---------|-------|-------|
| `where` clauses | 178 | Trait bounds common |
| `dyn` | 298 | Trait objects (storage backends, callbacks) |
| `impl Trait` (return position) | 0 | Not clearly present |
| `PhantomData` | 69 | Type-state, marker types |
| `impl Drop` | 44 | RAII guards |

## Async

| Pattern | Count | Notes |
|---------|-------|-------|
| `async` | 209 | Limited to gRPC/MCP/server crates |
| `await` | 220 | |
| `tokio` imports | 11 | Explicitly optional per workspace design |
| `tonic` imports | 18 | gRPC server only |

## Collections

| Pattern | Count | Notes |
|---------|-------|-------|
| `Vec<` | 3975 | Ubiquitous |
| `HashMap<` | 554 | |
| `BTreeMap<` | 36 | |

## Closures & Functional Patterns

| Pattern | Count | Notes |
|---------|-------|-------|
| `move \|` (move closures) | 135 | |
| `Fn(` bounds | 104 | Callbacks, routing, policy hooks |
| `FnOnce(` bounds | 45 | |
| `FnMut(` bounds | 10 | |
| `into_iter()` | 272 | |

## Iterators

| Pattern | Count | Notes |
|---------|-------|-------|
| `Iterator for` | 44 | Custom iterators (scan, HNSW, etc.) |
| `IntoIterator for` | 0 | Not clearly present |
| `type Item =` | 38 | Associated type in custom iterators |

## Derive Macros

| Pattern | Count | Notes |
|---------|-------|-------|
| `#[derive(...)]` total | 1655 | |
| `derive(...Debug...)` | 1553 | Almost every type |
| `derive(...Clone...)` | 1399 | |
| `derive(...Default...)` | 194 | |
| `derive(...Serialize...)` | 233 | Serde |
| `derive(...Error...)` | 23 | `thiserror` |

## Declared Macros

| Pattern | Count | Notes |
|---------|-------|-------|
| `macro_rules!` | 5 | SIMD profiling, filter IR |

## Module System & Visibility

| Pattern | Count | Notes |
|---------|-------|-------|
| `mod ` | 774 | Dense module hierarchy |
| `pub(crate)` | 63 | |
| `pub(super)` | 1 | |
| `cfg(` attributes | 866 | Platform/feature gating |
| `repr(` attributes | 87 | C repr for FFI, packed for storage |
| `#[inline` | 949 | Performance-critical paths |

## Type System Patterns

| Pattern | Count | Notes |
|---------|-------|-------|
| `pub type` aliases | ~20+ | Result<T>, TxnId, etc. |
| `const fn` | 51 | |
| `From<` impls | 61 | |
| `Into<` impls | 203 | |

## FFI

| Pattern | Count | Notes |
|---------|-------|-------|
| `extern "C"` | 105 | Heavy FFI in `sochdb-storage/src/ffi.rs` |
| `#[unsafe(no_mangle)]` | ~30+ | C API exports |
| `#[repr(C)]` | ~10+ | C-compatible structs |

## External Crates (by import frequency)

| Crate | Count | Usage |
|-------|-------|-------|
| `std::` | 1267 | |
| `serde::` | 64 | Serialization |
| `thiserror::` | 13 | Error enums |
| `parking_lot::` | 89 | Faster sync primitives |
| `tracing::` | 5 | Logging (lightweight) |
| `bytes::` | 2 | gRPC/proto |
| `tonic::` | 18 | gRPC server |
| `arrow::` | 9 | Columnar/Arrow IPC |
| `pyo3::` | 3 | Python bindings (`sochdb-python`) |
| `criterion::` | 19 | Benchmarks |

## Not Clearly Present

- `impl Trait` in return position (RPIT) — not found in search
- `const generics` (`const N: usize`) — 0 occurrences
- `GATs` (generic associated types) — not clearly present
- `async fn` in traits (native) — codebase uses `async-trait` crate optionally
- `pin!` macro / `Pin<` — only 2 occurrences of `Pin<`
- `no_std` — not used
- `Rc<` — not used (Arc only)
- `LazyLock` — not used (OnceLock instead)
