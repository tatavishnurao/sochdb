# 01 Concept Map — Rust Concepts Used in SochDB

This file maps each Rust concept to **where it appears**, **how advanced it is**, and **concrete examples** from the actual source.

---

## 1. Basic Types & Data Structures

### Structs
- **Frequency:** 1,789 `pub struct` + 201 `struct`
- **Where:** Every crate, every module.
- **Examples:**
  - `sochdb-core/src/soch.rs:263` — `pub struct SochSchema { ... }`
  - `sochdb-core/src/soch.rs:349` — `pub struct SochRow { pub values: Vec<SochValue> }`
  - `sochdb-core/src/transaction_typestate.rs:160` — `pub struct Transaction<State: TransactionState, Mode: TransactionMode = ReadWrite>`
  - `sochdb-kernel/src/wasm_runtime.rs:236` — `pub struct WasmPluginInstance { ... }`

### Enums
- **Frequency:** 442 `pub enum` + 21 `enum`
- **Where:** Error types, state machines, column types, value types.
- **Examples:**
  - `sochdb-core/src/error.rs:24` — `pub enum SochDBError { Io(#[from] io::Error), ... }`
  - `sochdb-core/src/soch.rs:39` — `pub enum SochValue { Null, Bool(bool), Int(i64), ... }`
  - `sochdb-core/src/concurrency.rs` — `pub enum LockMode { Shared, Exclusive, ... }`
  - `sochdb-kernel/src/wasm_runtime.rs:171` — `pub enum WasmPluginState { Loading, Ready, Executing, ... }`

### Type Aliases
- **Frequency:** ~20+ `pub type`
- **Examples:**
  - `sochdb-core/src/error.rs:105` — `pub type Result<T> = std::result::Result<T, SochDBError>;`
  - `sochdb-core/src/txn.rs` — `pub type TxnId = u64;`
  - `sochdb-client/src/routing.rs` — `pub type ToolHandler = Arc<dyn Fn(&str, &Value) -> ... + Send + Sync>;`

---

## 2. Ownership, Borrowing, Lifetimes

### Lifetimes (`<'a>`)
- **Frequency:** 382 occurrences
- **Complexity:** Intermediate–Advanced
- **Examples:**
  - `sochdb-core/src/tbp.rs` — `pub struct NullBitmap<'a> { data: &'a [u8], ... }`
  - `sochdb-core/src/reclamation.rs` — `pub struct HazardGuard<'a> { domain: &'a HazardDomain, ... }`
  - `sochdb-core/src/soch.rs:637` — `pub struct SochCursor<'a, C: ColumnAccess> { access: &'a C, ... }`
  - `sochdb-vector/src/outlier_encoding.rs` — `impl<'a> Iterator for OutlierIterator<'a>`

### Borrowing Patterns
- `&self` / `&mut self` — ubiquitous.
- `&[u8]` slices — used everywhere for zero-copy keys/values.
- `&str` — common in parsers and formatters.

---

## 3. Generics & Trait Bounds

### Generic Structs
- **Examples:**
  - `sochdb-core/src/transaction_typestate.rs:160` — `Transaction<State, Mode>` (two type params)
  - `sochdb-core/src/epoch_gc.rs` — `EpochGC<K, V>`
  - `sochdb-vector/src/hybrid.rs` — `HybridSearchEngine<V, L>`

### `where` Clauses
- **Frequency:** 178
- **Examples:**
  - `sochdb-core/src/transaction_typestate.rs:556` — `impl<State: TransactionState, Mode: TransactionMode> std::fmt::Debug for Transaction<State, Mode>`
  - Various `impl<T: Clone>`, `impl<K, V>` across the codebase.

### Trait Objects (`dyn Trait`)
- **Frequency:** 298
- **Examples:**
  - `sochdb-core/src/transaction_typestate.rs:182` — `storage: Arc<dyn TransactionStorage>`
  - `sochdb-client/src/routing.rs` — `Arc<dyn Fn(&str, &Value) -> ... + Send + Sync>`
  - `sochdb-storage/src/ffi.rs:587` — `Box<dyn Iterator<Item = Result<(Vec<u8>, Vec<u8>), SochDBError>>>`

---

## 4. Traits

### Custom Traits
- **Frequency:** 80 `pub trait` + 1 `trait`
- **Examples:**
  - `sochdb-core/src/soch.rs:624` — `pub trait ColumnAccess { fn row_count(&self) -> usize; ... }`
  - `sochdb-core/src/transaction_typestate.rs:196` — `pub trait TransactionStorage: Send + Sync { ... }`
  - `sochdb-core/src/version_chain.rs` — `pub trait MvccVersionChain { ... }`
  - `sochdb-grpc/src/pg_wire.rs` — `pub trait PgSqlExecutor: Send + Sync + 'static { ... }`
  - `sochdb-index/src/vector_storage.rs` — `pub trait VectorStorage: Send + Sync { ... }`

### Sealed Traits
- **Where:** `sochdb-core/src/transaction_typestate.rs`
- **Example:**
  ```rust
  mod private { pub trait Sealed {} }
  pub trait TransactionState: private::Sealed {}
  ```

### Trait Implementations (`impl Trait for Type`)
- **Frequency:** 2,078
- **Examples:**
  - `sochdb-core/src/soch.rs:119` — `impl fmt::Display for SochValue`
  - `sochdb-core/src/soch.rs:655` — `impl<'a, C: ColumnAccess> Iterator for SochCursor<'a, C>`
  - `sochdb-core/src/transaction_typestate.rs:313` — `impl<Mode: TransactionMode> Transaction<Active, Mode>`

---

## 5. Smart Pointers

### `Arc<T>`
- **Frequency:** 610
- **Usage:** Shared ownership across threads (storage backends, plugin registries, indexes).
- **Examples:**
  - `sochdb-core/src/reclamation.rs` — `hazard: Arc<HazardDomain>`
  - `sochdb-kernel/src/wasm_runtime.rs` — `plugins: RwLock<HashMap<String, Arc<WasmPluginInstance>>>`

### `Box<T>`
- **Frequency:** 253
- **Usage:** Heap allocation, trait objects, FFI pointer boxing.
- **Examples:**
  - `sochdb-storage/src/ffi.rs` — `let ptr = Box::new(DatabasePtr(db)); Box::into_raw(ptr)`

### `PhantomData<T>`
- **Frequency:** 69
- **Usage:** Type-state markers, variance control.
- **Example:**
  - `sochdb-core/src/transaction_typestate.rs:164-166` — `_state: PhantomData<State>, _mode: PhantomData<Mode>`

---

## 6. Concurrency

### `std::sync::atomic`
- **Frequency:** 503
- **Usage:** Lock-free counters, flags, epoch tracking.
- **Examples:**
  - `sochdb-core/src/reclamation.rs` — `global_epoch: AtomicU64`
  - `sochdb-kernel/src/wasm_runtime.rs` — `fuel_remaining: AtomicU64`

### `RwLock` / `Mutex`
- **Frequency:** 308 RwLock, 103 Mutex
- **Usage:** `parking_lot::RwLock` preferred in hot paths; `std::sync::Mutex` in FFI.
- **Examples:**
  - `sochdb-kernel/src/wasm_runtime.rs:240` — `state: RwLock<WasmPluginState>`
  - `sochdb-storage/src/ffi.rs:43` — `static COLLECTION_INDEXES: OnceLock<Mutex<HashMap<...>>>`

### `crossbeam`
- **Frequency:** 31
- **Usage:** Skip lists, channels, epoch-based reclamation.
- **Workspace deps:** `crossbeam-skiplist`, `crossbeam-channel`

---

## 7. Unsafe Rust

### `unsafe` Blocks & Functions
- **Frequency:** 1,070
- **Complexity:** Advanced
- **Categories:**
  1. **SIMD intrinsics** (`core::arch::x86_64`) — `sochdb-index/src/simd_distance.rs`, `sochdb-vector/src/simd/`
  2. **FFI** — `sochdb-storage/src/ffi.rs` (C API)
  3. **Zero-copy / pointer casting** — `sochdb-core/src/zero_copy.rs`, `sochdb-storage/src/zero_copy_serde.rs`
  4. **Type punning / raw memory** — `sochdb-index/src/simd_batch_distance.rs`

### Examples
- `sochdb-vector/src/simd/mod.rs` — defines `SimdBackend` trait wrapping `unsafe` intrinsics.
- `sochdb-storage/src/ffi.rs:183` — `pub unsafe extern "C" fn sochdb_open_with_config(...)`
- `sochdb-index/src/lib.rs:20` — `#![allow(unsafe_op_in_unsafe_fn)]`

---

## 8. Error Handling

### `thiserror`
- **Frequency:** 13 imports, 23 `derive(Error)`
- **Example:**
  - `sochdb-core/src/error.rs:23` — `#[derive(Error, Debug)] pub enum SochDBError { ... }`

### `Result<T, E>` Aliases
- **Example:** `sochdb-core/src/error.rs:105` — `pub type Result<T> = std::result::Result<T, SochDBError>;`

### `?` Operator
- Ubiquitous. Every function that can fail uses it.

---

## 9. Macros

### Declarative Macros (`macro_rules!`)
- **Frequency:** 5
- **Examples:**
  - `sochdb-index/src/profiling.rs` — `macro_rules! profile_section { ... }`
  - `sochdb-index/src/simd_batch_distance.rs` — `macro_rules! process_candidate { ... }`
  - `sochdb-query/src/filter_ir.rs` — `macro_rules! filter_ir { ... }`

### Procedural Macros (via `derive`)
- **serde:** `Serialize`, `Deserialize` — 233 occurrences
- **thiserror:** `Error` — 23 occurrences
- **clap:** `Parser` (in `sochdb-tools/src/cli.rs` likely)

---

## 10. Async / Await

### `async` / `await`
- **Frequency:** 209 / 220
- **Where:** `sochdb-grpc`, `sochdb-mcp`, `sochdb-client` (optional feature)
- **Crates:** `tokio` (optional), `tonic`, `async-trait` (optional in index)
- **Examples:**
  - `sochdb-grpc/src/lib.rs` — gRPC server uses `tonic` (generated async code)
  - `sochdb-vector/src/async_rotation.rs` — `impl<T> BoundedChannel<T>`
  - `sochdb-vector/src/async_lsm.rs` — Non-blocking LSM sealing

---

## 11. FFI (Foreign Function Interface)

### C API Exports
- **Where:** `sochdb-storage/src/ffi.rs`
- **Size:** ~1,500+ lines of FFI code
- **Concepts used:**
  - `#[unsafe(no_mangle)]`
  - `extern "C"` functions
  - `#[repr(C)]` structs (`C_TxnHandle`, `C_CommitResult`, `C_DatabaseConfig`, etc.)
  - Raw pointers (`*const c_char`, `*mut DatabasePtr`)
  - `CStr::from_ptr`, `slice::from_raw_parts`
  - `Box::into_raw` / `Box::from_raw` for memory management across FFI boundary
  - `std::panic::catch_unwind` for FFI safety

---

## 12. Iterators & Closures

### Custom Iterators
- **Frequency:** 44 `impl Iterator for`
- **Examples:**
  - `sochdb-core/src/soch.rs:655` — `impl<'a, C: ColumnAccess> Iterator for SochCursor<'a, C>`
  - `sochdb-core/src/block_storage.rs` — `impl<'a> Iterator for WalIterator<'a>`
  - `sochdb-vector/src/outlier_encoding.rs` — `impl<'a> Iterator for OutlierIterator<'a>`

### Closures & `Fn` Traits
- **Frequency:** 104 `Fn(`, 45 `FnOnce(`, 10 `FnMut(`, 135 `move |`
- **Examples:**
  - `sochdb-client/src/routing.rs` — `pub type ToolHandler = Arc<dyn Fn(&str, &Value) -> Result<Value, String> + Send + Sync>;`
  - `sochdb-storage/src/ffi.rs:639` — `let iter = Box::new(rows.into_iter().map(Ok));`

---

## 13. Builder Patterns

- **Where:** `sochdb-core/src/soch.rs:291`, `sochdb-core/src/transaction_typestate.rs:231`, `sochdb-tools/src/builder.rs`
- **Example:**
  ```rust
  // sochdb-core/src/soch.rs
  impl SochSchema {
      pub fn field(mut self, name: impl Into<String>, field_type: SochType) -> Self { ... }
      pub fn primary_key(mut self, field: impl Into<String>) -> Self { ... }
  }
  ```

---

## 14. Type-State Pattern

- **Where:** `sochdb-core/src/transaction_typestate.rs`
- **Concept:** Encode state machine in types using `PhantomData`.
- **Key types:** `Active`, `Committed`, `Aborted`, `ReadOnly`, `ReadWrite`, `WriteOnly`
- **Effect:** `commit()` consumes `self`, making use-after-commit a compile error.

---

## 15. Module System & Visibility

### `pub(crate)` / `pub(super)`
- **Frequency:** 63 `pub(crate)`, 1 `pub(super)`
- **Example:** `sochdb-core/src/transaction_typestate.rs` has `mod private { pub trait Sealed {} }`

### `cfg` Attributes
- **Frequency:** 866
- **Usage:** Platform gating (`#[cfg(unix)]`), feature gating (`#[cfg(feature = "analytics")]`), test gating (`#[cfg(test)]`)

---

## 16. serde & Serialization

- **Frequency:** 64 `use serde::` imports, 233 `derive(...Serialize...)`
- **Usage:** Every data type that crosses a boundary (WAL, network, FFI, Python).
- **Example:** `sochdb-core/src/soch.rs:38` — `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]`

---

## 17. `const` Patterns

### `const fn`
- **Frequency:** 51
- **Usage:** Bit manipulation, constants, inline helpers.

### `static`
- **Example:** `sochdb-core/src/lib.rs:83` — `static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;`

---

## 18. `Drop` & RAII

- **Frequency:** 44 `impl Drop`
- **Usage:** Hazard pointer guards, epoch guards, write locks.
- **Examples:**
  - `sochdb-core/src/reclamation.rs` — `impl<'a> Drop for HazardGuard<'a>`
  - `sochdb-core/src/concurrency.rs` — `impl<'a> Drop for WriteGuard<'a>`
