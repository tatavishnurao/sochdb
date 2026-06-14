# 05 Examples to Study — Annotated Source Files

Each file below is chosen because it teaches a specific Rust concept that is heavily used in SochDB. Read them in order.

---

## 1. `sochdb-core/src/soch.rs` — The "Rust 101" File
**Lines:** 768  
**Concepts:** Structs, enums, `impl`, `match`, `Display`, `Option`, `Result`, builder pattern, generics, lifetimes, custom `Iterator`.

### Why Study It
This file defines the TOON data format. It is the most pedagogical file in the repo because it uses basic Rust features to build a complete parser and formatter.

### Key Sections
- **Lines 38-54:** `SochValue` enum — recursive enum with data variants. Study how `Array(Vec<SochValue>)` and `Object(HashMap<String, SochValue>)` allow arbitrary nesting without `Box` indirection (for those variants).
- **Lines 56-99:** `impl SochValue` — accessor methods using `match` and `Option`. Notice `as_uint()` handles `Int` to `UInt` conversion safely.
- **Lines 119-169:** `impl fmt::Display for SochValue` — exhaustive `match` with string escaping logic. This is a masterclass in pattern matching.
- **Lines 172-186:** `SochType` enum — recursive type system with `Box<SochType>` for `Array` and `Optional`. This is how Rust handles recursive types.
- **Lines 261-345:** `SochSchema` and builder methods (`field(mut self, ...)`). Study the consuming builder pattern.
- **Lines 373-412:** `SochRow::parse()` — manual parser using `chars.peekable()`, state machine with `in_quotes` flag.
- **Lines 617-688:** `SochCursor<'a, C: ColumnAccess>` — generic lifetime struct that implements `Iterator`. Study associated type `type Item = String;`.

### After Reading This File, You Should Be Able To
- Write structs, enums, and impl blocks
- Use `match` exhaustively
- Implement `Display` and basic traits
- Understand `Box<T>` for recursion
- Write a simple consuming builder
- Understand generic lifetime parameters in structs

---

## 2. `sochdb-core/src/error.rs` — Error Handling Idiom
**Lines:** 105  
**Concepts:** `thiserror`, `#[derive(Error)]`, `#[from]`, structured error variants, type aliases.

### Why Study It
Every crate in the workspace copies this pattern.

### Key Sections
- **Lines 23-103:** `SochDBError` enum with `#[derive(Error, Debug)]`. Notice how `Io(#[from] io::Error)` auto-implements `From<io::Error>`. Notice named fields in variants (`DataCorruption { details, location, hint }`).
- **Line 105:** `pub type Result<T> = std::result::Result<T, SochDBError>;` — the ubiquitous alias pattern.

### After Reading This File, You Should Be Able To
- Use `thiserror` to define rich error enums
- Use `#[from]` for automatic error conversion
- Create `Result<T>` type aliases

---

## 3. `sochdb-core/src/transaction_typestate.rs` — Type-State Pattern
**Lines:** 732  
**Concepts:** `PhantomData`, sealed traits, generic structs with multiple params, consuming methods, `where` clauses, `impl` specialization by type param, `#[cfg(test)]` mocks.

### Why Study It
This is the most advanced "purely safe Rust" file in `sochdb-core`. It encodes a transaction state machine in the type system.

### Key Sections
- **Lines 77-112:** Sealed trait pattern. `mod private { pub trait Sealed {} }` prevents external implementations of `TransactionState`.
- **Lines 160-167:** `Transaction<State, Mode>` struct with `PhantomData` markers. The markers are zero-sized; they exist only to bind the type parameters to the struct.
- **Lines 313-336:** `impl<Mode: TransactionMode> Transaction<Active, Mode>` — methods available in `Active` state. Notice `abort(self)` takes ownership (consumes `self`).
- **Lines 340-363:** `impl Transaction<Active, ReadOnly>` — read-only specific methods.
- **Lines 365-444:** `impl Transaction<Active, ReadWrite>` — read-write methods including `get(&mut self)` and `commit(self)`.
- **Lines 406-430:** `commit(self)` consumes the transaction, moves data out, and returns `Transaction<Committed, ReadWrite>`. After this call, the original `txn` variable is invalid.
- **Lines 591-732:** Test module with a `MockStorage` implementing `TransactionStorage`. Study how `Arc<RwLock<HashMap<...>>>` is used for thread-safe mock state.

### After Reading This File, You Should Be Able To
- Design type-state APIs with `PhantomData`
- Use sealed traits to control implementors
- Write generic `impl` blocks specialized by type parameter
- Understand why `self` vs `&mut self` affects ownership
- Write mock trait implementations for tests

---

## 4. `sochdb-core/src/tbp.rs` — Zero-Copy Parsing with Lifetimes
**Concepts:** Lifetimes (`<'a>`), `&'a [u8]`, `impl<'a>` blocks, zero-copy views.

### Why Study It
This file shows how SochDB avoids copying data when reading from bytes.

### Key Sections
- `pub struct NullBitmap<'a> { data: &'a [u8], columns: usize }` — a view into existing memory.
- `pub struct TbpReader<'a> { data: &'a [u8], header: TbpHeader }` — parser that borrows input.

### After Reading This File, You Should Be Able To
- Write structs that borrow data with explicit lifetimes
- Understand why zero-copy requires lifetime annotations
- Separate owning types (`Vec<u8>`) from borrowing types (`&'a [u8]`)

---

## 5. `sochdb-core/src/reclamation.rs` — RAII and Atomics
**Concepts:** `Arc`, atomics, `Drop` impl, RAII guards, enums.

### Why Study It
This file implements hazard-pointer-style memory reclamation.

### Key Sections
- `pub struct HazardGuard<'a> { domain: &'a HazardDomain, slot_idx: usize }`
- `impl<'a> Drop for HazardGuard<'a>` — releases the hazard pointer slot when the guard goes out of scope.
- `pub struct EpochDomain { global_epoch: AtomicU64, ... }`

### After Reading This File, You Should Be Able To
- Write `Drop` impls for custom cleanup
- Use `AtomicU64` with `Ordering`
- Understand RAII guard patterns in Rust

---

## 6. `sochdb-storage/src/ffi.rs` — Complete FFI Example
**Lines:** ~1,500+  
**Concepts:** `extern "C"`, `#[unsafe(no_mangle)]`, `#[repr(C)]`, raw pointers, `unsafe` blocks, `CStr`, `slice::from_raw_parts`, `Box::into_raw` / `Box::from_raw`, `std::panic::catch_unwind`, `OnceLock`, `Mutex`, `dyn Iterator` in `Box`, batch binary protocols.

### Why Study It
This is the largest FFI file in the repo. It exposes a full C API for SochDB.

### Key Sections
- **Lines 134-176:** `#[repr(C)]` structs (`C_TxnHandle`, `C_CommitResult`, `C_DatabaseConfig`).
- **Lines 183-236:** `sochdb_open_with_config` — null checks, `CStr::from_ptr`, config mapping, `Box::into_raw` for pointer transfer.
- **Lines 438-483:** `sochdb_get` — reads database, allocates heap buffer, leaks it to C via `Box::into_raw`.
- **Lines 585-646:** `ScanIteratorPtr` — `Box<dyn Iterator<Item = Result<...>>>` inside an opaque struct for C interop.
- **Lines 858-915:** `sochdb_put_many` — binary batch protocol parsing from raw bytes.
- **Lines 1241-1271:** `sochdb_set_table_index_policy` — string-to-enum mapping from C integers.

### After Reading This File, You Should Be Able To
- Write `extern "C"` functions with safety comments
- Transfer ownership across the FFI boundary with `Box`
- Parse binary protocols from raw byte slices
- Use `catch_unwind` to prevent panics from crossing FFI
- Understand `OnceLock<Mutex<T>>` for lazy global initialization

---

## 7. `sochdb-kernel/src/wasm_runtime.rs` — Concurrency and State Machines
**Lines:** 896  
**Concepts:** `Arc`, `RwLock`, `AtomicU64`, `Ordering`, `Default`, enums, `std::any::Any`, `dyn Any`, `HashMap`, `Instant`, `Path`, `Box`, `std::fs::read`, `#[cfg(test)]`.

### Why Study It
This is a clean, safe-Rust file that demonstrates how to manage shared mutable state without `unsafe`.

### Key Sections
- **Lines 75-163:** `WasmPluginCapabilities` — struct with `Default`, builder-like helper methods (`observability_only()`, `read_only(tables)`), and logic methods (`can_read`, `can_write`).
- **Lines 236-251:** `WasmPluginInstance` — fields include `RwLock<WasmPluginState>`, `AtomicU64`, `RwLock<WasmPluginStats>`.
- **Lines 286-335:** `call()` method — reads `RwLock`, writes `RwLock`, updates atomics with `Ordering::Acquire` / `AcqRel`.
- **Lines 468-612:** `WasmPluginRegistry` — `RwLock<HashMap<String, Arc<WasmPluginInstance>>>` for thread-safe plugin registry.
- **Lines 719-896:** Extensive tests demonstrating how to test concurrent structures with `Arc` and mock data.

### After Reading This File, You Should Be Able To
- Use `RwLock` and `AtomicU64` correctly
- Design state machines with enums
- Write thread-safe registries with `Arc<RwLock<HashMap<...>>>`
- Use `Ordering` variants appropriately

---

## 8. `sochdb-vector/src/simd/mod.rs` — Abstracting `unsafe` with Traits
**Lines:** 63  
**Concepts:** Traits with associated types, `unsafe` intrinsic abstraction, `cfg` feature gating.

### Why Study It
This file shows how to design safe abstractions over `unsafe` SIMD code.

### Key Sections
- **Lines 45-63:** `trait SimdBackend` with associated types (`type U8x32`, `type F32x8`). Each backend (AVX2, NEON, scalar) implements this trait.
- **Lines 32-33:** `#[cfg(feature = "portable-simd")]` — optional module compilation.

### After Reading This File, You Should Be Able To
- Define traits with associated types
- Understand how to isolate `unsafe` behind safe traits
- Use `cfg` for platform-specific modules

---

## 9. `sochdb-index/src/profiling.rs` — Declarative Macros
**Concepts:** `macro_rules!`, hygiene, `$name:expr`, `$code:block`.

### Why Study It
One of the few `macro_rules!` in the repo.

### Key Section
```rust
macro_rules! profile_section {
    ($name:expr, $code:block) => {{
        if $crate::profiling::is_profiling_enabled() {
            let timer = $crate::profiling::Timer::start($name);
            let result = $code;
            timer.stop();
            result
        } else {
            $code
        }
    }};
}
```

### After Reading This File, You Should Be Able To
- Write simple `macro_rules!` macros
- Understand macro hygiene and `$crate`

---

## 10. `sochdb-query/src/filter_ir.rs` — DSL Macros
**Concepts:** `macro_rules!` with recursive patterns, empty pattern `()`.

### Why Study It
Shows how Rust macros can create internal DSLs.

### Key Section
```rust
macro_rules! filter_ir {
    () => { $crate::filter_ir::FilterIR::all() };
    // ... additional patterns
}
```

---

## 11. `sochdb-tools/src/builder.rs` — Builder Pattern + Error Enums
**Concepts:** Builder pattern, `thiserror::Error`, enums with named fields.

### Why Study It
Small, self-contained file showing two common patterns together.

---

## 12. `sochdb-grpc/src/lib.rs` — Async / Protobuf Integration
**Lines:** 88  
**Concepts:** `tonic::include_proto!`, `pub mod` hierarchy, async service organization.

### Why Study It
Entry point for understanding how the gRPC server is structured.

### Key Section
- **Line 54:** `tonic::include_proto!("sochdb.v1");` — generated protobuf code inclusion.

---

## Study Schedule Recommendation

| Day | File | Time | Goal |
|-----|------|------|------|
| 1-2 | `sochdb-core/src/soch.rs` | 2h | Master structs, enums, match, Display, builder |
| 3 | `sochdb-core/src/error.rs` | 30m | Master `thiserror` patterns |
| 4-5 | `sochdb-core/src/transaction_typestate.rs` | 3h | Master type-state, PhantomData, sealed traits |
| 6 | `sochdb-core/src/tbp.rs` | 1h | Master lifetimes and zero-copy |
| 7 | `sochdb-core/src/reclamation.rs` | 1h | Master Drop and atomics |
| 8-9 | `sochdb-storage/src/ffi.rs` | 3h | Master FFI patterns |
| 10 | `sochdb-kernel/src/wasm_runtime.rs` | 2h | Master concurrency primitives |
| 11 | `sochdb-vector/src/simd/mod.rs` | 1h | Master trait abstractions over unsafe |
| 12 | `sochdb-index/src/profiling.rs` + `sochdb-query/src/filter_ir.rs` | 1h | Master declarative macros |

Total focused study: ~17 hours to reach intermediate comfort with this codebase.
