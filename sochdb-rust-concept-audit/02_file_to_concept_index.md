# 02 File-to-Concept Index

Reverse lookup: for important files, which Rust concepts appear inside them.

---

## `sochdb-core/src/soch.rs`
**Concepts:** Structs, Enums, `impl` blocks, `Display` trait, `Serialize`/`Deserialize`, generic lifetime struct (`SochCursor<'a, C>`), custom `Iterator` impl, `Box<SochType>` recursion, builder pattern (`SochSchema::field(mut self, ...)`), `match` exhaustiveness, `Option<T>`, `Result<T, E>`, string parsing with `peekable()` iterator, `format!` macro.

---

## `sochdb-core/src/error.rs`
**Concepts:** Enums with payload, `thiserror::Error` derive, `#[from]` conversion, `#[error("...")]` formatting, type alias (`pub type Result<T>`), structured error variants with named fields.

---

## `sochdb-core/src/transaction_typestate.rs`
**Concepts:** Type-state pattern, `PhantomData`, sealed traits, generic structs with multiple type params (`Transaction<State, Mode>`), `where` clauses, trait bounds (`Send + Sync`), builder pattern (`TransactionBuilder`), consuming methods (`self` instead of `&mut self`), `std::marker::PhantomData`, `impl` specialization by mode (`impl Transaction<Active, ReadOnly>`, `impl Transaction<Active, ReadWrite>`), `#[cfg(test)]` module, mock trait implementations.

---

## `sochdb-core/src/reclamation.rs`
**Concepts:** `Arc<T>`, atomics (`AtomicU64`), lifetimes (`HazardGuard<'a>`), `Drop` impl for RAII, enums (`ReclaimStrategy`), structs with generic bounds, `std::sync::atomic::Ordering`.

---

## `sochdb-core/src/lib.rs`
**Concepts:** Workspace crate root, `pub mod` declarations, feature gating (`#[cfg(feature = "jemalloc")]`), `#[global_allocator]`, `static` with generic type, module re-exports (`pub use`), doc comments with examples, `const` definitions.

---

## `sochdb-storage/src/lib.rs`
**Concepts:** Massive crate root (~371 lines), `pub mod` for ~80 modules, feature gating (`#[cfg(unix)]` for `ipc_server`, `io_uring`), re-export heavy (`pub use ...` for every submodule), module documentation.

---

## `sochdb-storage/src/ffi.rs`
**Concepts:** FFI (`extern "C"`), `#[unsafe(no_mangle)]`, `#[repr(C)]`, raw pointers (`*const c_char`, `*mut u8`), `unsafe` blocks, `CStr::from_ptr`, `slice::from_raw_parts`, `Box::into_raw` / `Box::from_raw`, `std::panic::catch_unwind`, `OnceLock<Mutex<HashMap<...>>>` for lazy global registry, `static` globals, pointer null checks, memory leaking across boundary (`let _ = Box::into_raw(buf)`), batch binary protocol parsing, `CString::new` / `CString::into_raw`, iterator boxing (`Box<dyn Iterator<...>>`), `dyn` trait objects.

---

## `sochdb-index/src/lib.rs`
**Concepts:** Crate root with `#![allow(unsafe_op_in_unsafe_fn)]`, `pub mod` for ~50 modules, re-exports, `pub use sochdb_core::learned_index` for cross-crate dependency, feature gating via `#[cfg(test)]`.

---

## `sochdb-index/src/simd_distance.rs` / `simd_batch_distance.rs`
**Concepts:** `unsafe` blocks, `core::arch::x86_64` intrinsics (`_mm256_loadu_ps`, `_mm256_sub_ps`, `_mm256_fmadd_ps`), `macro_rules!`, `#[inline]` attributes, `#[cfg(target_arch = "x86_64")]` gating, raw pointer arithmetic, `std::mem::transmute`, `const fn` for lane size.

---

## `sochdb-vector/src/lib.rs`
**Concepts:** Crate root, module hierarchy, re-exports, SIMD dispatch re-exports, feature module comments.

---

## `sochdb-vector/src/simd/mod.rs`
**Concepts:** Trait with associated types (`SimdBackend`), `unsafe` intrinsic abstraction, `cfg(feature = "portable-simd")`, module re-exports, generic backend design.

---

## `sochdb-grpc/src/lib.rs`
**Concepts:** `tonic::include_proto!`, `pub mod` hierarchy, async service modules, re-exports.

---

## `sochdb-kernel/src/wasm_runtime.rs`
**Concepts:** `Arc<T>`, `RwLock<T>`, `AtomicU64`, `std::any::Any`, `dyn Any`, `Default` impl, builder-like methods, enum state machine, `std::time::Instant`, `Ordering::Acquire` / `Release` / `Relaxed` / `AcqRel`, `std::fs::read`, `Path`, `Box::new`, `#[cfg(test)]` tests with mock objects.

---

## `sochdb-client/src/routing.rs`
**Concepts:** `Arc<dyn Fn(...) + Send + Sync>`, trait objects with closure bounds, `HashMap`, `std::sync` primitives.

---

## `sochdb-tools/src/builder.rs`
**Concepts:** Builder pattern, `thiserror::Error`, enums with named fields, `Result<T>`.

---

## `sochdb-query/src/filter_ir.rs`
**Concepts:** `macro_rules!` DSL (`filter_ir!{}`), recursive macro patterns.

---

## `sochdb-core/src/tbp.rs`
**Concepts:** Lifetime structs (`NullBitmap<'a>`, `TbpReader<'a>`, `TbpRowWriter<'a>`), `impl<'a>` blocks, zero-copy parsing, `&'a [u8]` slices.

---

## `sochdb-core/src/columnar.rs`
**Concepts:** Enums with struct variants (`TypedColumn::Int64 { values: Vec<i64>, validity: ValidityBitmap }`), recursive enums, `Vec<T>`, `&str`.

---

## `sochdb-core/src/concurrency.rs`
**Concepts:** Type aliases (`TableId = u64`, `RowId = u128`, `TxnId = u64`), enums (`IntentLock`, `LockMode`, `LockResult`), `impl<'a> WriteGuard<'a>`, `Drop` impl for lock release.

---

## `Cargo.toml` (workspace root)
**Concepts:** Workspace configuration, `[workspace.dependencies]`, feature flags table, profile optimization (`lto = "fat"`, `codegen-units = 1`, `panic = "abort"`), `[profile.dev]` / `[profile.test]` overrides, dependency management strategy comments.
