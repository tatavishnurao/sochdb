#!/usr/bin/env bash
set -euo pipefail

# verify-vector-metadata-persistence.sh
# Run all vector-parent-metadata tests and report pass/fail per label.
# Used by CI to gate the VECTOR_METADATA_OK signal.

STATUS=0
declare -A LABELS

# ── Test 1: LEGACY_INSERT_COMPAT / PARENT_ZERO ──────────────────────────
LABEL="LEGACY_INSERT_COMPAT"
if cargo test -p sochdb-grpc --lib -- legacy_insert_without_metadata_returns_absent_metadata 2>&1; then
    echo "${LABEL}=passed"
    LABELS["${LABEL}"]=1
else
    echo "${LABEL}=FAILED"
    STATUS=1
fi

LABEL="PARENT_ZERO"
if cargo test -p sochdb-grpc --lib -- legacy_insert_without_metadata_returns_absent_metadata 2>&1; then
    echo "${LABEL}=passed"
    LABELS["${LABEL}"]=1
else
    echo "${LABEL}=FAILED"
    STATUS=1
fi

# ── Test 2: MISSING_METADATA / SEARCH_METADATA ──────────────────────────
LABEL="MISSING_METADATA"
if cargo test -p sochdb-grpc --lib -- batch_insert_with_mixed_metadata_is_returned_by_search 2>&1; then
    echo "${LABEL}=passed"
    LABELS["${LABEL}"]=1
else
    echo "${LABEL}=FAILED"
    STATUS=1
fi

LABEL="SEARCH_METADATA"
if cargo test -p sochdb-grpc --lib -- batch_insert_with_mixed_metadata_is_returned_by_search 2>&1; then
    echo "${LABEL}=passed"
    LABELS["${LABEL}"]=1
else
    echo "${LABEL}=FAILED"
    STATUS=1
fi

# ── Test 3: BATCH_SEARCH_METADATA ───────────────────────────────────────
LABEL="BATCH_SEARCH_METADATA"
if cargo test -p sochdb-grpc --lib -- search_batch_returns_metadata_for_mixed_presence 2>&1; then
    echo "${LABEL}=passed"
    LABELS["${LABEL}"]=1
else
    echo "${LABEL}=FAILED"
    STATUS=1
fi

# ── Test 4: SNAPSHOT_ROUNDTRIP ──────────────────────────────────────────
LABEL="SNAPSHOT_ROUNDTRIP"
if cargo test -p sochdb-index --lib -- test_save_and_load_preserves_metadata_trailer 2>&1; then
    echo "${LABEL}=passed"
    LABELS["${LABEL}"]=1
else
    echo "${LABEL}=FAILED"
    STATUS=1
fi

# ── Test 5: INDEX_ISOLATION ─────────────────────────────────────────────
LABEL="INDEX_ISOLATION"
if cargo test -p sochdb-index --lib -- test_index_isolation_metadata_not_leaked 2>&1; then
    echo "${LABEL}=passed"
    LABELS["${LABEL}"]=1
else
    echo "${LABEL}=FAILED"
    STATUS=1
fi

# ── Final gate ──────────────────────────────────────────────────────────
if [ "${STATUS}" -eq 0 ]; then
    echo "VECTOR_METADATA_OK=1"
else
    echo "VECTOR_METADATA_OK=0"
fi

exit "${STATUS}"
