#!/usr/bin/env bash
# Validation script for SochDB Rust examples fixes
# Run from workspace root

set -euo pipefail

echo "========================================"
echo "SochDB Examples Validation"
echo "========================================"
echo

# Check workspace member is registered
if ! grep -q '"examples/rust"' Cargo.toml; then
    echo "FAIL: examples/rust not in workspace members"
    exit 1
fi
echo "✓ examples/rust is a workspace member"

# Check examples Cargo.toml exists
if [ ! -f "examples/rust/Cargo.toml" ]; then
    echo "FAIL: examples/rust/Cargo.toml missing"
    exit 1
fi
echo "✓ examples/rust/Cargo.toml exists"

# Check each example compiles
for example in 01_basic_database 02_transactions 03_vector_search 04_sql_queries; do
    echo
    echo "Checking $example..."
    if cargo check -p sochdb-examples --bin "$example" 2>&1 | grep -q "^error"; then
        echo "FAIL: $example does not compile"
        exit 1
    fi
    echo "  ✓ $example compiles"
done

echo
echo "========================================"
echo "All validations passed!"
echo "========================================"
