#!/bin/bash
# Build portable wheel with generic x86-64 kernels (no AVX2 requirement).
# Use build.sh for local development (compiles with --target=native).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA="${EA:-ea}"
TARGET="${TARGET:-x86-64}"

cd "$SCRIPT_DIR"

echo "Building portable kernels (--target=$TARGET)..."
$EA chacha20.ea --lib --opt-level=3 --target="$TARGET"
$EA chacha20_search.ea --lib --opt-level=3 --target="$TARGET"
$EA chacha20_search_v2.ea --lib --opt-level=3 --target="$TARGET"

echo "Copying to package..."
cp chacha20.so src/eachacha/lib/
cp chacha20_search.so src/eachacha/lib/
cp chacha20_search_v2.so src/eachacha/lib/

echo "Done. Now run: python -m build --wheel"
