#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA="${EA:-/root/dev/eacompute/target/release/ea}"

cd "$SCRIPT_DIR"

echo "Building chacha20.ea..."
$EA chacha20.ea --lib
$EA bind chacha20.ea --python

echo "Done."
