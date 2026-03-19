#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA="${EA:-/root/dev/eacompute/target/release/ea}"

cd "$SCRIPT_DIR"

echo "Building chacha20.ea..."
$EA chacha20.ea --lib
$EA bind chacha20.ea --python

echo "Building chacha20_ref.c..."
cc -O3 -shared -fPIC -o libchacha20_ref.so chacha20_ref.c

echo "Done."
