#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EA="${EA:-ea}"

cd "$SCRIPT_DIR"

echo "Building chacha20.ea..."
$EA chacha20.ea --lib --opt-level=3
$EA bind chacha20.ea --python

echo "Building chacha20_fused.ea..."
$EA chacha20_fused.ea --lib --opt-level=3
$EA bind chacha20_fused.ea --python

if [ -f chacha20_search.ea ]; then
  echo "Building chacha20_search.ea..."
  $EA chacha20_search.ea --lib --opt-level=3
  $EA bind chacha20_search.ea --python
fi

echo "Building chacha20_ref.c..."
cc -O3 -shared -fPIC -o libchacha20_ref.so chacha20_ref.c

echo "Done."
