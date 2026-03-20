#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR"

URL="https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"
OUT="$SCRIPT_DIR/nasa_jul95.log"

if [ -f "$OUT" ]; then
    echo "Already exists: $OUT ($(wc -c < "$OUT") bytes)"
    exit 0
fi

echo "Downloading NASA HTTP access log (July 1995)..."
curl -L -o "$OUT.gz" "$URL" \
    || curl -L -o "$OUT.gz" "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"
gunzip "$OUT.gz"
echo "Downloaded: $OUT ($(wc -c < "$OUT") bytes)"
