# The Searchable Cipher v2: Multi-Needle + Context Lines

**Date:** 2026-03-20
**Status:** Approved
**Project:** eachacha
**Depends on:** v1 searchable cipher (chacha20_search.ea)

## Summary

Extends the searchable cipher with multi-needle search (multiple patterns in one decryption pass) and context-line extraction (kernel finds \n boundaries and copies matched lines to output). Benchmarked against NASA HTTP access logs (real-world public dataset).

**Pitch upgrade:** "Search 100 GB of encrypted logs for ERROR, FATAL, and PANIC in a single pass — and get the full log lines back — without ever decrypting to disk."

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Multi-needle approach | All needles in one kernel call, one decrypt pass | Preserves fusion argument — decrypting N times defeats the purpose |
| Needle input format | Packed array + offset/length tables | Ea can't do structs; parallel arrays match v1 pattern |
| SIMD filtering | OR:ed bitmasks from `.==` + `movemask` per unique first-byte | Linear in unique first-bytes, not total needles |
| Decryption window | 4 KB configurable (`window_size` param, default 4096) | 256-byte pt_buf too small for line extraction (~80-200 byte log lines). 4 KB covers ~20-50 lines. |
| Context extraction | Kernel finds \n boundaries within decrypt window, copies line to packed output buffer | Plaintext in decrypt window — \n search is near-free with `.==` + `movemask` |
| Truncated lines | No flags parameter — Python infers from line content | Lines at window boundary shown with `...` prefix/suffix by Python |
| Line length limit | Configurable `max_line_len` parameter, default 1 KB | Covers >99% of log lines; limits both copy AND scan distance |
| Duplicate lines | Emitted per match (no dedup in kernel) | Simpler; Python-side dedup trivial via match_offsets |
| Output format | Packed buffer + parallel arrays (offset, len, match_offset, needle_id) + lines_written count | Memory-efficient for variable-length lines |
| Benchmark dataset | NASA HTTP access logs (July 1995, ~200 MB) | Well-known, free, realistic log search patterns |
| v1 kernel | Unchanged — v2 is a new file | No regression risk |
| Max needles | 64 | Enforced by early-return in kernel |
| lines_buf zeroing | Caller's responsibility (it's an output buffer they intentionally read) | Kernel zeroes pt_buf (internal); lines_buf is external output |

## Kernel Architecture

### New file: `chacha20_search_v2.ea`

### Decryption Window Architecture

v1 used a 256-byte `pt_buf` — too small for line extraction. v2 decrypts into a larger window (default 4 KB) before searching. The 3-tier decrypt logic fills this window:

```
while bytes_remaining > 0:
    // Phase 1: Fill decrypt window (4 KB or remaining bytes, whichever is smaller)
    fill_offset = 0
    while fill_offset + 256 <= window_fill_target:
        // 4-block ILP decrypt → store to pt_buf[fill_offset..fill_offset+256]
        fill_offset += 256
    while fill_offset + 64 <= window_fill_target:
        // single-block decrypt → store to pt_buf[fill_offset..fill_offset+64]
        fill_offset += 64
    if fill_offset < window_fill_target:
        // sub-block tail → store to pt_buf[fill_offset..fill_offset+remaining]

    // Phase 2: Search overlap region (from previous window)
    // Phase 3: Multi-needle SIMD search of pt_buf[0..fill_offset]
    // Phase 4: Extract context lines for matches found
    // Phase 5: Save overlap, zero pt_buf
```

This means `pt_buf` must be at least `window_size` bytes (caller-allocated, default 4096). The `pt_i32` alias covers the same buffer. Tier 1/2/3 ILP decrypt logic is identical to v1 — the only change is that multiple decrypt iterations fill a larger buffer before searching.

**Security:** The window is zeroed after each search phase. Max plaintext in memory: `window_size` bytes (4 KB default) vs v1's 256 bytes. Still a 25-million-fold reduction vs full-file decryption for a 100 GB file.

### Signature (26 parameters)

```
export func chacha20_search_v2(
    // ChaCha20 params
    key: *restrict i32, nonce: *restrict i32, ctr_init: i32,
    ct_u8: *restrict u8, len: i32,
    ks_i32: *restrict mut i32, ks_u8: *restrict mut u8,
    ct_i32: *restrict i32,
    pt_buf: *mut u8, pt_i32: *mut i32,
    overlap: *restrict mut u8,

    // Multi-needle input
    needles: *restrict u8,          // concatenated: "ERROR\0FATAL\0PANIC"
    needle_offsets: *restrict i32,   // [0, 6, 12] start of each needle
    needle_lens: *restrict i32,      // [5, 5, 5] length of each needle
    needle_count: i32,               // 3

    // Output: matched lines
    lines_buf: *restrict mut u8,     // packed line bytes
    lines_buf_cap: i32,              // capacity of lines_buf
    line_offsets: *restrict mut i32,  // start of each line in lines_buf
    line_lens: *restrict mut i32,    // length of each line
    match_offsets: *restrict mut i32, // global byte offset of match
    needle_ids: *restrict mut i32,   // which needle matched (0-indexed)
    max_matches: i32,
    max_line_len: i32,               // max bytes per line (default 1024), limits scan + copy
    window_size: i32,                // decrypt window size (default 4096, min 256)
    match_count: *restrict mut i32,  // output: total matches found
    lines_written: *restrict mut i32 // output: actual lines written to lines_buf (may be < match_count)
)
```

Note: `pt_buf`/`pt_i32` must NOT use `*restrict` (they alias — v1 lesson).

**Early exits:** `needle_count <= 0`, `needle_count > 64`, `len <= 0` → set `match_count[0] = 0`, `lines_written[0] = 0`, return.

### Multi-Needle SIMD Filtering

**Init phase (before main loop):**

Extract unique first-bytes from all needles into a fixed-size array. Ea data layout:

```ea
// Fixed arrays (stack-allocated, max 64 needles)
let mut unique_fb: *mut u8 = ...        // unique first-bytes, up to 64
let mut unique_count: i32 = 0
let mut fb_to_needles: *mut i32 = ...   // for each unique_fb[i], packed list of needle indices

// Dedup loop: O(n^2) for n <= 64, negligible
for ni in 0..needle_count:
    let fb = needles[needle_offsets[ni]]
    // check if fb already in unique_fb[0..unique_count]
    // if not: add it, map it to needle ni
    // if yes: append needle ni to existing mapping
```

Since Ea has no dynamic arrays, the mapping uses a flat 2D layout:
- `fb_needle_map[i * 64 + j]` = j-th needle index for unique_fb[i]
- `fb_needle_count[i]` = number of needles for unique_fb[i]

Max memory: 64 * 64 * 4 = 16 KB for the map. Allocated by caller or as scratch buffer parameter.

**Per u8x16 chunk:**

```ea
let mut bits: i32 = 0
let mut fi: i32 = 0
while fi < unique_count {
    bits = bits | movemask(chunk .== splat(unique_fb[fi]))
    fi = fi + 1
}
if bits == 0 {
    skip chunk  // no first-byte from any needle
} else {
    // scalar scan positions where bits are set
    // at each candidate: test all needles whose first byte matches buf[pos]
    // on match: record match, extract context line
}
```

**Verify at candidate position:** When `buf[pos]` matches a first-byte, iterate only the needles that share that first-byte (via `fb_needle_map`), not all needles. Full byte-by-byte verify for each candidate needle.

### Context Line Extraction

When a match is found at position `p` in `pt_buf` (which is now the large decrypt window):

**Find line_start (backward \n search):**

```ea
newline_splat = splat(10)  // '\n'
// Scan backward from p, max max_line_len bytes
// movemask(load(pt_buf, scan_pos) .== newline_splat) → find highest set bit
// Stop at: \n found, pt_buf start reached, or max_line_len distance
// line_start = position after \n (or 0 if no \n found = truncated start)
```

**Find line_end (forward \n search):**

```ea
// Scan forward from p + needle_len, max max_line_len bytes
// movemask(load(pt_buf, scan_pos) .== newline_splat) → find lowest set bit
// Stop at: \n found, pt_buf end reached, or max_line_len distance
// line_end = position of \n (or buf_len if no \n found = truncated end)
```

**max_line_len limits both scan distance AND copy length.** No unbounded scans.

**Copy line to output:**

```ea
line_len = min(line_end - line_start, max_line_len)
if write_pos + line_len <= lines_buf_cap {
    copy pt_buf[line_start..line_start+line_len] → lines_buf[write_pos]
    line_offsets[lines_written] = write_pos
    line_lens[lines_written] = line_len
    needle_ids[lines_written] = matched_needle_index
    write_pos += line_len
    lines_written += 1
}
match_offsets[match_count] = global_offset + p
match_count += 1
```

**lines_written vs match_count:** `match_count` always increments (total matches). `lines_written` only increments when the line fits in `lines_buf`. Python can iterate `line_offsets[0..lines_written]` safely.

**Duplicate lines:** If two needles match on the same line, the line is emitted twice. Python deduplicates if needed.

### Overlap Handling

**Overlap size:** `max(needle_lens[0..needle_count]) - 1`. Kernel computes this at init by scanning `needle_lens`. The overlap buffer must be allocated by Python to at least 63 bytes (max needle is 64 bytes → overlap 63). Python should allocate 64 bytes to be safe (same as v1).

**Multi-needle overlap search:** The overlap region is searched against ALL needles, not just one. The `search_overlap` function from v1 is adapted to `search_overlap_multi` which loops over all needles for each candidate position in the overlap buffer. Context-line extraction is NOT performed for overlap matches (the overlap buffer is too small). These matches get `match_offsets` but empty lines (`line_lens = 0`). Python can do a targeted decrypt for these rare cases.

### Overlap matches and line extraction

Overlap matches (needle straddling two windows) are rare and their context line spans two windows. For these matches:
- `match_offsets[i]` is set correctly
- `line_lens[i]` is set to 0 (no line extracted)
- `needle_ids[i]` is set correctly
- Python can detect `line_lens[i] == 0` and do a targeted decrypt if needed

## Python Bindings & CLI

### Auto-generated: `chacha20_search_v2.py`

Built by `ea bind chacha20_search_v2.ea --python`.

### Updated CLI: `eachacha_grep.py`

Accept multiple needle arguments:

```bash
python3 eachacha_grep.py "404" "500" "302" encrypted_nasa.bin \
    --key <hex> --nonce <hex>
```

Output:
```
Found 15832 matches in 204,928,317 bytes (3 patterns)
  [404]  offset 1823:  10.0.0.1 - - [01/Jul/1995:00:00:12] "GET /missing.html HTTP/1.0" 404 -
  [500]  offset 9471:  10.0.0.5 - - [01/Jul/1995:00:01:43] "POST /cgi-bin/bad HTTP/1.0" 500 -
  ...
```

When single needle is given, falls back to v1 kernel (simpler, slightly faster).

### NASA log download

Script `data/download_nasa.sh`:

```bash
#!/bin/bash
mkdir -p data
curl -L -o data/nasa_jul95.log.gz \
    "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz" \
    || curl -L -o data/nasa_jul95.log.gz \
    "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"
gunzip -k data/nasa_jul95.log.gz
echo "Downloaded $(wc -c < data/nasa_jul95.log) bytes"
```

## Test Suite

### New file: `test_search_v2.py`

**Regression tests (v1 equivalents adapted to v2 signature):**
1-9. Same correctness tests as v1 but calling v2 with single needle — verify identical match offsets

**Multi-needle tests:**
10. Two needles, separate locations — both found with correct needle_id
11. Three needles, overlapping first-bytes ("ERROR", "EXIT") — verify dedup works, both found
12. Five needles, dense matches — verify all found
13. Needle not present in multi-set — zero matches for that needle_id, others found
14. Two needles match at same position ("AB" and "ABC" at same offset) — both reported, scan does not skip past

**Context line tests:**
15. Match mid-line — verify \n boundaries correct, full line extracted
16. Match at line start — verify line_start is after previous \n
17. Match at line end (just before \n) — verify line_end is at \n
18. Multiple matches on same line — line emitted per match (duplicates expected)
19. Line longer than max_line_len — truncated to max_line_len
20. Match near window boundary — truncated line (no crash), line starts/ends at window edge

**Cross-verification:**
21. Random data with injected needles, decrypt + Python search as reference — verify all match offsets
22. NASA log subset (1 KB) — encrypt, multi-needle search, verify against plaintext

**Edge cases:**
23. lines_buf overflow — lines_written < match_count, no crash, match_count correct
24. Zero needles (needle_count=0) — return 0 matches
25. needle_count=1 — match offsets identical to v1
26. needle_count=65 — early return, 0 matches
27. Overlap match — line_lens[i] == 0, match_offsets correct

## Benchmark Suite

### New file: `bench_search_v2.py`

**Synthetic benchmarks (64 MB):**

| # | Implementation | Description |
|---|---|---|
| 1 | Ea v2 multi-needle (3 needles) | Single pass, all needles + context lines |
| 2 | Ea v1 single-needle x3 | v1 kernel called 3 times |
| 3 | Ea decrypt → grep x3 | Decrypt to buffer, grep per needle |
| 4 | C memmem x3 on plaintext | Baseline: memmem per needle on plaintext |

**NASA log benchmarks (~200 MB):**

| # | Implementation | Needles |
|---|---|---|
| 5 | Ea v2 multi-needle | ["404", "500", "302"] |
| 6 | Ea v1 x3 | Same, 3 separate calls |
| 7 | grep on plaintext NASA log | `grep -c "404\|500\|302"` |

**Target headlines:**
- v2 multi-needle vs v1 x3 → fusion speedup (should be ~2-3x for 3 needles)
- v2 on encrypted NASA log vs grep on plaintext NASA log → the "wow" number
- Context line overhead vs match-only (v1 single needle, no lines) → cost of line extraction

## New Files

| File | Purpose |
|------|---------|
| `chacha20_search_v2.ea` | Multi-needle decrypt+search+context kernel |
| `chacha20_search_v2.py` | Auto-generated Python bindings |
| `test_search_v2.py` | v2 test suite (27 tests) |
| `bench_search_v2.py` | v2 benchmark suite (NASA + synthetic) |
| `data/download_nasa.sh` | NASA log download script |

## Modified Files

| File | Change |
|------|--------|
| `build.sh` | Add chacha20_search_v2.ea build lines |
| `eachacha_grep.py` | Multi-needle support, v2 kernel for multi, v1 for single |

## Out of Scope (v3)

- Multi-core search (ThreadPoolExecutor, split by counter offset)
- Case-insensitive search (OR 0x20 on alpha bytes in kernel)
- Regex patterns
- Binary file search (non-text, no \n boundaries)
