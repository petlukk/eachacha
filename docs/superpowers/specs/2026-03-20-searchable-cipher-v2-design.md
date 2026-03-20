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
| Context extraction | Kernel finds \n boundaries, copies line to packed output buffer | Plaintext already in pt_buf — \n search is near-free with same `.==` + `movemask` |
| Truncated lines | No flags parameter — Python infers from line_offsets vs match context | Simpler; lines at pt_buf boundary shown with `...` prefix/suffix |
| Line length limit | Configurable `max_line_len` parameter, default 1 KB | Covers >99% of log lines; JSON logs can increase to 4 KB |
| Output format | Packed buffer + parallel arrays (offset, len, match_offset, needle_id) | Memory-efficient for variable-length lines |
| Benchmark dataset | NASA HTTP access logs (July 1995, ~200 MB) | Well-known, free, realistic log search patterns |
| v1 kernel | Unchanged — v2 is a new file | No regression risk |

## Kernel Architecture

### New file: `chacha20_search_v2.ea`

### Signature (22 parameters)

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
    max_line_len: i32,               // max bytes per line (default 1024)
    match_count: *restrict mut i32
)
```

Note: `pt_buf`/`pt_i32` must NOT use `*restrict` (they alias — v1 lesson).

### Three-Tier Processing

Same 3-tier structure as v1: 4-block ILP (256B) → single-block (64B) → sub-block tail. The decrypt logic is identical. Only the search + output phase changes.

### Multi-Needle SIMD Filtering

**Init phase (before main loop):**

Extract unique first-bytes from all needles. Max 64 needles supported.

```
unique_first_bytes[]: deduplicated first bytes of all needles
unique_count: number of unique first bytes
// For verify: map first_byte → list of needle indices with that first byte
```

**Per u8x16 chunk:**

```
bits = 0
for each unique first byte fb:
    bits = bits | movemask(chunk .== splat(fb))
if bits == 0:
    skip chunk
else:
    scalar scan positions where bits are set
    for each candidate position:
        test all needles with matching first byte
        on match: extract context line, write to output
```

Cost: one `vpcmpeqb` + `vpmovmskb` + OR per unique first-byte per chunk. For 3 needles with unique first-bytes: 3 comparisons + 2 ORs per chunk.

### Context Line Extraction

When a match is found at position `p` in `pt_buf`:

**Find line_start (backward \n search):**

```
newline_splat = splat(10)  // '\n'
// Scan backward from p in 16-byte chunks
// movemask(chunk .== newline_splat) → find highest set bit
// If no \n found before pt_buf start: line_start = 0 (truncated)
```

**Find line_end (forward \n search):**

```
// Scan forward from p + needle_len in 16-byte chunks
// movemask(chunk .== newline_splat) → find lowest set bit
// If no \n found before pt_buf end: line_end = current_buf_len (truncated)
```

**Copy line to output:**

```
line_len = min(line_end - line_start, max_line_len)
copy pt_buf[line_start..line_start+line_len] → lines_buf[write_pos]
line_offsets[match_idx] = write_pos
line_lens[match_idx] = line_len
match_offsets[match_idx] = global_offset + p
needle_ids[match_idx] = matched_needle_index
write_pos += line_len
```

**Overflow protection:** If `write_pos + line_len > lines_buf_cap`, stop recording lines (but continue counting matches). `match_count` reflects total matches; actual lines in buffer may be fewer.

**Truncated lines:** Lines that start at pt_buf position 0 or end at pt_buf boundary are likely truncated (the real \n is in the previous/next iteration). Python can detect this by checking if the line starts/ends with \n. No kernel-side flags needed.

### Overlap Handling

Same as v1: `needle_len - 1` bytes carried between iterations. For multi-needle, use `max(needle_lens[0..needle_count]) - 1` as the overlap size.

The overlap is for needle matching only, not for line extraction. Lines that cross iteration boundaries will be truncated, which is acceptable — Python shows `...` prefix/suffix.

## Python Bindings & CLI

### Auto-generated: `chacha20_search_v2.py`

Built by `ea bind chacha20_search_v2.ea --python`.

### Updated CLI: `eachacha_grep.py`

Add `--multi` mode or accept multiple needles:

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

## Test Suite

### New file: `test_search_v2.py`

**Regression tests (v1 equivalents adapted to v2 signature):**
1-9. Same correctness tests as v1 but calling v2 with single needle — verify identical results

**Multi-needle tests:**
10. Two needles, separate locations — both found with correct needle_id
11. Three needles, some overlapping first-bytes ("ERROR", "EXIT") — verify dedup works
12. Five needles, dense matches — verify all found
13. Needle not present in multi-set — zero matches for that needle_id
14. All needles match at same position (e.g., "AB" and "ABC" at same offset) — both reported

**Context line tests:**
15. Match mid-line — verify \n boundaries correct, full line extracted
16. Match at line start — verify line_start is after previous \n
17. Match at line end — verify line_end is at \n
18. Multiple matches on same line — line extracted once per match (or deduplicated?)
19. Line longer than max_line_len — truncated to max_line_len
20. Match near pt_buf boundary — truncated line, no crash

**Cross-verification:**
21. Random data with injected needles, decrypt + Python search as reference
22. NASA log subset (1 KB) — encrypt, search, verify against plaintext grep

**Edge cases:**
23. lines_buf overflow — more matches than buffer capacity
24. Zero needles (needle_count=0) — return 0 matches
25. needle_count=1 — behaves identically to v1

## Benchmark Suite

### Updated: `bench_search.py` or new `bench_search_v2.py`

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
- v2 multi-needle vs v1 x3 → fusion speedup (should be ~2-3x)
- v2 on encrypted NASA log vs grep on plaintext NASA log → the "wow" number
- Context line overhead vs match-only (v1) → cost of line extraction

## New Files

| File | Purpose |
|------|---------|
| `chacha20_search_v2.ea` | Multi-needle decrypt+search+context kernel |
| `chacha20_search_v2.py` | Auto-generated Python bindings |
| `test_search_v2.py` | v2 test suite (25 tests) |
| `bench_search_v2.py` | v2 benchmark suite (NASA + synthetic) |
| `data/nasa_jul95.log.gz` | NASA HTTP access log (downloaded, compressed) |

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
