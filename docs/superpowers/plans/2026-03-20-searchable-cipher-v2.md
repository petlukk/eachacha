# Searchable Cipher v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-needle fused decrypt+search kernel with context-line extraction, benchmarked against NASA HTTP access logs.

**Architecture:** The v2 kernel decrypts ciphertext into a configurable window (default 4 KB), searches for multiple needles using OR:ed `.==` + `movemask` bitmasks per unique first-byte, and extracts matched log lines by finding \n boundaries with the same SIMD primitives. The 3-tier decrypt (4-block ILP / single-block / tail) fills the window, then search + line extraction operates on the full window before zeroing.

**Tech Stack:** Ea SIMD language (ea-compiler 1.10.0), Python 3 (ctypes, numpy), NASA HTTP access logs

**Spec:** `docs/superpowers/specs/2026-03-20-searchable-cipher-v2-design.md`

**Ea compiler:** `EA=/usr/local/lib/python3.13/dist-packages/ea/bin/ea`

**Key Ea patterns from v1:**
- `pt_buf`/`pt_i32` must NOT use `*restrict` (they alias)
- u8 equality: `!(a < b) && !(b < a)` (no scalar `==` on u8)
- Vector compare: `movemask(chunk .== splat(byte))` → i32 bitmask
- `store(pt_i32, fixed_offset, load(ct_i32, elem_off) .^ keystream)` for decrypt

---

## File Structure

| File | Responsibility |
|------|---------------|
| `chacha20_search_v2.ea` | Multi-needle decrypt+search+context kernel (new) |
| `chacha20_search_v2.py` | Auto-generated Python bindings (ea bind) |
| `test_search_v2.py` | v2 test suite — 27 tests (new) |
| `bench_search_v2.py` | v2 benchmark suite — NASA + synthetic (new) |
| `data/download_nasa.sh` | NASA log download script (new) |
| `build.sh` | Add v2 build lines (modify) |
| `eachacha_grep.py` | Multi-needle support (modify) |

---

## Task 1: Build setup and NASA download script

**Files:**
- Modify: `build.sh`
- Create: `data/download_nasa.sh`

- [ ] **Step 1: Add v2 build lines to build.sh**

Add after the existing chacha20_search.ea block:

```bash
if [ -f chacha20_search_v2.ea ]; then
  echo "Building chacha20_search_v2.ea..."
  $EA chacha20_search_v2.ea --lib --opt-level=3
  $EA bind chacha20_search_v2.ea --python
fi
```

- [ ] **Step 2: Create data/download_nasa.sh**

```bash
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
curl -L -o "$OUT.gz" "$URL"
gunzip "$OUT.gz"
echo "Downloaded: $OUT ($(wc -c < "$OUT") bytes)"
```

Run: `chmod +x data/download_nasa.sh`

- [ ] **Step 3: Verify existing build and tests still pass**

```bash
cd /root/dev/eachacha
EA=/usr/local/lib/python3.13/dist-packages/ea/bin/ea ./build.sh
python3 test_vectors.py && python3 test_fused.py && python3 test_search.py
```

- [ ] **Step 4: Commit**

```bash
git add build.sh data/download_nasa.sh
git commit -m "build: add v2 kernel build lines and NASA download script"
```

---

## Task 2: Write the v2 kernel — decrypt window + multi-needle search helpers

The kernel is large (~700-800 lines). This task writes the helper functions and the decrypt-window-fill logic. Task 3 adds context-line extraction. Task 4 wires everything into the main export.

**Files:**
- Create: `chacha20_search_v2.ea`

- [ ] **Step 1: Write rotation helpers and chacha20_block**

Copy verbatim from `chacha20_search.ea` lines 1-65 (rotation helpers + chacha20_block). These are identical across all kernels.

- [ ] **Step 2: Write search_buf_multi — multi-needle scalar search**

```ea
// Search buf for multiple needles. For each position, test all needles whose
// first byte matches. Returns updated match_count.
// needle_ids_out[mc] = index of matching needle.
func search_buf_multi(
    buf: *mut u8, buf_len: i32,
    needles: *restrict u8, needle_offsets: *restrict i32,
    needle_lens: *restrict i32, needle_count: i32,
    matches: *restrict mut i32, needle_ids_out: *restrict mut i32,
    match_count: i32, max_matches: i32,
    base_offset: i32
) -> i32 {
    let mut mc: i32 = match_count
    // For each position in buf, check all needles
    let mut pos: i32 = 0
    while pos < buf_len {
        if mc >= max_matches {
            return mc
        }
        let b: u8 = buf[pos]
        let mut ni: i32 = 0
        while ni < needle_count {
            let nlen: i32 = needle_lens[ni]
            let noff: i32 = needle_offsets[ni]
            if pos + nlen <= buf_len {
                let first: u8 = needles[noff]
                if !(b < first) && !(first < b) {
                    // First byte matches — verify rest
                    let mut j: i32 = 1
                    let mut matched: i32 = 1
                    while j < nlen {
                        let bj: u8 = buf[pos + j]
                        let nj: u8 = needles[noff + j]
                        if bj < nj {
                            matched = 0
                            j = nlen
                        } else {
                            if nj < bj {
                                matched = 0
                                j = nlen
                            }
                        }
                        j = j + 1
                    }
                    if matched == 1 {
                        if mc < max_matches {
                            matches[mc] = base_offset + pos
                            needle_ids_out[mc] = ni
                            mc = mc + 1
                        }
                    }
                }
            }
            ni = ni + 1
        }
        pos = pos + 1
    }
    return mc
}
```

- [ ] **Step 3: Write search_buf_multi_simd — SIMD fast-skip with OR:ed bitmasks**

```ea
// SIMD multi-needle search: OR:ed .== + movemask per unique first-byte.
func search_buf_multi_simd(
    buf: *mut u8, buf_len: i32,
    needles: *restrict u8, needle_offsets: *restrict i32,
    needle_lens: *restrict i32, needle_count: i32,
    unique_fb: *restrict u8, unique_count: i32,
    matches: *restrict mut i32, needle_ids_out: *restrict mut i32,
    match_count: i32, max_matches: i32,
    base_offset: i32
) -> i32 {
    let mut mc: i32 = match_count
    let mut chunk_off: i32 = 0

    while chunk_off + 16 <= buf_len {
        if mc >= max_matches {
            return mc
        }
        let chunk: u8x16 = load(buf, chunk_off)

        // OR bitmasks for all unique first-bytes
        let mut bits: i32 = 0
        let mut fi: i32 = 0
        while fi < unique_count {
            bits = bits | movemask(chunk .== splat(unique_fb[fi]))
            fi = fi + 1
        }

        if bits == 0 {
            chunk_off = chunk_off + 16
        } else {
            // Scalar verify in this 16-byte region
            let scan_end: i32 = chunk_off + 16
            let mut i: i32 = chunk_off
            while i < scan_end {
                if mc >= max_matches {
                    return mc
                }
                let b: u8 = buf[i]
                let mut ni: i32 = 0
                while ni < needle_count {
                    let nlen: i32 = needle_lens[ni]
                    let noff: i32 = needle_offsets[ni]
                    if i + nlen <= buf_len {
                        let first: u8 = needles[noff]
                        if !(b < first) && !(first < b) {
                            let mut j: i32 = 1
                            let mut matched: i32 = 1
                            while j < nlen {
                                let bj: u8 = buf[i + j]
                                let nj: u8 = needles[noff + j]
                                if bj < nj {
                                    matched = 0
                                    j = nlen
                                } else {
                                    if nj < bj {
                                        matched = 0
                                        j = nlen
                                    }
                                }
                                j = j + 1
                            }
                            if matched == 1 {
                                if mc < max_matches {
                                    matches[mc] = base_offset + i
                                    needle_ids_out[mc] = ni
                                    mc = mc + 1
                                }
                            }
                        }
                    }
                    ni = ni + 1
                }
                i = i + 1
            }
            chunk_off = scan_end
        }
    }

    // Remaining < 16 bytes — delegate to scalar
    if chunk_off < buf_len {
        mc = search_buf_multi(buf, buf_len, needles, needle_offsets, needle_lens, needle_count,
                              matches, needle_ids_out, mc, max_matches, base_offset + chunk_off - chunk_off)
        // NOTE: Actually need to offset buf pointer. Since Ea doesn't have pointer arithmetic,
        // the scalar fallback should scan from chunk_off to buf_len within the same buffer.
        // Rewrite: inline the scalar tail here, scanning from chunk_off to buf_len.
    }

    return mc
}
```

**Important:** The scalar tail at the end of `search_buf_multi_simd` cannot call `search_buf_multi` with an offset into `buf` because Ea lacks pointer arithmetic. Instead, inline a scalar scan loop from `chunk_off` to `buf_len` directly in the function (same logic as `search_buf_multi` but starting at `chunk_off` instead of 0). The `base_offset` for this tail section is `base_offset + chunk_off`. During implementation, the agent should handle this by writing the tail loop inline.

- [ ] **Step 4: Write search_overlap_multi — multi-needle overlap search**

Adapts v1's `search_overlap` to test all needles:

```ea
func search_overlap_multi(
    overlap_buf: *restrict mut u8, overlap_len: i32,
    new_buf: *mut u8,
    needles: *restrict u8, needle_offsets: *restrict i32,
    needle_lens: *restrict i32, needle_count: i32,
    max_needle_len: i32,
    matches: *restrict mut i32, needle_ids_out: *restrict mut i32,
    match_count: i32, max_matches: i32,
    overlap_global_offset: i32,
    temp: *restrict mut u8
) -> i32 {
    if overlap_len == 0 {
        return match_count
    }
    // Copy overlap + first (max_needle_len - 1) bytes of new_buf into temp
    let new_bytes: i32 = max_needle_len - 1
    let mut ci: i32 = 0
    while ci < overlap_len {
        temp[ci] = overlap_buf[ci]
        ci = ci + 1
    }
    let mut ni: i32 = 0
    while ni < new_bytes {
        temp[overlap_len + ni] = new_buf[ni]
        ni = ni + 1
    }
    let temp_len: i32 = overlap_len + new_bytes
    return search_buf_multi(temp, temp_len, needles, needle_offsets, needle_lens, needle_count,
                            matches, needle_ids_out, match_count, max_matches, overlap_global_offset)
}
```

- [ ] **Step 5: Build to verify helpers compile**

Write a minimal export function stub so the kernel compiles:

```ea
export func chacha20_search_v2(
    key: *restrict i32, nonce: *restrict i32, ctr_init: i32,
    ct_u8: *restrict u8, len: i32,
    ks_i32: *restrict mut i32, ks_u8: *restrict mut u8,
    ct_i32: *restrict i32,
    pt_buf: *mut u8, pt_i32: *mut i32,
    overlap: *restrict mut u8,
    needles: *restrict u8, needle_offsets: *restrict i32,
    needle_lens: *restrict i32, needle_count: i32,
    lines_buf: *restrict mut u8, lines_buf_cap: i32,
    line_offsets: *restrict mut i32, line_lens: *restrict mut i32,
    match_offsets: *restrict mut i32, needle_ids: *restrict mut i32,
    max_matches: i32, max_line_len: i32, window_size: i32,
    match_count: *restrict mut i32, lines_written: *restrict mut i32
) {
    match_count[0] = 0
    lines_written[0] = 0
}
```

Build: `EA=/usr/local/lib/python3.13/dist-packages/ea/bin/ea ./build.sh`

Fix any compilation errors in the helpers (likely: mut pointer annotations, u8 comparison syntax).

- [ ] **Step 6: Commit**

```bash
git add chacha20_search_v2.ea
git commit -m "feat(v2): kernel scaffolding — multi-needle search helpers + stub export"
```

---

## Task 3: Write context-line extraction helper

**Files:**
- Modify: `chacha20_search_v2.ea`

- [ ] **Step 1: Write find_line_start — backward \n search with SIMD**

```ea
// Find the start of the line containing position `pos` in buf.
// Scans backward for \n using .== + movemask. Returns position after \n,
// or 0 if no \n found (truncated line start).
// max_scan limits backward scan distance.
func find_line_start(buf: *mut u8, pos: i32, max_scan: i32) -> i32 {
    let newline_splat: u8x16 = splat(10)
    let scan_limit: i32 = pos - max_scan
    let mut scan_pos: i32 = pos - 16
    // SIMD scan backward in 16-byte chunks
    while scan_pos >= 0 && scan_pos >= scan_limit {
        let chunk: u8x16 = load(buf, scan_pos)
        let bits: i32 = movemask(chunk .== newline_splat)
        if bits != 0 {
            // Find highest set bit = rightmost \n in this chunk
            // Count from bit 15 down to 0
            let mut bit: i32 = 15
            while bit >= 0 {
                if (bits .>> bit) .& 1 != 0 {
                    let nl_pos: i32 = scan_pos + bit
                    if nl_pos < pos {
                        return nl_pos + 1  // line starts after \n
                    }
                }
                bit = bit - 1
            }
        }
        scan_pos = scan_pos - 16
    }
    // Scalar scan for remaining bytes between scan_limit and last SIMD chunk
    let scalar_start: i32 = pos - 1
    let scalar_end: i32 = scan_limit
    let mut si: i32 = scalar_start
    let nl_byte: u8 = 10
    while si >= 0 && si >= scalar_end {
        if !(buf[si] < nl_byte) && !(nl_byte < buf[si]) {
            return si + 1
        }
        si = si - 1
    }
    return 0  // truncated: no \n found
}
```

**Note:** The backward scan with `movemask` and bit extraction is tricky. The `bits .>> bit` approach finds the highest set bit position. The agent should verify this compiles — `i32` shift operators (`.>>`) are available in Ea. If `bits .>> bit` doesn't work on scalar i32, use a loop checking `bits .& (1 .<< bit)` or similar.

- [ ] **Step 2: Write find_line_end — forward \n search with SIMD**

```ea
// Find the end of the line containing position `pos` in buf.
// Scans forward for \n. Returns position of \n,
// or buf_len if no \n found (truncated line end).
// max_scan limits forward scan distance.
func find_line_end(buf: *mut u8, pos: i32, buf_len: i32, max_scan: i32) -> i32 {
    let newline_splat: u8x16 = splat(10)
    let scan_limit: i32 = pos + max_scan
    let mut scan_pos: i32 = pos
    // SIMD scan forward in 16-byte chunks
    while scan_pos + 16 <= buf_len && scan_pos < scan_limit {
        let chunk: u8x16 = load(buf, scan_pos)
        let bits: i32 = movemask(chunk .== newline_splat)
        if bits != 0 {
            // Find lowest set bit = leftmost \n in this chunk
            let mut bit: i32 = 0
            while bit < 16 {
                if (bits .>> bit) .& 1 != 0 {
                    return scan_pos + bit  // position of \n
                }
                bit = bit + 1
            }
        }
        scan_pos = scan_pos + 16
    }
    // Scalar scan for remaining bytes
    let nl_byte: u8 = 10
    let end: i32 = buf_len
    let mut si: i32 = scan_pos
    while si < end && si < scan_limit {
        if !(buf[si] < nl_byte) && !(nl_byte < buf[si]) {
            return si
        }
        si = si + 1
    }
    return buf_len  // truncated: no \n found
}
```

- [ ] **Step 3: Write extract_line — copy matched line to output buffer**

```ea
// Extract the line around a match at `match_pos` in buf, copy to lines_buf.
// Returns 1 if line was written, 0 if lines_buf is full.
func extract_line(
    buf: *mut u8, buf_len: i32, match_pos: i32,
    max_line_len: i32,
    lines_buf: *restrict mut u8, lines_buf_cap: i32, write_pos: i32,
    line_offsets: *restrict mut i32, line_lens: *restrict mut i32,
    line_idx: i32
) -> i32 {
    let ls: i32 = find_line_start(buf, match_pos, max_line_len)
    let le: i32 = find_line_end(buf, match_pos, buf_len, max_line_len)
    let mut line_len: i32 = le - ls
    if line_len > max_line_len {
        line_len = max_line_len
    }
    if write_pos + line_len > lines_buf_cap {
        return 0  // lines_buf full
    }
    // Copy line bytes
    let mut ci: i32 = 0
    while ci < line_len {
        lines_buf[write_pos + ci] = buf[ls + ci]
        ci = ci + 1
    }
    line_offsets[line_idx] = write_pos
    line_lens[line_idx] = line_len
    return 1  // success
}
```

- [ ] **Step 4: Build to verify**

```bash
EA=/usr/local/lib/python3.13/dist-packages/ea/bin/ea ./build.sh
```

Fix any issues (scalar i32 shift operators, comparison syntax, mut annotations).

- [ ] **Step 5: Commit**

```bash
git add chacha20_search_v2.ea
git commit -m "feat(v2): context-line extraction helpers (find_line_start/end, extract_line)"
```

---

## Task 4: Write the main export function

**Files:**
- Modify: `chacha20_search_v2.ea`

- [ ] **Step 1: Replace the stub export with the full implementation**

The main function structure:

```
1. Early exits (needle_count <= 0, > 64, len <= 0)
2. Init: compute max_needle_len, extract unique first-bytes, build fb_needle_map
3. Main loop: fill decrypt window → search overlap → multi-needle SIMD search with line extraction → save overlap → zero window
4. Write outputs
```

Key differences from v1:
- **Decrypt window fill:** Instead of search-after-each-256B, decrypt fills the full window (default 4 KB) before searching. The 4-block ILP, single-block, and tail tiers all write into `pt_buf` at increasing offsets within the window.
- **Search phase:** After filling the window, call `search_buf_multi_simd` on the entire window. For each match, call `extract_line` to copy the matched line.
- **pt_i32 offsets:** In v1, pt_i32 stores used fixed offsets (0..60) because pt_buf was reused each 256B iteration. In v2, pt_i32 offsets increment within the window (`fill_elem_off` tracks the i32 offset within pt_buf).

The implementation agent should:
1. Read `chacha20_search.ea` lines 249-480 for the v1 main function structure
2. Restructure into: outer loop (per window) → inner decrypt fill loop (3-tier, filling pt_buf) → search phase → line extraction → overlap save → zero
3. The 4-block ILP round loop, single-block round loop, and tail are copy-paste from v1 — only the store offsets change (use `fill_elem_off` instead of fixed 0..60)

**Pseudocode for the main loop:**

```
while global_offset < len:
    // Phase 1: Fill decrypt window
    window_fill = min(window_size, len - global_offset)
    fill_offset = 0  // byte offset within pt_buf
    fill_elem_off = 0  // i32 element offset within pt_i32

    // Tier 1: 4-block ILP (256B at a time into pt_buf)
    while fill_offset + 256 <= window_fill:
        // [4-block ILP decrypt: load from ct_i32 at ct_elem_off, store to pt_i32 at fill_elem_off..fill_elem_off+60]
        fill_offset += 256
        fill_elem_off += 64
        ct_elem_off += 64
        ctr += 4

    // Tier 2: single-block (64B at a time)
    while fill_offset + 64 <= window_fill:
        // [single-block decrypt: store to pt_i32 at fill_elem_off..fill_elem_off+12]
        fill_offset += 64
        fill_elem_off += 16
        ct_elem_off += 16
        ctr += 1

    // Tier 3: sub-block tail
    if fill_offset < window_fill:
        chacha20_block(key, nonce, ctr, ks_i32)
        // byte-by-byte XOR into pt_buf[fill_offset..]
        // use load_masked/store_masked for 16-byte chunks, scalar for remainder

    // Phase 2: Search overlap
    mc = search_overlap_multi(overlap, overlap_len, pt_buf, ...)

    // Phase 3: Multi-needle SIMD search of pt_buf[0..window_fill]
    // For each match found, extract_line
    // (This requires integrating line extraction into the search loop,
    //  or doing search first, then line extraction in a second pass over matches)

    // Simpler approach: search first (get all match positions), then extract lines
    old_mc = mc
    mc = search_buf_multi_simd(pt_buf, window_fill, ..., mc, max_matches, global_offset)
    // Extract lines for new matches
    for each match from old_mc to mc:
        match_pos_in_buf = match_offsets[i] - global_offset
        if can_write_line:
            extract_line(pt_buf, window_fill, match_pos_in_buf, ...)

    // Phase 4: Save overlap (last max_needle_len - 1 bytes)
    // Phase 5: Zero pt_buf

    global_offset += window_fill
```

**Important:** The "search first, extract lines second" approach is cleaner than integrating extraction into the SIMD search loop. The match offsets from the search phase index into pt_buf, so line extraction can find \n boundaries directly.

- [ ] **Step 2: Build and verify compilation**

```bash
EA=/usr/local/lib/python3.13/dist-packages/ea/bin/ea ./build.sh
```

- [ ] **Step 3: Commit**

```bash
git add chacha20_search_v2.ea
git commit -m "feat(v2): main export function — windowed decrypt + multi-needle search + line extraction"
```

---

## Task 5: Write test suite — regression + multi-needle tests

**Files:**
- Create: `test_search_v2.py`

- [ ] **Step 1: Write test infrastructure**

Follow `test_search.py` patterns. Load `chacha20.so`, `chacha20_search.so` (for v1 comparison), and `chacha20_search_v2.so`.

The v2 argtypes (26 params):
```python
_v2.chacha20_search_v2.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,  # key, nonce, ctr
    ct.POINTER(ct.c_uint8), ct.c_int32,                           # ct_u8, len
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),               # ks_i32, ks_u8
    ct.POINTER(ct.c_int32),                                        # ct_i32
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),               # pt_buf, pt_i32
    ct.POINTER(ct.c_uint8),                                        # overlap
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),               # needles, needle_offsets
    ct.POINTER(ct.c_int32), ct.c_int32,                           # needle_lens, needle_count
    ct.POINTER(ct.c_uint8), ct.c_int32,                           # lines_buf, lines_buf_cap
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),               # line_offsets, line_lens
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),               # match_offsets, needle_ids
    ct.c_int32, ct.c_int32, ct.c_int32,                           # max_matches, max_line_len, window_size
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),               # match_count, lines_written
]
```

Write helpers:
- `encrypt_data(plaintext_bytes)` — same as v1
- `search_v2(ciphertext, needle_list, max_matches=10000, max_line_len=1024, window_size=4096)` — returns `(match_offsets, needle_ids, lines)` where `lines` is a list of `bytes`
- `find_all_occurrences_multi(data, needles)` — Python reference: returns `[(offset, needle_idx), ...]`

- [ ] **Step 2: Write regression tests 1-9 (single needle, same as v1)**

Call v2 with a single needle. Verify match offsets are identical to v1 for:
1. Known offset, 2. No match, 3. Multiple matches, 4. Boundary positions,
5. Cross-block boundary, 6. Cross-iteration boundary, 7. Single-byte needle,
8. Max needle 64 bytes, 9. Overlapping matches

- [ ] **Step 3: Write multi-needle tests 10-14**

```python
# Test 10: Two needles, separate locations
# Test 11: Three needles, overlapping first-bytes ("ERROR", "EXIT")
# Test 12: Five needles, dense matches
# Test 13: One needle not present in multi-set
# Test 14: Two needles match at same offset ("AB" and "ABC")
```

- [ ] **Step 4: Write context-line tests 15-20**

```python
# Test 15: Match mid-line — verify full line between \n's
# Test 16: Match at line start
# Test 17: Match at line end (just before \n)
# Test 18: Multiple matches on same line — duplicate lines emitted
# Test 19: Line longer than max_line_len — truncated
# Test 20: Match near window boundary — truncated line
```

For these tests, create plaintext with embedded \n characters:
```python
pt = b"first line\nERROR something broke\nthird line\n"
```

- [ ] **Step 5: Write cross-verification and edge case tests 21-27**

```python
# Test 21: Random data + injected needles, compare v2 vs Python reference
# Test 22: NASA log subset (1 KB) — encrypt, search, verify against plaintext
# Test 23: lines_buf overflow (small lines_buf_cap)
# Test 24: needle_count=0 → 0 matches
# Test 25: needle_count=1 → identical to v1 match offsets
# Test 26: needle_count=65 → early return, 0 matches
# Test 27: Overlap match → line_lens[i] == 0
```

- [ ] **Step 6: Run tests, fix failures**

```bash
python3 test_search_v2.py
```

- [ ] **Step 7: Commit**

```bash
git add test_search_v2.py
git commit -m "test(v2): 27 tests — regression, multi-needle, context lines, edge cases"
```

---

## Task 6: Write benchmark suite

**Files:**
- Create: `bench_search_v2.py`

- [ ] **Step 1: Write synthetic benchmarks (64 MB)**

Four implementations:
1. Ea v2 multi-needle (3 needles, ["ERROR", "FATAL", "PANIC"])
2. Ea v1 single-needle x3 (call v1 kernel 3 times)
3. Ea decrypt → Python find x3
4. C memmem x3 on plaintext

Follow `bench_search.py` patterns. Pre-allocate all key/nonce arrays outside timed closures.

**v2 buffer allocation:**
```python
# pt_buf must be window_size bytes (default 4096)
window_size = 4096
pt_buf = (ct.c_uint8 * window_size)()
pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
overlap = (ct.c_uint8 * 64)()
# Multi-needle input
needles_packed = b"ERROR\x00FATAL\x00PANIC"
needles_buf = (ct.c_uint8 * len(needles_packed))(*needles_packed)
offsets = (ct.c_int32 * 3)(0, 6, 12)
lens = (ct.c_int32 * 3)(5, 5, 5)
# Line output
lines_buf_cap = 1024 * 1024  # 1 MB
lines_buf = (ct.c_uint8 * lines_buf_cap)()
line_offsets = (ct.c_int32 * max_matches)()
line_lens = (ct.c_int32 * max_matches)()
match_offsets = (ct.c_int32 * max_matches)()
needle_ids = (ct.c_int32 * max_matches)()
match_count = (ct.c_int32 * 1)()
lines_written = (ct.c_int32 * 1)()
```

- [ ] **Step 2: Add NASA log benchmarks (if downloaded)**

Check if `data/nasa_jul95.log` exists. If so, add benchmarks 5-7:
5. Ea v2 multi-needle on encrypted NASA log
6. Ea v1 x3 on encrypted NASA log
7. grep on plaintext NASA log

- [ ] **Step 3: Run benchmarks**

```bash
python3 bench_search_v2.py
```

- [ ] **Step 4: Commit**

```bash
git add bench_search_v2.py
git commit -m "bench(v2): multi-needle benchmark suite (synthetic + NASA)"
```

---

## Task 7: Update CLI for multi-needle

**Files:**
- Modify: `eachacha_grep.py`

- [ ] **Step 1: Update argparse to accept multiple needles**

Change `needle` from a single positional to `nargs='+'`:
```python
parser.add_argument("needles", nargs="+", help="Strings to search for")
```

- [ ] **Step 2: Route to v1 or v2 kernel based on needle count**

```python
if len(args.needles) == 1:
    # Use v1 kernel (simpler, slightly faster)
    # ... existing v1 code ...
else:
    # Use v2 kernel (multi-needle + context lines)
    # Pack needles, call chacha20_search_v2
    # Display results with [needle_name] prefix per match
```

- [ ] **Step 3: Test with single and multiple needles**

```bash
# Single needle (v1 path)
python3 eachacha_grep.py "ERROR" /tmp/test_encrypted.bin --key ... --nonce ...

# Multiple needles (v2 path)
python3 eachacha_grep.py "ERROR" "INFO" /tmp/test_encrypted.bin --key ... --nonce ...
```

- [ ] **Step 4: Commit**

```bash
git add eachacha_grep.py
git commit -m "feat: multi-needle support in eachacha-grep (v2 kernel)"
```

---

## Task 8: Integration test

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

```bash
cd /root/dev/eachacha
python3 test_vectors.py && python3 test_fused.py && python3 test_search.py && python3 test_search_v2.py
```

- [ ] **Step 2: Run v2 benchmarks**

```bash
python3 bench_search_v2.py
```

- [ ] **Step 3: Download NASA log and run NASA benchmarks (if not already done)**

```bash
bash data/download_nasa.sh
python3 bench_search_v2.py  # re-run, now includes NASA benchmarks
```

- [ ] **Step 4: Commit any fixes**

```bash
git add -A && git commit -m "fix: integration test fixes" || echo "Nothing to commit"
```
