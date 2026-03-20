# Searchable Cipher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fused ChaCha20-Decrypt + String-Match kernel that searches encrypted data in a single streaming pass, with benchmarks proving it competitive with grep on plaintext.

**Architecture:** The kernel mirrors `chacha20_fused.ea`'s 3-tier structure (4-block ILP → single-block → sub-block tail) but replaces stats accumulation with SIMD-accelerated string search. Decrypted plaintext flows through a 256-byte working buffer (`pt_buf`), searched via XOR+reduce_min fast-skip with scalar verify, then zeroed. An overlap buffer carries `needle_len-1` bytes between iterations for boundary matches.

**Tech Stack:** Ea SIMD language, Python 3 (ctypes, numpy), libc memmem, grep

**Spec:** `docs/superpowers/specs/2026-03-20-searchable-cipher-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `chacha20_search.ea` | Fused decrypt+search kernel (new) |
| `chacha20_search.py` | Auto-generated Python bindings (ea bind) |
| `test_search.py` | Search kernel test suite (new) |
| `bench_search.py` | Search benchmark suite (new) |
| `eachacha_grep.py` | CLI demo wrapper (new) |
| `build.sh` | Add search kernel build lines (modify) |

---

## Task 1: Update build.sh and verify Ea compiler works

**Files:**
- Modify: `build.sh`

- [ ] **Step 1: Add chacha20_search.ea build lines to build.sh**

In `build.sh`, add before the C reference build:

```bash
echo "Building chacha20_search.ea..."
$EA chacha20_search.ea --lib --opt-level=3
$EA bind chacha20_search.ea --python
```

- [ ] **Step 2: Verify existing build still works**

Run: `cd /root/dev/eachacha && ./build.sh`
Expected: existing kernels build successfully (chacha20_search.ea doesn't exist yet, so those lines will fail — that's expected)

- [ ] **Step 3: Verify existing tests still pass**

Run: `cd /root/dev/eachacha && python3 test_vectors.py && python3 test_fused.py`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add build.sh
git commit -m "build: add chacha20_search.ea to build script"
```

---

## Task 2: Write the search kernel — decrypt + scalar search (no SIMD fast-skip yet)

Start with a correct but simple kernel: decrypt into `pt_buf`, scalar byte-by-byte search, overlap handling. SIMD fast-skip is Task 3.

**Files:**
- Create: `chacha20_search.ea`

- [ ] **Step 1: Write chacha20_search.ea with rotation helpers and chacha20_block**

Copy the rotation helpers (`rotl16`, `rotl12`, `rotl8`, `rotl7`) and `chacha20_block` function from `chacha20_fused.ea` lines 1-65 (use `*i32` without `restrict` for internal functions, matching the fused kernel pattern). These are identical across all kernels.

**Ea syntax warning:** The search kernel uses operations not seen in the existing kernels: scalar u8 `==`, `!=`, `^`, and `splat()` on u8 for u8x16. These need compiler validation at build time. If any fail:
- For `==`: replace `buf[i] == needle[0]` with `!(buf[i] < needle[0]) && !(needle[0] < buf[i])` (using `<` which is proven to work on u8 in `chacha20_fused.ea` line 248)
- For `^` on scalar u8: use `load_masked`/`store_masked` with vector `.^` instead of byte-by-byte XOR in the tail
- For `splat(u8_val)`: try explicit type annotation or widen+truncate if needed

- [ ] **Step 2: Write the scalar search helper function**

This function searches a byte buffer for a needle, writing match offsets. It will be used by all three tiers.

```ea
// Search buf[0..buf_len) for needle, writing match offsets to matches array.
// Returns updated match_count. Matches resume from each byte (overlapping matches).
// base_offset is added to positions to get global byte offset.
func search_buf(
    buf: *restrict u8, buf_len: i32,
    needle: *restrict u8, needle_len: i32,
    matches: *restrict mut i32, match_count: i32, max_matches: i32,
    base_offset: i32
) -> i32 {
    let mut mc: i32 = match_count
    let limit: i32 = buf_len - needle_len + 1
    let mut i: i32 = 0
    while i < limit {
        if mc >= max_matches {
            return mc
        }
        if buf[i] == needle[0] {
            let mut j: i32 = 1
            let mut matched: i32 = 1
            while j < needle_len {
                if buf[i + j] != needle[j] {
                    matched = 0
                    j = needle_len  // break
                }
                j = j + 1
            }
            if matched == 1 {
                matches[mc] = base_offset + i
                mc = mc + 1
            }
        }
        i = i + 1
    }
    return mc
}
```

- [ ] **Step 3: Write the overlap search helper**

Searches the boundary region between iterations. Concatenates overlap + first bytes of new pt_buf into a temp buffer, scans positions 0..needle_len-2 only.

```ea
// Search the overlap boundary region for matches spanning iterations.
// overlap_buf[0..overlap_len) = tail of previous iteration
// new_buf[0..] = start of current iteration
// Only scans starting positions 0..overlap_len-1 (positions in new_buf
// are found by the main search).
func search_overlap(
    overlap_buf: *restrict u8, overlap_len: i32,
    new_buf: *restrict u8,
    needle: *restrict u8, needle_len: i32,
    matches: *restrict mut i32, match_count: i32, max_matches: i32,
    overlap_global_offset: i32,
    temp: *restrict mut u8  // caller provides needle_len*2 bytes scratch
) -> i32 {
    if overlap_len == 0 {
        return match_count
    }
    // Copy overlap + first (needle_len-1) bytes of new_buf into temp
    let new_bytes: i32 = needle_len - 1
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
    // Only scan positions 0..overlap_len-1 (later positions are in new_buf's range)
    let total_len: i32 = overlap_len + new_bytes
    return search_buf(temp, total_len, needle, needle_len,
                      matches, match_count, max_matches, overlap_global_offset)
}
```

Note: `search_buf` already limits to `buf_len - needle_len + 1`, so if `overlap_len < needle_len`, positions are naturally bounded. But we want to avoid double-counting matches that start in `new_buf`. The scan limit should be `min(overlap_len, total_len - needle_len + 1)`. Since `search_buf` scans up to `total_len - needle_len + 1 = overlap_len + new_bytes - needle_len + 1 = overlap_len`, this is exactly right — positions 0 through overlap_len-1 are scanned, matching our spec.

- [ ] **Step 4: Write the main export function — chacha20_search**

The 3-tier structure mirroring `chacha20_fused.ea`:

```ea
export func chacha20_search(
    key: *restrict i32, nonce: *restrict i32, ctr_init: i32,
    ct_u8: *restrict u8, len: i32,
    needle: *restrict u8, needle_len: i32,
    ks_i32: *restrict mut i32, ks_u8: *restrict u8,
    ct_i32: *restrict i32,
    pt_buf: *restrict mut u8, pt_i32: *restrict mut i32,
    overlap: *restrict mut u8,
    matches: *restrict mut i32, max_matches: i32,
    match_count: *restrict mut i32
) {
    match_count[0] = 0

    // Early exit for invalid needle
    if needle_len <= 0 {
        return
    }
    if needle_len > 64 {
        return
    }
    if len <= 0 {
        return
    }

    let row0_init: i32x4 = [1634760805, 857760878, 2036477234, 1797285236]i32x4
    let row1_init: i32x4 = load(key, 0)
    let row2_init: i32x4 = load(key, 4)
    let n0: i32 = nonce[0]
    let n1: i32 = nonce[1]
    let n2: i32 = nonce[2]
    let mut offset: i32 = 0
    let mut elem_off: i32 = 0
    let mut ctr: i32 = ctr_init
    let mut mc: i32 = 0
    let mut overlap_len: i32 = 0

    // Scratch for overlap search (128 bytes is enough: 63 + 63 + 2 spare)
    // ORDERING DEPENDENCY: We reuse ks_u8 as overlap_temp. This is safe because:
    // - ks_i32/ks_u8 are only consumed in Tier 3 (sub-block tail)
    // - In Tier 3, keystream is generated into ks_i32 first, consumed by XOR into pt_buf,
    //   then search_overlap is called (which writes to overlap_temp/ks_u8).
    //   By that point, the keystream has already been consumed, so the overwrite is safe.
    let overlap_temp: *mut u8 = ks_u8

    // ---- Tier 1: 4-block ILP hot loop (256 bytes per iteration) ----
    while offset + 256 <= len {
        // [4-block ILP decrypt — identical to chacha20_fused.ea lines 110-240
        //  but with reversed direction: load(ct_i32, ...) .^ keystream → store(pt_i32, ...)]
        let r3a_init: i32x4 = [ctr, n0, n1, n2]i32x4
        let r3b_init: i32x4 = [ctr + 1, n0, n1, n2]i32x4
        let r3c_init: i32x4 = [ctr + 2, n0, n1, n2]i32x4
        let r3d_init: i32x4 = [ctr + 3, n0, n1, n2]i32x4
        let mut a0: i32x4 = row0_init
        let mut a1: i32x4 = row1_init
        let mut a2: i32x4 = row2_init
        let mut a3: i32x4 = r3a_init
        let mut b0: i32x4 = row0_init
        let mut b1: i32x4 = row1_init
        let mut b2: i32x4 = row2_init
        let mut b3: i32x4 = r3b_init
        let mut c0: i32x4 = row0_init
        let mut c1: i32x4 = row1_init
        let mut c2: i32x4 = row2_init
        let mut c3: i32x4 = r3c_init
        let mut d0: i32x4 = row0_init
        let mut d1: i32x4 = row1_init
        let mut d2: i32x4 = row2_init
        let mut d3: i32x4 = r3d_init

        // 10 double-rounds (identical to chacha20_fused.ea lines 132-221)
        let mut round: i32 = 0
        while round < 10 {
            a0 = a0 .+ a1
            b0 = b0 .+ b1
            c0 = c0 .+ c1
            d0 = d0 .+ d1
            a3 = rotl16(a3 .^ a0)
            b3 = rotl16(b3 .^ b0)
            c3 = rotl16(c3 .^ c0)
            d3 = rotl16(d3 .^ d0)
            a2 = a2 .+ a3
            b2 = b2 .+ b3
            c2 = c2 .+ c3
            d2 = d2 .+ d3
            a1 = rotl12(a1 .^ a2)
            b1 = rotl12(b1 .^ b2)
            c1 = rotl12(c1 .^ c2)
            d1 = rotl12(d1 .^ d2)
            a0 = a0 .+ a1
            b0 = b0 .+ b1
            c0 = c0 .+ c1
            d0 = d0 .+ d1
            a3 = rotl8(a3 .^ a0)
            b3 = rotl8(b3 .^ b0)
            c3 = rotl8(c3 .^ c0)
            d3 = rotl8(d3 .^ d0)
            a2 = a2 .+ a3
            b2 = b2 .+ b3
            c2 = c2 .+ c3
            d2 = d2 .+ d3
            a1 = rotl7(a1 .^ a2)
            b1 = rotl7(b1 .^ b2)
            c1 = rotl7(c1 .^ c2)
            d1 = rotl7(d1 .^ d2)
            // Column → diagonal shuffle
            a1 = shuffle(a1, [1, 2, 3, 0])
            b1 = shuffle(b1, [1, 2, 3, 0])
            c1 = shuffle(c1, [1, 2, 3, 0])
            d1 = shuffle(d1, [1, 2, 3, 0])
            a2 = shuffle(a2, [2, 3, 0, 1])
            b2 = shuffle(b2, [2, 3, 0, 1])
            c2 = shuffle(c2, [2, 3, 0, 1])
            d2 = shuffle(d2, [2, 3, 0, 1])
            a3 = shuffle(a3, [3, 0, 1, 2])
            b3 = shuffle(b3, [3, 0, 1, 2])
            c3 = shuffle(c3, [3, 0, 1, 2])
            d3 = shuffle(d3, [3, 0, 1, 2])
            // Diagonal round
            a0 = a0 .+ a1
            b0 = b0 .+ b1
            c0 = c0 .+ c1
            d0 = d0 .+ d1
            a3 = rotl16(a3 .^ a0)
            b3 = rotl16(b3 .^ b0)
            c3 = rotl16(c3 .^ c0)
            d3 = rotl16(d3 .^ d0)
            a2 = a2 .+ a3
            b2 = b2 .+ b3
            c2 = c2 .+ c3
            d2 = d2 .+ d3
            a1 = rotl12(a1 .^ a2)
            b1 = rotl12(b1 .^ b2)
            c1 = rotl12(c1 .^ c2)
            d1 = rotl12(d1 .^ d2)
            a0 = a0 .+ a1
            b0 = b0 .+ b1
            c0 = c0 .+ c1
            d0 = d0 .+ d1
            a3 = rotl8(a3 .^ a0)
            b3 = rotl8(b3 .^ b0)
            c3 = rotl8(c3 .^ c0)
            d3 = rotl8(d3 .^ d0)
            a2 = a2 .+ a3
            b2 = b2 .+ b3
            c2 = c2 .+ c3
            d2 = d2 .+ d3
            a1 = rotl7(a1 .^ a2)
            b1 = rotl7(b1 .^ b2)
            c1 = rotl7(c1 .^ c2)
            d1 = rotl7(d1 .^ d2)
            // Diagonal → column unshuffle
            a1 = shuffle(a1, [3, 0, 1, 2])
            b1 = shuffle(b1, [3, 0, 1, 2])
            c1 = shuffle(c1, [3, 0, 1, 2])
            d1 = shuffle(d1, [3, 0, 1, 2])
            a2 = shuffle(a2, [2, 3, 0, 1])
            b2 = shuffle(b2, [2, 3, 0, 1])
            c2 = shuffle(c2, [2, 3, 0, 1])
            d2 = shuffle(d2, [2, 3, 0, 1])
            a3 = shuffle(a3, [1, 2, 3, 0])
            b3 = shuffle(b3, [1, 2, 3, 0])
            c3 = shuffle(c3, [1, 2, 3, 0])
            d3 = shuffle(d3, [1, 2, 3, 0])
            round = round + 1
        }

        // Decrypt: XOR ciphertext with keystream → store to pt_buf (via pt_i32)
        // Note: reversed from encrypt kernels (ct_i32 is source, pt_i32 is dest)
        store(pt_i32, 0,  load(ct_i32, elem_off)      .^ (a0 .+ row0_init))
        store(pt_i32, 4,  load(ct_i32, elem_off + 4)   .^ (a1 .+ row1_init))
        store(pt_i32, 8,  load(ct_i32, elem_off + 8)   .^ (a2 .+ row2_init))
        store(pt_i32, 12, load(ct_i32, elem_off + 12)  .^ (a3 .+ r3a_init))
        store(pt_i32, 16, load(ct_i32, elem_off + 16)  .^ (b0 .+ row0_init))
        store(pt_i32, 20, load(ct_i32, elem_off + 20)  .^ (b1 .+ row1_init))
        store(pt_i32, 24, load(ct_i32, elem_off + 24)  .^ (b2 .+ row2_init))
        store(pt_i32, 28, load(ct_i32, elem_off + 28)  .^ (b3 .+ r3b_init))
        store(pt_i32, 32, load(ct_i32, elem_off + 32)  .^ (c0 .+ row0_init))
        store(pt_i32, 36, load(ct_i32, elem_off + 36)  .^ (c1 .+ row1_init))
        store(pt_i32, 40, load(ct_i32, elem_off + 40)  .^ (c2 .+ row2_init))
        store(pt_i32, 44, load(ct_i32, elem_off + 44)  .^ (c3 .+ r3c_init))
        store(pt_i32, 48, load(ct_i32, elem_off + 48)  .^ (d0 .+ row0_init))
        store(pt_i32, 52, load(ct_i32, elem_off + 52)  .^ (d1 .+ row1_init))
        store(pt_i32, 56, load(ct_i32, elem_off + 56)  .^ (d2 .+ row2_init))
        store(pt_i32, 60, load(ct_i32, elem_off + 60)  .^ (d3 .+ r3d_init))

        // Search overlap boundary (skip on first iteration)
        let overlap_base: i32 = offset - overlap_len
        mc = search_overlap(overlap, overlap_len, pt_buf, needle, needle_len,
                           matches, mc, max_matches, overlap_base, overlap_temp)

        // Search pt_buf[0..255]
        mc = search_buf(pt_buf, 256, needle, needle_len,
                       matches, mc, max_matches, offset)

        // Save last (needle_len - 1) bytes to overlap
        let save_len: i32 = needle_len - 1
        let save_start: i32 = 256 - save_len
        let mut si: i32 = 0
        while si < save_len {
            overlap[si] = pt_buf[save_start + si]
            si = si + 1
        }
        overlap_len = save_len

        // Zero pt_buf (security)
        let zero_vec: i32x4 = [0, 0, 0, 0]i32x4
        let mut zi: i32 = 0
        while zi < 64 {
            store(pt_i32, zi, zero_vec)
            zi = zi + 4
        }

        offset = offset + 256
        elem_off = elem_off + 64
        ctr = ctr + 4
    }

    // ---- Tier 2: Single-block loop (64 bytes) ----
    while offset + 64 <= len {
        let row3_init: i32x4 = [ctr, n0, n1, n2]i32x4
        let mut r0: i32x4 = row0_init
        let mut r1: i32x4 = row1_init
        let mut r2: i32x4 = row2_init
        let mut r3: i32x4 = row3_init
        let mut round: i32 = 0
        while round < 10 {
            r0 = r0 .+ r1
            r3 = rotl16(r3 .^ r0)
            r2 = r2 .+ r3
            r1 = rotl12(r1 .^ r2)
            r0 = r0 .+ r1
            r3 = rotl8(r3 .^ r0)
            r2 = r2 .+ r3
            r1 = rotl7(r1 .^ r2)
            r1 = shuffle(r1, [1, 2, 3, 0])
            r2 = shuffle(r2, [2, 3, 0, 1])
            r3 = shuffle(r3, [3, 0, 1, 2])
            r0 = r0 .+ r1
            r3 = rotl16(r3 .^ r0)
            r2 = r2 .+ r3
            r1 = rotl12(r1 .^ r2)
            r0 = r0 .+ r1
            r3 = rotl8(r3 .^ r0)
            r2 = r2 .+ r3
            r1 = rotl7(r1 .^ r2)
            r1 = shuffle(r1, [3, 0, 1, 2])
            r2 = shuffle(r2, [2, 3, 0, 1])
            r3 = shuffle(r3, [1, 2, 3, 0])
            round = round + 1
        }
        // Decrypt into pt_buf (64 bytes, offset 0 in pt_i32)
        store(pt_i32, 0,  load(ct_i32, elem_off)      .^ (r0 .+ row0_init))
        store(pt_i32, 4,  load(ct_i32, elem_off + 4)   .^ (r1 .+ row1_init))
        store(pt_i32, 8,  load(ct_i32, elem_off + 8)   .^ (r2 .+ row2_init))
        store(pt_i32, 12, load(ct_i32, elem_off + 12)  .^ (r3 .+ row3_init))

        // Overlap search
        let overlap_base2: i32 = offset - overlap_len
        mc = search_overlap(overlap, overlap_len, pt_buf, needle, needle_len,
                           matches, mc, max_matches, overlap_base2, overlap_temp)

        // Main search
        mc = search_buf(pt_buf, 64, needle, needle_len,
                       matches, mc, max_matches, offset)

        // Save overlap
        let save_len2: i32 = needle_len - 1
        let save_start2: i32 = 64 - save_len2
        let mut si2: i32 = 0
        while si2 < save_len2 {
            overlap[si2] = pt_buf[save_start2 + si2]
            si2 = si2 + 1
        }
        overlap_len = save_len2

        // Zero pt_buf (first 64 bytes)
        let zero_vec2: i32x4 = [0, 0, 0, 0]i32x4
        store(pt_i32, 0, zero_vec2)
        store(pt_i32, 4, zero_vec2)
        store(pt_i32, 8, zero_vec2)
        store(pt_i32, 12, zero_vec2)

        offset = offset + 64
        elem_off = elem_off + 16
        ctr = ctr + 1
    }

    // ---- Tier 3: Sub-block tail (< 64 bytes) ----
    if offset < len {
        chacha20_block(key, nonce, ctr, ks_i32)
        let remaining: i32 = len - offset

        // XOR ciphertext with keystream byte-by-byte into pt_buf
        let mut bi: i32 = 0
        while bi < remaining {
            pt_buf[bi] = ct_u8[offset + bi] ^ ks_u8[bi]
            bi = bi + 1
        }

        // Overlap search
        let overlap_base3: i32 = offset - overlap_len
        mc = search_overlap(overlap, overlap_len, pt_buf, needle, needle_len,
                           matches, mc, max_matches, overlap_base3, overlap_temp)

        // Main search
        mc = search_buf(pt_buf, remaining, needle, needle_len,
                       matches, mc, max_matches, offset)

        // Zero pt_buf and overlap (final cleanup)
        let mut zi3: i32 = 0
        while zi3 < remaining {
            pt_buf[zi3] = 0
            zi3 = zi3 + 1
        }
    }

    // Zero overlap on exit
    let mut oi: i32 = 0
    while oi < overlap_len {
        overlap[oi] = 0
        oi = oi + 1
    }

    match_count[0] = mc
}
```

**Critical note on pt_i32 offsets:** In the encrypt kernels, `store(ct_i32, elem_off, ...)` uses `elem_off` which increments globally. In our search kernel, `pt_buf` is a fixed 256-byte buffer reused each iteration. So we store at offsets 0..60 (Tier 1) or 0..12 (Tier 2) within `pt_i32`, NOT at `elem_off`. The `elem_off` is only used for reading from `ct_i32` (the ciphertext input). This is the key difference from the encrypt kernels.

**Critical note on overlap_temp:** We reuse `ks_u8` as scratch for `search_overlap`'s temp buffer. This works because `ks_i32`/`ks_u8` are only used in Tier 3 (sub-block tail), and `search_overlap` needs at most 126 bytes of temp space (63 + 63). The 256-byte `ks_i32` buffer is large enough. However, in Tier 3 we generate keystream into `ks_i32` first, then XOR into `pt_buf`, so by the time we call `search_overlap` in Tier 3 we've already consumed the keystream. The temp buffer is safe to reuse at that point.

- [ ] **Step 5: Build the kernel**

Run: `cd /root/dev/eachacha && ./build.sh`
Expected: all three kernels compile. `chacha20_search.so` and `chacha20_search.py` are generated.

If the build fails due to Ea syntax issues (e.g., `^` vs `.^` for scalar XOR, function pointer restrictions, etc.), fix them iteratively. The Ea language has quirks documented in the project memory:
- Scalar XOR for u8 may need different syntax than vector `.^`
- No tuples — return values via pointer params
- `*restrict` for alias elimination

- [ ] **Step 6: Commit**

```bash
git add chacha20_search.ea build.sh
git commit -m "feat: fused ChaCha20 decrypt+search kernel (scalar search)"
```

---

## Task 3: Write the test suite

**Files:**
- Create: `test_search.py`

- [ ] **Step 1: Write test infrastructure and helper functions**

Follow the pattern from `test_fused.py`: load `.so` via ctypes, define `to_i32_array`, `make_scratch`, `check` functions.

```python
"""Test suite for fused ChaCha20 decrypt+search kernel."""
import ctypes as ct
import numpy as np
import struct
import sys
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load libraries
_lib = ct.CDLL(os.path.join(_HERE, "chacha20.so"))
_search = ct.CDLL(os.path.join(_HERE, "chacha20_search.so"))

# chacha20_encrypt argtypes (for cross-verification: encrypt plaintext, then search ciphertext)
_lib.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
]
_lib.chacha20_encrypt.restype = None

# chacha20_search argtypes (16 params)
_search.chacha20_search.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,  # key, nonce, ctr
    ct.POINTER(ct.c_uint8), ct.c_int32,                           # ct_u8, len
    ct.POINTER(ct.c_uint8), ct.c_int32,                           # needle, needle_len
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),               # ks_i32, ks_u8
    ct.POINTER(ct.c_int32),                                        # ct_i32
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),               # pt_buf, pt_i32
    ct.POINTER(ct.c_uint8),                                        # overlap
    ct.POINTER(ct.c_int32), ct.c_int32,                           # matches, max_matches
    ct.POINTER(ct.c_int32),                                        # match_count
]
_search.chacha20_search.restype = None

KEY_U32 = [
    0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
    0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
]
NONCE_U32 = [0x00000000, 0x4a000000, 0x00000000]
COUNTER = 1

passed = 0
failed = 0


def to_i32_array(values):
    arr = (ct.c_int32 * len(values))()
    for i, v in enumerate(values):
        arr[i] = ct.c_int32(v & 0xFFFFFFFF).value
    return arr


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


def encrypt_data(plaintext_bytes):
    """Encrypt plaintext using chacha20_encrypt, return ciphertext bytes."""
    size = len(plaintext_bytes)
    pt = (ct.c_uint8 * size)(*plaintext_bytes)
    ct_buf = (ct.c_uint8 * size)()
    scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    pt_i32 = ct.cast(pt, ct.POINTER(ct.c_int32))
    ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))
    _lib.chacha20_encrypt(
        to_i32_array(KEY_U32), to_i32_array(NONCE_U32), ct.c_int32(COUNTER),
        pt, ct_buf, ct.c_int32(size), ks_i32, ks_u8, pt_i32, ct_i32)
    return bytes(ct_buf)


def search_ciphertext(ciphertext_bytes, needle_bytes, max_matches=10000):
    """Run chacha20_search on ciphertext, return list of match offsets."""
    size = len(ciphertext_bytes)
    ct_buf = (ct.c_uint8 * size)(*ciphertext_bytes)
    needle = (ct.c_uint8 * len(needle_bytes))(*needle_bytes)
    # Scratch buffers
    ks_scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks_scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks_scratch, ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))
    pt_buf = (ct.c_uint8 * 256)()
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
    overlap = (ct.c_uint8 * 64)()
    matches = (ct.c_int32 * max_matches)()
    match_count = (ct.c_int32 * 1)()
    _search.chacha20_search(
        to_i32_array(KEY_U32), to_i32_array(NONCE_U32), ct.c_int32(COUNTER),
        ct_buf, ct.c_int32(size),
        needle, ct.c_int32(len(needle_bytes)),
        ks_i32, ks_u8, ct_i32,
        pt_buf, pt_i32, overlap,
        matches, ct.c_int32(max_matches), match_count)
    return [matches[i] for i in range(match_count[0])]


def find_all_occurrences(data, needle):
    """Python reference: find all (overlapping) occurrences of needle in data."""
    positions = []
    start = 0
    while True:
        pos = data.find(needle, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions
```

- [ ] **Step 2: Write correctness tests 1-4 (basic cases)**

```python
# Test 1: Known needle at known offset
print("=== Test 1: Known needle at known offset ===")
pt = bytearray(128)
pt[50:55] = b"ERROR"
ct_data = encrypt_data(bytes(pt))
results = search_ciphertext(ct_data, b"ERROR")
check("finds ERROR at offset 50", results == [50], f"got {results}")

# Test 2: Needle not present
print("\n=== Test 2: Needle not present ===")
pt2 = bytearray(256)
ct_data2 = encrypt_data(bytes(pt2))
results2 = search_ciphertext(ct_data2, b"ERROR")
check("zero matches", results2 == [], f"got {results2}")

# Test 3: Multiple matches
print("\n=== Test 3: Multiple matches ===")
pt3 = bytearray(256)
pt3[10:15] = b"ERROR"
pt3[100:105] = b"ERROR"
pt3[200:205] = b"ERROR"
ct_data3 = encrypt_data(bytes(pt3))
results3 = search_ciphertext(ct_data3, b"ERROR")
check("finds 3 matches", results3 == [10, 100, 200], f"got {results3}")

# Test 4: Needle at byte 0 and at last possible byte
print("\n=== Test 4: Boundary positions ===")
pt4 = bytearray(128)
pt4[0:5] = b"ERROR"
pt4[123:128] = b"ERROR"
ct_data4 = encrypt_data(bytes(pt4))
results4 = search_ciphertext(ct_data4, b"ERROR")
check("finds at 0 and 123", results4 == [0, 123], f"got {results4}")
```

- [ ] **Step 3: Write cross-block boundary tests (5-6)**

```python
# Test 5: Cross-block boundary (needle straddles bytes 62-66)
print("\n=== Test 5: Cross-block boundary (64-byte) ===")
pt5 = bytearray(256)
pt5[62:67] = b"ERROR"  # straddles block boundary at 64
ct_data5 = encrypt_data(bytes(pt5))
results5 = search_ciphertext(ct_data5, b"ERROR")
check("finds ERROR across 64-byte block boundary", results5 == [62], f"got {results5}")

# Test 6: Cross-iteration boundary (needle straddles bytes 254-258)
print("\n=== Test 6: Cross-iteration boundary (256-byte) ===")
pt6 = bytearray(512)
pt6[254:259] = b"ERROR"  # straddles 4-block iteration boundary at 256
ct_data6 = encrypt_data(bytes(pt6))
results6 = search_ciphertext(ct_data6, b"ERROR")
check("finds ERROR across 256-byte iteration boundary", results6 == [254], f"got {results6}")
```

- [ ] **Step 4: Write edge case tests (7-9)**

```python
# Test 7: Single-byte needle
print("\n=== Test 7: Single-byte needle ===")
pt7 = bytearray(64)
pt7[0] = ord('X')
pt7[32] = ord('X')
pt7[63] = ord('X')
ct_data7 = encrypt_data(bytes(pt7))
results7 = search_ciphertext(ct_data7, b"X")
check("finds single byte at 0, 32, 63", results7 == [0, 32, 63], f"got {results7}")

# Test 8: Needle length 64 (maximum)
print("\n=== Test 8: Max needle length (64) ===")
needle8 = b"A" * 64
pt8 = bytearray(256)
pt8[100:164] = needle8
ct_data8 = encrypt_data(bytes(pt8))
results8 = search_ciphertext(ct_data8, needle8)
check("finds 64-byte needle at offset 100", results8 == [100], f"got {results8}")

# Test 9: Overlapping matches
print("\n=== Test 9: Overlapping matches ===")
pt9 = bytearray(64)
pt9[10:14] = b"aaaa"
ct_data9 = encrypt_data(bytes(pt9))
results9 = search_ciphertext(ct_data9, b"aa")
check("finds overlapping 'aa' at 10, 11, 12", results9 == [10, 11, 12], f"got {results9}")
```

- [ ] **Step 5: Write cross-verification tests (10-11)**

```python
# Test 10: Cross-verification with Python reference — random data, multiple sizes
print("\n=== Test 10: Cross-verification (random data) ===")
random.seed(42)
for size in [64, 128, 256, 512, 1024, 4096]:
    pt_data = bytearray(random.randint(0, 255) for _ in range(size))
    # Inject some needles
    needle = b"FIND"
    for pos in range(0, size - 4, size // 5):
        pt_data[pos:pos+4] = needle
    pt_bytes = bytes(pt_data)
    expected = find_all_occurrences(pt_bytes, needle)
    ct_data_xv = encrypt_data(pt_bytes)
    got = search_ciphertext(ct_data_xv, needle)
    check(f"size={size}: kernel matches python reference",
          got == expected, f"expected {expected}, got {got}")

# Test 11: Realistic log data
print("\n=== Test 11: Realistic log data ===")
random.seed(123)
lines = []
for i in range(200):
    if random.random() < 0.1:
        lines.append(f"2026-03-20 12:00:{i:02d} ERROR connection refused\n")
    else:
        lines.append(f"2026-03-20 12:00:{i:02d} INFO request processed ok\n")
log_data = "".join(lines).encode()
expected_log = find_all_occurrences(log_data, b"ERROR")
ct_log = encrypt_data(log_data)
got_log = search_ciphertext(ct_log, b"ERROR")
check(f"log data: {len(expected_log)} ERRORs found",
      got_log == expected_log,
      f"expected {len(expected_log)} matches, got {len(got_log)}")
```

- [ ] **Step 6: Write size sweep and edge case tests (12-15)**

```python
# Test 12: Size sweep
print("\n=== Test 12: Size sweep ===")
for size in [0, 1, 15, 16, 63, 64, 65, 127, 128, 255, 256, 257, 1000, 4096, 1048576]:
    pt_sw = bytearray(size)
    needle_sw = b"AB"
    # Place needle at offset 0 if it fits
    if size >= 2:
        pt_sw[0:2] = needle_sw
    pt_bytes_sw = bytes(pt_sw)
    expected_sw = find_all_occurrences(pt_bytes_sw, needle_sw)
    ct_sw = encrypt_data(pt_bytes_sw)
    got_sw = search_ciphertext(ct_sw, needle_sw)
    check(f"size={size}", got_sw == expected_sw,
          f"expected {expected_sw}, got {got_sw}")

# Test 13: max_matches overflow protection
print("\n=== Test 13: max_matches overflow ===")
pt13 = bytes([ord('A')] * 256)
ct13 = encrypt_data(pt13)
results13 = search_ciphertext(ct13, b"A", max_matches=5)
check("max_matches=5 limits output to 5", len(results13) == 5,
      f"got {len(results13)} matches")

# Test 14: Empty needle
print("\n=== Test 14: Empty needle ===")
pt14 = bytearray(64)
ct14 = encrypt_data(bytes(pt14))
results14 = search_ciphertext(ct14, b"")
check("empty needle returns 0 matches", results14 == [], f"got {results14}")

# Test 15: Zero-length ciphertext
print("\n=== Test 15: Zero-length input ===")
results15 = search_ciphertext(b"", b"ERROR")
check("zero-length input returns 0 matches", results15 == [], f"got {results15}")

# Test 16: Needle > 64 bytes (graceful rejection)
print("\n=== Test 16: Needle too long (>64) ===")
pt16 = bytearray(256)
ct16 = encrypt_data(bytes(pt16))
results16 = search_ciphertext(ct16, b"A" * 65)
check("needle_len=65 returns 0 matches", results16 == [], f"got {results16}")

# Test 17: Tier 2 to Tier 3 overlap boundary
print("\n=== Test 17: Tier 2→Tier 3 boundary ===")
# Data size 350: Tier 1 handles 256 bytes, Tier 2 handles 64 (offset 256-320), Tier 3 handles 30 (320-350)
# Place needle straddling bytes 318-322 (Tier 2→Tier 3 boundary at 320)
pt17 = bytearray(350)
pt17[318:323] = b"ERROR"
ct17 = encrypt_data(bytes(pt17))
results17 = search_ciphertext(ct17, b"ERROR")
check("finds ERROR across Tier 2→Tier 3 boundary", results17 == [318], f"got {results17}")
```

- [ ] **Step 7: Add summary and run**

```python
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All tests passed!")
```

Run: `cd /root/dev/eachacha && python3 test_search.py`
Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add test_search.py
git commit -m "test: search kernel test suite (15 tests, cross-verification)"
```

---

## Task 4: Add SIMD fast-skip optimization

Replace the pure scalar search in the hot path with XOR + reduce_min fast-skip per u8x16 chunk.

**Files:**
- Modify: `chacha20_search.ea`

- [ ] **Step 1: Add fast-skip search function**

Add a new function that uses SIMD to skip u8x16 chunks where the first byte of the needle doesn't appear:

```ea
// SIMD-accelerated search: XOR + reduce_min to skip chunks without first-byte match.
// Falls back to scalar verify when a chunk might contain the first byte.
func search_buf_simd(
    buf: *restrict u8, buf_len: i32,
    needle: *restrict u8, needle_len: i32,
    matches: *restrict mut i32, match_count: i32, max_matches: i32,
    base_offset: i32
) -> i32 {
    let mut mc: i32 = match_count
    let first_byte: u8 = needle[0]
    let first_splat: u8x16 = splat(first_byte)
    let mut chunk_off: i32 = 0

    // Process full 16-byte chunks with SIMD fast-skip
    while chunk_off + 16 <= buf_len {
        if mc >= max_matches {
            return mc
        }
        let chunk: u8x16 = load(buf, chunk_off)
        let xored: u8x16 = chunk .^ first_splat
        if reduce_min(xored) != 0 {
            // No byte in this chunk matches first byte — skip
            chunk_off = chunk_off + 16
        } else {
            // Potential match — scalar scan this 16-byte region
            let scan_end: i32 = chunk_off + 16
            let mut i: i32 = chunk_off
            while i < scan_end {
                if mc >= max_matches {
                    return mc
                }
                if buf[i] == first_byte {
                    // Verify full needle
                    let mut j: i32 = 1
                    let mut matched: i32 = 1
                    while j < needle_len {
                        if i + j >= buf_len {
                            matched = 0
                            j = needle_len
                        } else {
                            if buf[i + j] != needle[j] {
                                matched = 0
                                j = needle_len
                            }
                        }
                        j = j + 1
                    }
                    if matched == 1 {
                        matches[mc] = base_offset + i
                        mc = mc + 1
                    }
                }
                i = i + 1
            }
            chunk_off = scan_end
        }
    }

    // Remaining bytes (< 16) — scalar
    let mut i: i32 = chunk_off
    let limit: i32 = buf_len - needle_len + 1
    while i < limit {
        if mc >= max_matches {
            return mc
        }
        if buf[i] == first_byte {
            let mut j: i32 = 1
            let mut matched: i32 = 1
            while j < needle_len {
                if buf[i + j] != needle[j] {
                    matched = 0
                    j = needle_len
                }
                j = j + 1
            }
            if matched == 1 {
                matches[mc] = base_offset + i
                mc = mc + 1
            }
        }
        i = i + 1
    }

    return mc
}
```

- [ ] **Step 2: Replace search_buf calls with search_buf_simd in Tier 1 and Tier 2**

In the main function body, replace:
```
mc = search_buf(pt_buf, 256, ...)  // Tier 1
mc = search_buf(pt_buf, 64, ...)   // Tier 2
```
with:
```
mc = search_buf_simd(pt_buf, 256, ...)  // Tier 1
mc = search_buf_simd(pt_buf, 64, ...)   // Tier 2
```

Keep `search_buf` for Tier 3 (sub-block tail < 64 bytes, not worth SIMD) and for `search_overlap` (at most 126 bytes).

- [ ] **Step 3: Build and test**

Run: `cd /root/dev/eachacha && ./build.sh && python3 test_search.py`
Expected: all 15+ tests still pass

- [ ] **Step 4: Commit**

```bash
git add chacha20_search.ea
git commit -m "perf: SIMD fast-skip for search (XOR + reduce_min per u8x16 chunk)"
```

---

## Task 5: Write the benchmark suite

**Files:**
- Create: `bench_search.py`

- [ ] **Step 1: Write benchmark infrastructure**

Follow `bench.py` patterns: system info, shared test data, bench harness. Generate test data with "ERROR" injected at ~1 per 4KB.

```python
"""Benchmark suite for searchable cipher — fused decrypt+search vs alternatives."""
import ctypes as ct
import ctypes.util
import numpy as np
import os
import sys
import time
import platform
import statistics
import resource
import subprocess
import tempfile
from pathlib import Path

_soft, _hard = resource.getrlimit(resource.RLIMIT_STACK)
_target = 64 * 1024 * 1024
if _soft != resource.RLIM_INFINITY and _soft < _target:
    _new = _target if _hard == resource.RLIM_INFINITY else min(_target, _hard)
    resource.setrlimit(resource.RLIMIT_STACK, (_new, _hard))

DATA_SIZE = 64 * 1024 * 1024  # 64 MB default
WARMUP = 3
TIMED = 10
NEEDLE = b"ERROR"

_HERE = Path(__file__).resolve().parent

# Load libraries
_ea = ct.CDLL(str(_HERE / "chacha20.so"))
_ea.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
]
_ea.chacha20_encrypt.restype = None

_search_lib = ct.CDLL(str(_HERE / "chacha20_search.so"))
_search_lib.chacha20_search.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_int32),
]
_search_lib.chacha20_search.restype = None

# libc memmem
_libc = ct.CDLL(ctypes.util.find_library("c"))
_libc.memmem.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
_libc.memmem.restype = ct.c_void_p

KEY_U32 = [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
           0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c]
NONCE_U32 = [0x00000000, 0x4a000000, 0x00000000]


def to_i32_array(values):
    arr = (ct.c_int32 * len(values))()
    for i, v in enumerate(values):
        arr[i] = ct.c_int32(v & 0xFFFFFFFF).value
    return arr
```

- [ ] **Step 2: Generate test data with injected needles**

```python
def generate_test_data(size, needle=NEEDLE, density=4096):
    """Generate random data with needle injected every ~density bytes."""
    rng = np.random.RandomState(42)
    data = rng.randint(0, 256, size=size, dtype=np.uint8)
    # Inject needles
    needle_arr = np.frombuffer(needle, dtype=np.uint8)
    count = 0
    for offset in range(0, size - len(needle), density):
        pos = offset + rng.randint(0, min(density, size - offset - len(needle)))
        data[pos:pos+len(needle)] = needle_arr
        count += 1
    return data, count


def encrypt_test_data(plaintext_np):
    """Encrypt numpy array, return ciphertext numpy array."""
    size = len(plaintext_np)
    ct_buf = np.empty(size, dtype=np.uint8)
    scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    pt_ptr = plaintext_np.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_ptr = ct_buf.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pt_i32 = ct.cast(pt_ptr, ct.POINTER(ct.c_int32))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    _ea.chacha20_encrypt(
        to_i32_array(KEY_U32), to_i32_array(NONCE_U32), ct.c_int32(1),
        pt_ptr, ct_ptr, ct.c_int32(size), ks_i32, ks_u8, pt_i32, ct_i32)
    return ct_buf
```

- [ ] **Step 3: Write the 5 benchmark functions**

```python
def bench(name, fn, data_size, warmup=WARMUP, timed=TIMED):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(timed):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    gbps = [data_size / t / 1e9 for t in times]
    med = statistics.median(gbps)
    sd = statistics.stdev(gbps) if len(gbps) > 1 else 0.0
    print(f"  {name:<48s}  {med:8.3f} GB/s  (sd {sd:.3f})")
    return med, sd


def bench_fused_search(ciphertext_np, needle):
    """Benchmark 1: Ea fused decrypt+search."""
    size = len(ciphertext_np)
    ct_ptr = ciphertext_np.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    needle_buf = (ct.c_uint8 * len(needle))(*needle)
    ks = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks, ct.POINTER(ct.c_uint8))
    pt_buf = (ct.c_uint8 * 256)()
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
    overlap = (ct.c_uint8 * 64)()
    max_m = 100000
    matches = (ct.c_int32 * max_m)()
    mc = (ct.c_int32 * 1)()
    # Pre-allocate key/nonce OUTSIDE timed loop
    key_arr = to_i32_array(KEY_U32)
    nonce_arr = to_i32_array(NONCE_U32)

    def fn():
        _search_lib.chacha20_search(
            key_arr, nonce_arr, ct.c_int32(1),
            ct_ptr, ct.c_int32(size),
            needle_buf, ct.c_int32(len(needle)),
            ks_i32, ks_u8, ct_i32,
            pt_buf, pt_i32, overlap,
            matches, ct.c_int32(max_m), mc)
    return bench("Ea fused decrypt+search", fn, size)


def bench_decrypt_then_python_find(ciphertext_np, plaintext_np, needle):
    """Benchmark 2: Decrypt to buffer, then Python bytes.find."""
    size = len(ciphertext_np)
    decrypt_buf = np.empty(size, dtype=np.uint8)
    scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    # Pre-allocate outside timed loop
    key_arr = to_i32_array(KEY_U32)
    nonce_arr = to_i32_array(NONCE_U32)
    ct_ptr = ciphertext_np.ctypes.data_as(ct.POINTER(ct.c_uint8))
    dec_ptr = decrypt_buf.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_i32_p = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    dec_i32 = ct.cast(dec_ptr, ct.POINTER(ct.c_int32))

    def fn():
        _ea.chacha20_encrypt(
            key_arr, nonce_arr, ct.c_int32(1),
            ct_ptr, dec_ptr, ct.c_int32(size), ks_i32, ks_u8, ct_i32_p, dec_i32)
        # Search
        data = bytes(decrypt_buf)
        pos = 0
        count = 0
        while True:
            pos = data.find(needle, pos)
            if pos == -1:
                break
            count += 1
            pos += 1
    return bench("Ea decrypt → Python find", fn, size)


def bench_decrypt_then_memmem(ciphertext_np, needle):
    """Benchmark 3: Decrypt to buffer, then libc memmem."""
    size = len(ciphertext_np)
    decrypt_buf = np.empty(size, dtype=np.uint8)
    scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    needle_buf = (ct.c_uint8 * len(needle))(*needle)
    needle_ptr = ct.cast(needle_buf, ct.c_void_p)
    nlen = len(needle)
    # Pre-allocate outside timed loop
    key_arr = to_i32_array(KEY_U32)
    nonce_arr = to_i32_array(NONCE_U32)
    ct_ptr = ciphertext_np.ctypes.data_as(ct.POINTER(ct.c_uint8))
    dec_ptr = decrypt_buf.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_i32_p = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    dec_i32 = ct.cast(dec_ptr, ct.POINTER(ct.c_int32))

    def fn():
        _ea.chacha20_encrypt(
            key_arr, nonce_arr, ct.c_int32(1),
            ct_ptr, dec_ptr, ct.c_int32(size), ks_i32, ks_u8, ct_i32_p, dec_i32)
        # memmem search
        base = ct.cast(dec_ptr, ct.c_void_p).value
        remaining = size
        count = 0
        while remaining >= nlen:
            result = _libc.memmem(ct.c_void_p(base), ct.c_size_t(remaining),
                                  needle_ptr, ct.c_size_t(nlen))
            if not result:
                break
            count += 1
            advance = result - base + 1
            base = base + advance
            remaining = remaining - advance
    return bench("Ea decrypt → C memmem", fn, size)


def bench_grep_file(plaintext_np, needle):
    """Benchmark 4: grep on plaintext file on disk."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(bytes(plaintext_np))
        tmpfile = f.name
    try:
        def fn():
            subprocess.run(["grep", "-c", needle.decode(), tmpfile],
                          capture_output=True)
        return bench("grep on plaintext file", fn, len(plaintext_np))
    finally:
        os.unlink(tmpfile)


def bench_memmem_plaintext(plaintext_np, needle):
    """Benchmark 5: C memmem on plaintext in memory."""
    size = len(plaintext_np)
    buf_ptr = plaintext_np.ctypes.data_as(ct.c_void_p)
    base_addr = buf_ptr.value
    needle_buf = (ct.c_uint8 * len(needle))(*needle)
    needle_ptr = ct.cast(needle_buf, ct.c_void_p)
    nlen = len(needle)

    def fn():
        base = base_addr
        remaining = size
        count = 0
        while remaining >= nlen:
            result = _libc.memmem(ct.c_void_p(base), ct.c_size_t(remaining),
                                  needle_ptr, ct.c_size_t(nlen))
            if not result:
                break
            count += 1
            advance = result - base + 1
            base = base + advance
            remaining = remaining - advance
    return bench("C memmem on plaintext (in-memory)", fn, size)
```

- [ ] **Step 4: Write main function with results table**

```python
def print_sysinfo():
    print("=" * 66)
    print("Searchable Cipher Benchmark")
    print("=" * 66)
    cpu_name = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    print(f"  CPU       : {cpu_name}")
    print(f"  Platform  : {platform.platform()}")
    print(f"  Data size : {DATA_SIZE / 1024 / 1024:.0f} MB")
    print(f"  Needle    : {NEEDLE!r}")
    print(f"  Warmup    : {WARMUP}   Timed: {TIMED}")
    print()


def main():
    print_sysinfo()

    plaintext, needle_count = generate_test_data(DATA_SIZE)
    ciphertext = encrypt_test_data(plaintext)
    print(f"  Injected ~{needle_count} needles into {DATA_SIZE/1024/1024:.0f} MB\n")

    print("=" * 66)
    print("Benchmarks")
    print("=" * 66)

    results = []
    results.append(("Ea fused decrypt+search",)     + bench_fused_search(ciphertext, NEEDLE))
    results.append(("Ea decrypt → Python find",)     + bench_decrypt_then_python_find(ciphertext, plaintext, NEEDLE))
    results.append(("Ea decrypt → C memmem",)        + bench_decrypt_then_memmem(ciphertext, NEEDLE))
    results.append(("grep on plaintext file",)        + bench_grep_file(plaintext, NEEDLE))
    results.append(("C memmem on plaintext",)         + bench_memmem_plaintext(plaintext, NEEDLE))

    print()
    print("=" * 66)
    print(f"{'Implementation':<48s}  {'GB/s':>8s}  {'stddev':>8s}")
    print("-" * 66)
    for name, med, sd in results:
        print(f"  {name:<46s}  {med:8.3f}  {sd:8.3f}")
    print("=" * 66)

    # Ratios
    fused = results[0][1]
    decrypt_memmem = results[2][1]
    memmem_plain = results[4][1]
    if decrypt_memmem > 0:
        print(f"\n  Fused vs decrypt+memmem:    {fused/decrypt_memmem:.2f}x")
    if memmem_plain > 0:
        print(f"  Fused vs plaintext memmem:  {fused/memmem_plain:.2f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run benchmark**

Run: `cd /root/dev/eachacha && python3 bench_search.py`
Expected: benchmark completes, prints table with GB/s for all 5 implementations

- [ ] **Step 6: Commit**

```bash
git add bench_search.py
git commit -m "bench: searchable cipher benchmark suite (5 implementations)"
```

---

## Task 6: Write the CLI demo wrapper

**Files:**
- Create: `eachacha_grep.py`

- [ ] **Step 1: Write eachacha_grep.py**

```python
#!/usr/bin/env python3
"""eachacha-grep: Search encrypted files without decrypting to disk.

Usage:
    python3 eachacha_grep.py NEEDLE ENCRYPTED_FILE --key KEY_HEX --nonce NONCE_HEX [--counter N] [--max-matches N] [--context]
"""
import argparse
import ctypes as ct
import mmap
import os
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ea = ct.CDLL(str(_HERE / "chacha20.so"))
_search = ct.CDLL(str(_HERE / "chacha20_search.so"))

_ea.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
]
_ea.chacha20_encrypt.restype = None

_search.chacha20_search.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_int32),
]
_search.chacha20_search.restype = None


def hex_to_i32_array(hex_str):
    """Convert hex string to ctypes i32 array (little-endian u32 words)."""
    raw = bytes.fromhex(hex_str)
    words = struct.unpack(f"<{len(raw)//4}I", raw)
    arr = (ct.c_int32 * len(words))()
    for i, w in enumerate(words):
        arr[i] = ct.c_int32(w & 0xFFFFFFFF).value
    return arr


def main():
    parser = argparse.ArgumentParser(description="Search encrypted files without decrypting to disk")
    parser.add_argument("needle", help="String to search for")
    parser.add_argument("file", help="Encrypted file to search")
    parser.add_argument("--key", required=True, help="32-byte key as hex")
    parser.add_argument("--nonce", required=True, help="12-byte nonce as hex")
    parser.add_argument("--counter", type=int, default=1, help="Initial counter (default: 1)")
    parser.add_argument("--max-matches", type=int, default=1000000, help="Max matches to return")
    parser.add_argument("--context", action="store_true", help="Show ±40 bytes around each match (decrypts context)")
    args = parser.parse_args()

    file_size = os.path.getsize(args.file)
    if file_size == 0:
        print("Empty file.", file=sys.stderr)
        return

    key = hex_to_i32_array(args.key)
    nonce = hex_to_i32_array(args.nonce)
    needle = args.needle.encode()
    needle_buf = (ct.c_uint8 * len(needle))(*needle)

    # mmap the ciphertext
    with open(args.file, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        ct_buf = (ct.c_uint8 * file_size).from_buffer_copy(mm)
        ct_ptr = ct.cast(ct_buf, ct.POINTER(ct.c_uint8))
        ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))

        # Allocate buffers
        ks = (ct.c_uint8 * 256)()
        ks_i32 = ct.cast(ks, ct.POINTER(ct.c_int32))
        ks_u8 = ct.cast(ks, ct.POINTER(ct.c_uint8))
        pt_buf = (ct.c_uint8 * 256)()
        pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
        overlap = (ct.c_uint8 * 64)()
        matches = (ct.c_int32 * args.max_matches)()
        mc = (ct.c_int32 * 1)()

        _search.chacha20_search(
            key, nonce, ct.c_int32(args.counter),
            ct_ptr, ct.c_int32(file_size),
            needle_buf, ct.c_int32(len(needle)),
            ks_i32, ks_u8, ct_i32,
            pt_buf, pt_i32, overlap,
            matches, ct.c_int32(args.max_matches), mc)

        count = mc[0]
        print(f"Found {count} match{'es' if count != 1 else ''} in {file_size:,} bytes")
        for i in range(count):
            offset = matches[i]
            print(f"  offset {offset}")

            if args.context and count <= 100:
                # Targeted decrypt of ±40 bytes around match
                ctx_start = max(0, offset - 40)
                ctx_end = min(file_size, offset + len(needle) + 40)
                ctx_len = ctx_end - ctx_start
                ctx_ct = (ct.c_uint8 * ctx_len)(*ct_buf[ctx_start:ctx_end])
                ctx_pt = (ct.c_uint8 * ctx_len)()
                ctx_scratch = (ct.c_uint8 * 256)()
                ctx_ks_i32 = ct.cast(ctx_scratch, ct.POINTER(ct.c_int32))
                ctx_ks_u8 = ct.cast(ctx_scratch, ct.POINTER(ct.c_uint8))
                ctx_ct_i32 = ct.cast(ctx_ct, ct.POINTER(ct.c_int32))
                ctx_pt_i32 = ct.cast(ctx_pt, ct.POINTER(ct.c_int32))
                # Calculate block-aligned counter for this region
                block_counter = args.counter + (ctx_start // 64)
                block_offset = ctx_start % 64
                # Decrypt from block boundary
                aligned_start = ctx_start - block_offset
                aligned_len = ctx_end - aligned_start
                aligned_ct = (ct.c_uint8 * aligned_len)(*ct_buf[aligned_start:aligned_start + aligned_len])
                aligned_pt = (ct.c_uint8 * aligned_len)()
                aligned_ct_i32 = ct.cast(aligned_ct, ct.POINTER(ct.c_int32))
                aligned_pt_i32 = ct.cast(aligned_pt, ct.POINTER(ct.c_int32))
                _ea.chacha20_encrypt(
                    key, nonce, ct.c_int32(block_counter),
                    aligned_ct, aligned_pt, ct.c_int32(aligned_len),
                    ctx_ks_i32, ctx_ks_u8, aligned_ct_i32, aligned_pt_i32)
                context_bytes = bytes(aligned_pt[block_offset:block_offset + ctx_len])
                # Display, replacing non-printable chars
                display = context_bytes.decode("utf-8", errors="replace")
                print(f"    ...{display}...")

        mm.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the CLI with a generated encrypted file**

```bash
cd /root/dev/eachacha
# Generate test encrypted file using Python
python3 -c "
import ctypes as ct, struct, os
key_hex = '000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'
nonce_hex = '000000004a00000000000000'
plaintext = b'INFO normal log line\nERROR something broke\nINFO another line\n'
# Encrypt using chacha20.so
lib = ct.CDLL('./chacha20.so')
lib.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)]
lib.chacha20_encrypt.restype = None
key_bytes = bytes.fromhex(key_hex)
key_words = struct.unpack('<8I', key_bytes)
nonce_bytes = bytes.fromhex(nonce_hex)
nonce_words = struct.unpack('<3I', nonce_bytes)
key_arr = (ct.c_int32 * 8)(*[ct.c_int32(w & 0xFFFFFFFF).value for w in key_words])
nonce_arr = (ct.c_int32 * 3)(*[ct.c_int32(w & 0xFFFFFFFF).value for w in nonce_words])
pt = (ct.c_uint8 * len(plaintext))(*plaintext)
ct_buf = (ct.c_uint8 * len(plaintext))()
scratch = (ct.c_uint8 * 256)()
ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
pt_i32 = ct.cast(pt, ct.POINTER(ct.c_int32))
ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))
lib.chacha20_encrypt(key_arr, nonce_arr, ct.c_int32(1), pt, ct_buf, ct.c_int32(len(plaintext)), ks_i32, ks_u8, pt_i32, ct_i32)
with open('/tmp/test_encrypted.bin', 'wb') as f:
    f.write(bytes(ct_buf))
print(f'Wrote {len(plaintext)} bytes encrypted to /tmp/test_encrypted.bin')
"
python3 eachacha_grep.py "ERROR" /tmp/test_encrypted.bin \
    --key 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f \
    --nonce 000000004a00000000000000 \
    --context
```

Expected: finds "ERROR" at the correct offset, shows context line

- [ ] **Step 3: Commit**

```bash
git add eachacha_grep.py
git commit -m "feat: eachacha-grep CLI for searching encrypted files"
```

---

## Task 7: Integration test — full pipeline verification

Run all tests and benchmarks end-to-end to verify everything works together.

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

```bash
cd /root/dev/eachacha
python3 test_vectors.py && python3 test_fused.py && python3 test_search.py
```

Expected: all tests pass

- [ ] **Step 2: Run search benchmark**

```bash
cd /root/dev/eachacha && python3 bench_search.py
```

Expected: benchmark table with GB/s numbers for all 5 implementations

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: integration test fixes"
```

(Only if changes were needed)
