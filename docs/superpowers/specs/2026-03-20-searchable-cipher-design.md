# The Searchable Cipher (Fusion-magi)

**Date:** 2026-03-20
**Status:** Approved
**Project:** eachacha

## Summary

A fused ChaCha20-Decrypt + String-Match kernel that searches encrypted data in a single streaming pass. Plaintext exists only in a small 256-byte working buffer (hot in L1 cache), never as a full-file allocation. The buffer is zeroed after each iteration.

**Pitch:** "I can search my encrypted database faster than Linux grep can search plaintext."

**Scenario:** 100 GB encrypted log data. The normal way: decrypt all to RAM/disk, then grep (two passes, full plaintext exposure). The Ea way: stream-decrypt 256 bytes at a time, search, discard. Single pass, bounded plaintext exposure (256 bytes + 63-byte overlap, zeroed per iteration).

## Security Model

**What we guarantee:**
- No full-file plaintext buffer — only 256 bytes live at any time
- Working buffer zeroed after each iteration
- Plaintext never written to disk
- Only match offsets leave the kernel — no plaintext in output

**What we don't claim:**
- "Zero plaintext in RAM" — the 256-byte working buffer and 63-byte overlap buffer are in RAM (L1 cache). This is an inherent constraint of Ea's current type system: no bitcast between i32x4 and u8x16, so decrypted i32x4 must be stored and reloaded as u8x16 for byte-level search. This matches the pattern used by the existing `chacha20_fused.ea` stats kernel.

**Why this is still compelling:** The alternative (decrypt-then-grep) materializes the *entire file* as plaintext. 256 bytes vs 100 GB is a 400-million-fold reduction in exposure surface.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Demo scenario | Log search (fixed strings like "ERROR") | Most relatable real-world use case |
| Match output | Absolute byte offset only | Keeps kernel simple; caller handles presentation |
| Pattern type | Single fixed string per call | Sufficient for "faster than grep" benchmark; multi-string is v2 |
| Boundary handling | Overlap buffer (needle_len - 1 bytes between iterations) | Correct solution; tiny cost; no missed matches |
| Search algorithm | XOR + reduce_min fast-skip, scalar verify | Best algorithm achievable with Ea's current primitives (no cmpeq on u8x16) |
| Benchmark suite | Full matrix: fused vs decrypt-then-grep vs plaintext grep, 64KB-256MB | Proves consistent scaling, matches existing bench.py rigor |
| Overlapping matches | Reported (resume from p+1, not p+needle_len) | Correct grep-like semantics |

## Ea Primitive Constraints

The search algorithm is shaped by what Ea currently supports on u8x16:

**Available:** `load`, `load_masked`, `store`, `store_masked`, `shuffle`, `splat`, `.^` (XOR), `.&` (AND), `reduce_min`, `reduce_max`, `reduce_add`, `widen_u8_i32x4`

**Not available:** `cmpeq` (byte comparison → mask), `bitcast` (i32x4 ↔ u8x16), byte extraction by index from u8x16

This means we cannot do a true SIMD first-byte filter with bitmask extraction. Instead, we use XOR + reduce_min as a fast-skip heuristic, with scalar verify as the workhorse.

## Kernel Architecture

### New file: `chacha20_search.ea`

Fuses ChaCha20 decryption with streaming string search. Structure mirrors `chacha20_fused.ea`: same 4-block ILP decrypt, but replaces stats accumulation with search logic operating on a 256-byte working buffer.

### Signature

```
chacha20_search(
    key: *restrict i32,          // 32-byte key (i32 for ChaCha20 state loads)
    nonce: *restrict i32,        // 12-byte nonce (i32 for ChaCha20 state loads)
    ctr: i32,                    // initial counter
    ct_u8: *restrict u8,         // ciphertext input
    len: i32,                    // ciphertext length
    needle: *restrict u8,        // search pattern
    needle_len: i32,             // pattern length (1-64)
    ks_i32: *restrict i32,       // keystream buffer (4-block = 256 bytes)
    ks_u8: *restrict u8,         // alias of ks_i32
    ct_i32: *restrict i32,       // i32 alias of ciphertext
    pt_buf: *restrict u8,        // 256-byte plaintext working buffer (caller-allocated)
    pt_i32: *restrict i32,       // i32 alias of pt_buf
    overlap: *restrict u8,       // 64-byte overlap buffer (caller-allocated)
    matches: *restrict i32,      // output: array of match offsets
    max_matches: i32,            // capacity of matches array
    match_count: *restrict i32   // output: number of matches found (initialized by kernel)
)
```

### Three-Tier Processing (matches existing kernels)

**Early exit:** If `needle_len == 0`, set `*match_count = 0` and return immediately.

**Overlap buffer persists across all three tiers.** The same overlap state carries from the last Tier 1 iteration into the first Tier 2 iteration, and from the last Tier 2 iteration into Tier 3.

**Tier 1: 4-block hot loop (256 bytes per iteration)**
1. Decrypt 4 blocks: read from `ct_i32`, XOR with i32x4 keystream, store to `pt_i32` (decrypt direction: ct → pt, opposite of encrypt kernels which do pt → ct)
2. Search the overlap region: scan starting positions `0` through `needle_len - 2` in the concatenated `overlap[0..needle_len-2] + pt_buf[0..needle_len-2]` region for boundary-spanning matches (scalar, skipped on first iteration when overlap is empty)
3. Search `pt_buf[0..255]` using XOR + reduce_min fast-skip per u8x16 chunk, scalar verify on candidates
4. Save last `needle_len - 1` bytes of `pt_buf` into `overlap`
5. Zero `pt_buf` (security hygiene)

**Tier 2: Single-block loop (64 bytes)**
Same decrypt → overlap-search → main-search → save-overlap → zero pattern, operating on 64-byte chunks within `pt_buf`. Overlap carries from Tier 1.

**Tier 3: Sub-block tail (< 64 bytes)**
Generate keystream via `chacha20_block()` into `ks_i32`, XOR with ciphertext byte-by-byte into `pt_buf`, search the partial block. Handle overlap from Tier 2. No SIMD fast-skip (tail is < 64 bytes, not worth it). Zero `pt_buf` and `overlap` after final search (security: clear both buffers on exit).

### Search Algorithm

**Fast-skip (per u8x16 chunk):**

```
xored: u8x16 = load(pt_buf, offset) .^ splat(needle[0])
if reduce_min(xored) != 0:
    skip  // no byte in this chunk matches needle[0]
```

When `reduce_min` is nonzero, no byte equals `needle[0]` — skip the entire 16-byte chunk. For random data with a specific first byte, ~93.5% of chunks are skipped (1 - (1 - 1/256)^16 ≈ 6.1% hit rate). Note: real ASCII log data is biased toward printable characters, so hit rates will be higher (e.g., 'E' appears more often than 1/256). The fast-skip is still beneficial but less dominant on real text.

**Scalar verify (when chunk has candidates):**

When a chunk cannot be skipped, scan it byte-by-byte:
```
for i in 0..16:
    if pt_buf[offset + i] == needle[0]:
        // verify needle[1..needle_len] at offset+i+1
        // on full match: write offset to matches array
```

This is simple and correct. The fast-skip ensures the scalar path is rarely taken.

**Overlap handling:**

```
overlap[0..needle_len-2]  // last bytes from previous iteration
```

At the start of each iteration (except the first), concatenate `overlap` with the first `needle_len - 1` bytes of the new `pt_buf`. Scan only starting positions `0` through `needle_len - 2` in this concatenated region — later positions fall entirely within `pt_buf` and will be found by the main search. This avoids double-counting.

The overlap buffer is initialized empty (length 0). On the first iteration, the overlap scan is skipped.

**Match output:**

```
if *match_count < max_matches:
    matches[*match_count] = global_byte_offset
    *match_count += 1
```

**Global offset tracking:** The kernel maintains `iter_base` = byte offset of the current iteration's start within the ciphertext. For main-search matches: `global_byte_offset = iter_base + position_within_pt_buf`. For overlap matches: `global_byte_offset = iter_base - (needle_len - 1) + position_within_overlap_region`.

Kernel initializes `*match_count = 0` on entry. The `max_matches` parameter prevents buffer overflow.

**Needle length constraint:** v1 supports 1-64 bytes (one ChaCha20 block). Covers any realistic grep-style fixed string.

## Python Bindings & CLI

### Auto-generated: `chacha20_search.py`

Built by `ea bind chacha20_search.ea --python`. Wraps ctypes call with numpy array allocation for `pt_buf`, `overlap`, `matches`, and `match_count`.

### CLI wrapper: `eachacha_grep.py`

```
python3 eachacha_grep.py "ERROR" encrypted_logs.bin --key <hex> --nonce <hex>
```

Workflow:
1. mmap the ciphertext file
2. Allocate matches array (pre-sized, e.g., 1M entries — 4MB)
3. Call `chacha20_search()`
4. Print match offsets

Optional: targeted second-pass decrypt of ±80 bytes around each match using existing `chacha20_encrypt()` to show context lines. Only matched regions decrypted to user-visible memory.

### Build script update

Add to `build.sh`:
```bash
echo "Building chacha20_search.ea..."
$EA chacha20_search.ea --lib --opt-level=3
$EA bind chacha20_search.ea --python
```

## Test Suite

### New file: `test_search.py`

**Correctness tests:**
1. Known plaintext with known needle at known offset — exact position verified
2. Needle not present — zero matches
3. Multiple matches — all found with correct offsets
4. Needle at byte 0 and at last possible byte
5. Cross-block boundary match (needle straddling bytes 62-66)
6. Cross-iteration boundary match (needle straddling bytes 254-258)
7. Single-byte needle
8. Needle length == 64 (maximum)
9. Overlapping matches: "aa" in "aaaa" → matches at 0, 1, 2

**Cross-verification tests:**
10. Decrypt with `chacha20_encrypt()`, find all occurrences via Python `str.find()` loop, compare offsets — random data, multiple sizes
11. Same with realistic log-like data ("ERROR" scattered at random positions)

**Size sweep:**
12. Sizes: 0, 1, 15, 16, 63, 64, 65, 127, 128, 255, 256, 257, 1000, 4096, 1MB

**Edge cases:**
13. matches array too small (max_matches < actual matches) — verify no overflow, match_count == max_matches
14. Empty needle (needle_len == 0) — return 0 matches
15. First iteration overlap skip — no false matches from uninitialized overlap

Follows existing test style: pytest, numpy, ctypes.

## Benchmark Suite

### New file: `bench_search.py`

Throughput (GB/s) across 64KB → 256MB.

| # | Implementation | What it measures |
|---|----------------|-----------------|
| 1 | Ea fused decrypt+search | The new kernel — single pass, bounded plaintext |
| 2 | Ea decrypt → Python find | Two-pass: `chacha20_encrypt()` to buffer, then `bytes.find()` |
| 3 | Ea decrypt → C memmem | Two-pass: decrypt to buffer, then libc `memmem()` via ctypes |
| 4 | grep on plaintext file | Pre-decrypted file on disk, `subprocess.run(["grep", "-c"])`. Only meaningful at large sizes (includes process overhead). |
| 5 | C memmem on plaintext in-memory | `memmem` via ctypes on plaintext numpy buffer. Pure search speed baseline. |

**Test data (v1):** Random bytes with "ERROR" injected at ~1 per 4KB (realistic log density). Same data encrypted for benchmarks 1-3, plaintext for 4-5.

**v2 data:** Real public log dataset (public access logs, HTTP traffic corpus, etc.) for more credible real-world benchmarks.

**Output:** Median GB/s, stddev, table across all sizes. Same format as existing `bench.py`.

**Target headlines:**
- Fused vs two-pass → fusion speedup ratio
- Fused on encrypted vs grep/memmem on plaintext → the "wow" number
- Consistent scaling across sizes → credibility

## New Files

| File | Purpose |
|------|---------|
| `chacha20_search.ea` | Fused decrypt+search kernel |
| `chacha20_search.py` | Auto-generated Python bindings |
| `eachacha_grep.py` | CLI demo wrapper |
| `test_search.py` | Test suite (15+ tests) |
| `bench_search.py` | Benchmark suite |

## Modified Files

| File | Change |
|------|--------|
| `build.sh` | Add chacha20_search.ea build + bind lines |

## v2 Roadmap (out of scope for v1)

- Multiple fixed strings per call (["ERROR", "FATAL", "PANIC"])
- Case-insensitive search (ASCII case folding in kernel)
- Context-line extraction in kernel (find \n boundaries, copy matched line)
- Real public log dataset for benchmarks
- Parallel multi-core search (ThreadPoolExecutor, split file into chunks)
- If Ea adds `cmpeq` on u8x16: true SIMD first-byte filter with bitmask (replaces XOR+reduce_min heuristic)
