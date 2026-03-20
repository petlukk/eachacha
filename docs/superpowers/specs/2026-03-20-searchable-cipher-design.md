# The Searchable Cipher (Fusion-magi)

**Date:** 2026-03-20
**Status:** Approved
**Project:** eachacha

## Summary

A fused ChaCha20-Decrypt + String-Match kernel that searches encrypted data without ever writing plaintext to RAM. Decrypted bytes live and die in CPU registers — no `store` instruction for plaintext.

**Pitch:** "I can search my encrypted database faster than Linux grep can search plaintext."

**Scenario:** 100 GB encrypted log data. The normal way: decrypt all to RAM, then grep. The Ea way: decrypt in registers, search in registers, discard. Zero plaintext exposure.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Demo scenario | Log search (fixed strings like "ERROR") | Most relatable real-world use case |
| Match output | Block index + byte offset only | Keeps kernel simple; caller handles presentation |
| Pattern type | Single fixed string per call | Sufficient for "faster than grep" benchmark; multi-string is v2 |
| Boundary handling | Overlap buffer (needle_len - 1 bytes between iterations) | Correct solution; tiny cost; no missed matches |
| Search algorithm | SIMD first-byte filter + scalar verify | Same algorithm class as glibc memmem / GNU grep; genuinely SIMD-accelerated search |
| Benchmark suite | Full matrix: fused vs decrypt-then-grep vs plaintext grep, 64KB-256MB | Proves consistent scaling, matches existing bench.py rigor |

## Kernel Architecture

### New file: `chacha20_search.ea`

Fuses ChaCha20 decryption with SIMD string search in a single pass. Structure mirrors `chacha20_fused.ea` but replaces stats accumulation with search logic.

### Signature

```
chacha20_search(
    key: *restrict u8,           // 32-byte key
    nonce: *restrict u8,         // 12-byte nonce
    ctr: i32,                    // initial counter
    ct_u8: *restrict u8,         // ciphertext input
    len: i32,                    // ciphertext length
    needle: *restrict u8,        // search pattern
    needle_len: i32,             // pattern length (1-64)
    ks_i32: *restrict i32,       // keystream buffer (4-block = 256 bytes)
    ks_u8: *restrict u8,         // alias of ks_i32
    ct_i32: *restrict i32,       // i32 alias of ciphertext
    matches: *restrict i32,      // output: array of match offsets
    match_count: *restrict i32   // output: number of matches found
)
```

No plaintext pointer. Decrypted bytes exist only in registers, never stored to RAM. This is the security property.

### Hot Loop (4-block, 256 bytes per iteration)

1. Decrypt 4 blocks into i32x4 registers (same ILP pattern as `chacha20.ea`)
2. Reinterpret decrypted i32x4 as u8x16 chunks (16 chunks per 4-block iteration)
3. For each u8x16 chunk: `cmpeq` against `needle[0]`, extract bitmask
4. For each set bit: verify remaining `needle[1..needle_len]` bytes against decrypted data in registers
5. On match: write `global_offset` to `matches[*match_count]`, increment `match_count`
6. Before moving to next iteration: save last `needle_len - 1` decrypted bytes into overlap buffer for boundary matching

### Search Algorithm

**First-byte filter:**

For each decrypted u8x16 chunk, broadcast `needle[0]` to all 16 lanes and compare:

```
needle_first = broadcast_u8x16(needle[0])
mask = cmpeq_u8x16(chunk, needle_first)
```

Produces a bitmask of candidate positions. ~1 in 256 bytes match the first byte on random data, so most chunks hit the fast path (cmpeq + branch-on-zero).

**Verify path:**

When a candidate is found at position `p`, verify `needle[1..needle_len]` byte-by-byte against the decrypted data still in registers. For matches near chunk boundaries, index into the next chunk's register.

**Overlap buffer:**

Between iterations, hold `needle_len - 1` bytes from the end of the previous iteration:

```
overlap[0..needle_len-2] = last (needle_len-1) bytes of previous iteration
```

At the start of each new 256-byte iteration, scan the overlap concatenated with the first bytes of the new decrypted data. Scalar scan over at most 63 bytes — negligible cost.

**Match output:**

```
matches[*match_count] = block_start_offset + chunk_index * 16 + position_in_chunk
*match_count += 1
```

**Needle length constraint:** v1 supports 1-64 bytes. Covers any realistic grep-style fixed string.

## Python Bindings & CLI

### Auto-generated: `chacha20_search.py`

Built by `ea bind chacha20_search.ea --python`. Wraps ctypes call with numpy array allocation.

### CLI wrapper: `eachacha_grep.py`

```
python3 eachacha_grep.py "ERROR" encrypted_logs.bin --key <hex> --nonce <hex>
```

Workflow:
1. mmap the ciphertext file
2. Allocate matches array (pre-sized, e.g., 1M entries)
3. Call `chacha20_search()`
4. Print match offsets

Optional context-line feature: targeted second-pass decrypt of ~80 bytes around each match using existing `chacha20_encrypt()`. Only matched regions decrypted to user-visible memory.

No changes to existing files. The new kernel is purely additive.

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

**Cross-verification tests:**
9. Decrypt with `chacha20_encrypt()`, `str.find()` on plaintext, compare offsets — random data, multiple sizes
10. Same with realistic log-like data ("ERROR" scattered at random positions)

**Size sweep:**
11. Sizes: 0, 1, 15, 16, 63, 64, 65, 127, 128, 255, 256, 257, 1000, 4096, 1MB

Follows existing test style: pytest, numpy, ctypes.

## Benchmark Suite

### New file: `bench_search.py`

Throughput (GB/s) across 64KB → 256MB.

| # | Implementation | What it measures |
|---|----------------|-----------------|
| 1 | Ea fused decrypt+search | The new kernel — single pass, zero plaintext in RAM |
| 2 | Ea decrypt → Python memmem | Two-pass: `chacha20_encrypt()` to buffer, then `plaintext.find()` |
| 3 | Ea decrypt → C memmem | Two-pass: decrypt to buffer, then libc `memmem()` via ctypes |
| 4 | grep on plaintext file | Pre-decrypted file on disk, `subprocess.run(["grep"])` |
| 5 | grep on plaintext in-memory | `memmem` via ctypes on plaintext numpy buffer |

**Test data (v1):** Random bytes with "ERROR" injected at ~1 per 4KB (realistic log density). Same data encrypted for benchmarks 1-3, plaintext for 4-5.

**v2 data:** Real public log dataset (e.g., public access logs, HTTP traffic corpus) for more credible benchmarks.

**Output:** Median GB/s, stddev, table across all sizes. Same format as existing `bench.py`.

**Target headlines:**
- Fused vs two-pass → fusion speedup ratio
- Fused on encrypted vs grep on plaintext → the "wow" number
- Consistent scaling across sizes → credibility

## New Files

| File | Purpose |
|------|---------|
| `chacha20_search.ea` | Fused decrypt+search kernel |
| `chacha20_search.py` | Auto-generated Python bindings |
| `eachacha_grep.py` | CLI demo wrapper |
| `test_search.py` | Test suite |
| `bench_search.py` | Benchmark suite |

## v2 Roadmap (out of scope for v1)

- Multiple fixed strings per call (["ERROR", "FATAL", "PANIC"])
- Case-insensitive search (ASCII case folding in kernel)
- Context-line extraction in kernel (find \n boundaries, copy matched line)
- Real public log dataset for benchmarks
- Parallel multi-core search (ThreadPoolExecutor, split file into chunks)
