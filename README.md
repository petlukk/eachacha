# eachacha

ChaCha20 (RFC 7539) implemented in the Ea SIMD language, benchmarked against
generic C, OpenSSL, and NumPy. Includes a fused encrypt+statistics kernel that
computes sum, count, min, and max of the plaintext in the same pass as
encryption — demonstrating that operator fusion eliminates a second memory
traversal and delivers meaningful extra work for free.

## Build

Requires the `ea` compiler (`eacompute`) and a C compiler.

```bash
./build.sh          # builds chacha20.so, chacha20_fused.so, libchacha20_ref.so
```

## Verify

```bash
python3 test_vectors.py   # RFC 7539 test vectors + OpenSSL cross-check
python3 test_fused.py     # fused kernel correctness (19 tests)
```

## Benchmark

```bash
python3 bench.py
```

### Results

Measured on AMD EPYC 9354P (2 vCPUs), 64 MB data, median of 10 runs:

| Implementation                | GB/s  | stddev |
|-------------------------------|------:|-------:|
| NumPy XOR (baseline, not crypto) | 6.962 | 0.589 |
| Generic C (scalar, -O3)      | 0.540 | 0.014 |
| OpenSSL ChaCha20              | 0.587 | 0.019 |
| **Ea ChaCha20 (single core)**    | **1.687** | 0.061 |
| Ea ChaCha20 parallel (2 cores) | 1.563 | 0.176 |
| Ea fused (encrypt + stats)   | 0.576 | 0.011 |
| Ea encrypt + NumPy stats (separate) | 1.004 | 0.057 |

### Autoresearch optimization

The encrypt kernel was optimized by an automated search process (`autoresearch`)
that explored variants of the ChaCha20 inner loop. The key optimizations found:

- **4-block interleaved ILP.** Process four ChaCha20 blocks simultaneously,
  interleaving their quarter-round operations so the CPU's out-of-order engine
  can overlap independent dependency chains across blocks.
- **Direct i32x4 XOR.** XOR plaintext with keystream as `i32x4` vectors
  directly — no intermediate keystream buffer for full blocks. The `pt_i32` /
  `ct_i32` pointer pair gives the kernel aligned i32 access to the same memory
  as the u8 pointers, avoiding type-punning overhead.
- **`*restrict` pointers.** All pointer parameters carry `restrict`, letting
  LLVM assume no aliasing and optimize load/store scheduling.
- **Hoisted nonce reads.** The three nonce words are loaded once before the
  main loop and reused for every block, avoiding redundant memory accesses.
- **Single-block fallback.** After the 4-block loop, remaining full 64-byte
  blocks are processed one at a time. A final partial-block tail uses
  `load_masked` / `store_masked` for sub-16-byte remainders.

The result: **2.8x faster** single-core throughput (0.605 -> 1.687 GB/s),
in 272 lines of Ea. The kernel now **outperforms OpenSSL** (called via Python)
by 2.9x and **outperforms generic C (-O3)** by 3.1x.

### Analysis

**Where Ea wins:**

- **Ea is 3.1x faster than generic C** on single-core throughput (1.687 vs
  0.540 GB/s), showing the Ea compiler's SIMD code generation delivers real
  performance gains over auto-vectorization.
- **Ea is 2.9x faster than OpenSSL** in this configuration (1.687 vs 0.587
  GB/s). OpenSSL's ChaCha20 is going through the Python `cryptography`
  library's object allocation overhead on each call, which penalises it in
  this benchmark structure.

**Where Ea loses:**

- **NumPy XOR is 4.1x faster** — but it is doing a trivial XOR with no quarter
  rounds, no key schedule, no counter management. It just shows memory bandwidth
  ceiling (~7 GB/s on this machine).
- **OpenSSL in production** (called from C, with AVX2/AVX-512 codepaths, and
  amortised setup) would likely be 3-5x faster than what we see here.
  The `cryptography` Python wrapper adds per-call overhead that hides OpenSSL's
  real throughput.

### The fusion argument

The fused kernel (`chacha20_encrypt_stats`) computes encryption **and** four
statistics (sum, count, min, max) of the plaintext in a single pass. The fused
kernel has not yet been updated with the 4-block ILP optimization. Compare the
current state:

| Approach | GB/s |
|---|---:|
| Ea encrypt + NumPy stats (two passes) | 1.004 |
| Ea fused (one pass, not yet optimized) | 0.576 |

With the optimized encrypt kernel, the two-pass approach is now faster than
the unoptimized fused kernel. Once the fused kernel receives the same 4-block
ILP treatment, it should regain its advantage — the fusion principle still
holds: statistics computed during encryption avoid a second memory traversal.

## Complexity

272 lines of Ea produce a ChaCha20 implementation that:

- Passes all RFC 7539 test vectors
- Cross-verifies with OpenSSL byte-for-byte
- Achieves 1.7 GB/s single-core on a 2-vCPU cloud VM
- Supports arbitrary input lengths (not just block-aligned)

## Files

| File | Purpose |
|---|---|
| `chacha20.ea` | Ea ChaCha20 block + encrypt kernel (4-block ILP) |
| `chacha20_fused.ea` | Ea fused encrypt + statistics kernel |
| `chacha20_ref.c` | Generic C reference (no SIMD) |
| `chacha20.py` | Python bindings for `chacha20.so` |
| `chacha20_fused.py` | Python bindings for `chacha20_fused.so` |
| `test_vectors.py` | RFC 7539 test vectors |
| `test_fused.py` | Fused kernel verification |
| `bench.py` | Benchmark suite |
| `build.sh` | Build script |
