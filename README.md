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

Measured on AMD EPYC 9354P (2 vCPUs). Median throughput across data sizes:

| Size | Encrypt | Fused (encrypt+stats) | Separate (encrypt + numpy) | Fusion speedup |
|-----:|--------:|----------------------:|---------------------------:|---------------:|
| 64 KB | 1.63 GB/s | 1.28 GB/s | 0.98 GB/s | 1.31x |
| 1 MB | 1.78 GB/s | 1.48 GB/s | 1.14 GB/s | 1.30x |
| 16 MB | 1.81 GB/s | 1.43 GB/s | 1.00 GB/s | 1.42x |
| 64 MB | 1.76 GB/s | 1.43 GB/s | 1.08 GB/s | 1.33x |
| 256 MB | 1.78 GB/s | 1.38 GB/s | 1.05 GB/s | 1.31x |

For context (64 MB, median of 10 runs):

| Implementation | GB/s |
|---|---:|
| NumPy XOR (not real crypto) | ~7.0 |
| Generic C (-O3, no SIMD) | 0.54 |
| OpenSSL ChaCha20 (Python wrapper) | 0.59 |
| **Ea ChaCha20 (single core)** | **1.78** |
| **Ea fused (encrypt + stats, one pass)** | **1.43** |
| Ea encrypt + NumPy stats (two passes) | 1.08 |

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

The result: **2.9x faster** single-core throughput (0.61 -> 1.78 GB/s),
in 272 lines of Ea. The kernel **outperforms generic C (-O3)** by 3.3x.

### Analysis

**Where Ea wins:**

- **3.3x faster than generic C** (1.78 vs 0.54 GB/s). The Ea compiler's SIMD
  code generation delivers real performance gains over what `cc -O3` auto-vectorizes.
- **3.0x faster than OpenSSL's Python wrapper** (1.78 vs 0.59 GB/s). Though this
  comparison is unfair to OpenSSL — see below.
- **Fusion works at every scale.** The fused kernel delivers 1.3-1.4x speedup
  over separate passes consistently from 64 KB to 256 MB. This is not a cache
  artifact. Statistics come essentially free when computed during encryption.

**Where Ea loses:**

- **OpenSSL in production** (called from C, with AVX2/AVX-512 codepaths, and
  amortised setup) would likely be 3-5x faster than what we see here. The Python
  `cryptography` wrapper adds per-call overhead that hides OpenSSL's real throughput.
  We are honest about this.
- **NumPy XOR is ~4x faster** — but it does a trivial XOR with no rounds, no key
  schedule. It just shows memory bandwidth ceiling (~7 GB/s on this machine).

### The fusion argument

The fused kernel (`chacha20_encrypt_stats`) encrypts data **and** computes four
statistics (sum, count, min, max) of the plaintext in a single memory pass. Both
kernels use the same 4-block ILP optimization. Results across real-world data sizes:

| Size | Fused (one pass) | Separate (two passes) | Fusion speedup |
|-----:|------------------:|----------------------:|---------------:|
| 64 KB | 1.28 GB/s | 0.98 GB/s | 1.31x |
| 1 MB | 1.48 GB/s | 1.14 GB/s | 1.30x |
| 16 MB | 1.43 GB/s | 1.00 GB/s | 1.42x |
| 64 MB | 1.43 GB/s | 1.08 GB/s | 1.33x |
| 256 MB | 1.38 GB/s | 1.05 GB/s | 1.31x |

The fused kernel adds only ~20% overhead compared to encrypt-only (1.43 vs 1.78 GB/s),
while providing sum, count, min, and max for free. The separate approach pays for
a second full memory traversal via NumPy — that second pass is what fusion eliminates.

This is the core value proposition: OpenSSL is a black box. You send data in, get
ciphertext out, then do a second pass for analytics. With Ea, you write one kernel
that does both. One memory read instead of two.

## Complexity

| Implementation | Lines of code | Throughput |
|---|---:|---:|
| OpenSSL | ~100,000+ (C/ASM) | 0.59 GB/s* |
| Generic C | 45 | 0.54 GB/s |
| **Ea** | **272** | **1.78 GB/s** |
| **Ea fused** | **284** | **1.43 GB/s** (+ stats) |

*OpenSSL through Python wrapper; native would be faster.

272 lines of Ea produce a ChaCha20 implementation that:

- Passes all RFC 7539 test vectors
- Cross-verifies with OpenSSL byte-for-byte
- Achieves 1.78 GB/s single-core on a 2-vCPU cloud VM
- Scales consistently from 64 KB to 256 MB
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
