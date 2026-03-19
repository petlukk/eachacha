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
| NumPy XOR (baseline, not crypto) | 8.271 | 0.432 |
| Generic C (scalar, -O3)      | 0.603 | 0.022 |
| OpenSSL ChaCha20              | 0.548 | 0.027 |
| Ea ChaCha20 (single core)    | 0.605 | 0.006 |
| Ea ChaCha20 parallel (2 cores) | 0.941 | 0.076 |
| Ea fused (encrypt + stats)   | 0.572 | 0.007 |
| Ea encrypt + NumPy stats (separate) | 0.474 | 0.021 |

### Analysis

**Where Ea wins:**

- **Ea matches generic C** on single-core throughput (0.605 vs 0.603 GB/s),
  showing the Ea compiler generates competitive scalar code from a higher-level
  language.
- **Parallel scaling works.** The 2-core parallel variant reaches 0.941 GB/s
  — a 1.56x speedup over single-core — with zero code changes to the kernel
  itself, just Python-side partitioning.
- **Ea beats OpenSSL** in this configuration (0.605 vs 0.548 GB/s). OpenSSL's
  ChaCha20 is going through the Python `cryptography` library's object
  allocation overhead on each call, which penalises it in this benchmark
  structure.

**Where Ea loses:**

- **NumPy XOR is 14x faster** — but it is doing a trivial XOR with no quarter
  rounds, no key schedule, no counter management. It just shows memory bandwidth
  ceiling (~8 GB/s on this machine).
- **OpenSSL in production** (called from C, with AVX2/AVX-512 codepaths, and
  amortised setup) would likely be 3-5x faster than what we see here.
  The `cryptography` Python wrapper adds per-call overhead that hides OpenSSL's
  real throughput.

### The fusion argument

The fused kernel (`chacha20_encrypt_stats`) computes encryption **and** four
statistics (sum, count, min, max) of the plaintext in a single pass. Compare:

| Approach | GB/s |
|---|---:|
| Ea encrypt + NumPy stats (two passes) | 0.474 |
| Ea fused (one pass) | 0.572 |

Fused is **21% faster** than doing encrypt followed by separate NumPy
reductions. The plaintext data is already in registers/cache during encryption;
computing statistics there avoids a second 64 MB memory traversal. On larger
data or on machines where memory bandwidth is the bottleneck, the fusion
advantage grows — you get the statistics essentially for free.

This is the core value proposition of the Ea language for data pipelines:
write fused kernels that do multiple operations per byte loaded from memory,
rather than chaining separate library calls that each re-read the same data.

## Files

| File | Purpose |
|---|---|
| `chacha20.ea` | Ea ChaCha20 block + encrypt kernel |
| `chacha20_fused.ea` | Ea fused encrypt + statistics kernel |
| `chacha20_ref.c` | Generic C reference (no SIMD) |
| `chacha20.py` | Python bindings for `chacha20.so` |
| `chacha20_fused.py` | Python bindings for `chacha20_fused.so` |
| `test_vectors.py` | RFC 7539 test vectors |
| `test_fused.py` | Fused kernel verification |
| `bench.py` | Benchmark suite |
| `build.sh` | Build script |
