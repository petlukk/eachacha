# eachacha

ChaCha20 (RFC 7539) implemented in the Ea SIMD language, benchmarked against
generic C, OpenSSL, and NumPy. Features two fusion demos:

1. **Fused encrypt+statistics** — encrypts data and computes sum/count/min/max
   of the plaintext in one pass. Stats come for free.
2. **Fused decrypt+search ("The Searchable Cipher")** — searches encrypted data
   for a string pattern without ever decrypting to disk. Plaintext exists only
   in a 256-byte working buffer, zeroed after each iteration.

## Build

Requires the `ea` compiler (`pip install ea-compiler`) and a C compiler.

```bash
./build.sh          # builds chacha20.so, chacha20_fused.so, chacha20_search.so, libchacha20_ref.so
```

## Verify

```bash
python3 test_vectors.py   # RFC 7539 test vectors + OpenSSL cross-check (8 tests)
python3 test_fused.py     # fused encrypt+stats correctness (19 tests)
python3 test_search.py    # fused decrypt+search correctness (17 tests, 38 assertions)
```

## The Searchable Cipher

The standard process for searching encrypted logs:

```
Read file → Decrypt to /tmp (vulnerability!) → Read /tmp → Search → Delete /tmp
```

Result: lots of I/O, high risk, slow.

The Ea process:

```
Read encrypted file → Decrypt in buffer → Search in buffer → Report match → Zero buffer
```

Result: minimal I/O, bounded plaintext exposure (256 bytes at a time), 58% of plaintext search speed.

### Usage

```bash
python3 eachacha_grep.py "ERROR" encrypted_logs.bin \
    --key 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f \
    --nonce 000000004a00000000000000 \
    --context
```

Output:
```
Found 1 match in 61 bytes
  offset 21
    ...INFO normal log line.>>>ERROR<<< something broke.INFO another line....
```

### Search benchmark

```bash
python3 bench_search.py
```

Measured on AMD EPYC 9354P (2 vCPUs), 64 MB data with "ERROR" injected every ~4 KB:

| Implementation | GB/s |
|---|---:|
| **Ea fused decrypt+search** | **1.28** |
| Ea decrypt → C memmem (two-pass) | 0.96 |
| Ea decrypt → Python find (two-pass) | 0.65 |
| C memmem on plaintext (in-memory) | 2.22 |

- Fused vs two-pass decrypt+memmem: **1.34x faster**
- Fused vs plaintext memmem: **58%** of plaintext speed — on encrypted data, with zero full-file exposure

### Security model

| Property | Guarantee |
|---|---|
| Full-file plaintext buffer | Never created — only 256 bytes live at a time |
| Working buffer | Zeroed after each iteration |
| Plaintext on disk | Never written |
| Kernel output | Only match byte offsets — no plaintext |
| Exposure surface | 256 bytes vs 100 GB = 400-million-fold reduction |

### How it works

The kernel (`chacha20_search.ea`) fuses ChaCha20 decryption with SIMD string
search in a single streaming pass:

1. **Decrypt** 256 bytes of ciphertext into a working buffer (4-block ILP, same
   optimization as the encrypt kernel)
2. **Search** using `.==` + `movemask` first-byte filter (same algorithm as glibc
   memmem / GNU grep: `vpcmpeqb` + `vpmovmskb`), scalar verify on candidates
3. **Handle boundaries** via overlap buffer — last `needle_len - 1` bytes carry
   between iterations so matches spanning block boundaries are never missed
4. **Zero** the working buffer and move to the next 256 bytes

## Encrypt + Statistics benchmark

```bash
python3 bench.py
```

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
- **`*restrict` pointers.** Pointer parameters carry `restrict` where safe,
  letting LLVM assume no aliasing and optimize load/store scheduling. (Note:
  aliasing pointers like `pt_buf`/`pt_i32` must NOT use `restrict` — this
  caused LLVM to eliminate stores in the search kernel.)
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
kernels use the same 4-block ILP optimization. The fused kernel adds only ~20%
overhead compared to encrypt-only (1.43 vs 1.78 GB/s), while providing
sum, count, min, and max for free.

This is the core value proposition: OpenSSL is a black box. You send data in, get
ciphertext out, then do a second pass for analytics. With Ea, you write one kernel
that does both. One memory read instead of two. The searchable cipher takes this
further — you don't even need to decrypt to search.

## Complexity

| Implementation | Lines of code | Encrypt | Encrypt + stats | Decrypt + search |
|---|---:|---:|---:|---:|
| OpenSSL + tools | ~100,000+ | 0.59 GB/s* | N/A | N/A |
| Generic C + grep | 45 + grep | 0.54 GB/s | N/A | N/A |
| **Ea encrypt** | **272** | **1.78 GB/s** | — | — |
| **Ea fused stats** | **384** | — | **1.43 GB/s** | — |
| **Ea fused search** | **~480** | — | — | **1.28 GB/s** |

*OpenSSL through Python wrapper; native would be faster.

~480 lines of Ea produce a searchable cipher that:

- Passes all RFC 7539 test vectors
- Cross-verifies with OpenSSL byte-for-byte
- Searches encrypted data at 1.28 GB/s single-core
- Never writes plaintext to disk or allocates a full-file buffer
- Handles cross-block boundary matches correctly
- Scales consistently from 64 KB to 256 MB

## Files

| File | Purpose |
|---|---|
| `chacha20.ea` | Ea ChaCha20 block + encrypt kernel (4-block ILP) |
| `chacha20_fused.ea` | Ea fused encrypt + statistics kernel |
| `chacha20_search.ea` | Ea fused decrypt + search kernel (SIMD fast-skip) |
| `chacha20_ref.c` | Generic C reference (no SIMD) |
| `eachacha_grep.py` | CLI: search encrypted files without decrypting to disk |
| `test_vectors.py` | RFC 7539 test vectors + OpenSSL cross-check |
| `test_fused.py` | Fused encrypt+stats verification |
| `test_search.py` | Fused decrypt+search verification (17 tests) |
| `bench.py` | Encrypt + stats benchmark suite |
| `bench_search.py` | Searchable cipher benchmark suite |
| `build.sh` | Build script |
