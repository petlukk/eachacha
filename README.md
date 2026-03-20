# eachacha

ChaCha20 (RFC 7539) SIMD kernels in the [Eä language](https://github.com/petlukk/eacompute). Encrypt at 1.78 GB/s. Search encrypted data without decrypting to disk.

```
pip install eachacha
```

## Quick Start

```python
from eachacha import encrypt, search

key = bytes(range(32))     # 32-byte key
nonce = bytes(12)          # 12-byte nonce

# Encrypt
ct = encrypt(b"INFO ok\nERROR disk full\nINFO done\n", key, nonce)

# Search encrypted data — plaintext never touches disk
result = search(ct, b"ERROR", key, nonce)
print(result.offsets)  # [8]

# Multi-needle search with context lines (auto-selects v2 kernel)
result = search(ct, [b"ERROR", b"INFO"], key, nonce)
for i, line in enumerate(result.lines):
    print(f"[{result.needle_ids[i]}] {line}")
```

## The Searchable Cipher

Standard process for searching encrypted logs:

```
Read file → Decrypt to /tmp (vulnerability!) → Read /tmp → Search → Delete /tmp
```

The Eä process:

```
Read encrypted file → Decrypt in buffer → Search in buffer → Report match → Zero buffer
```

Plaintext never exists as a full-file allocation. Only a 4 KB window lives in memory at a time, zeroed after each iteration.

### CLI

```bash
# Single needle
eachacha-grep "ERROR" encrypted.bin --key <hex> --nonce <hex>

# Multi-needle with context lines
eachacha-grep "ERROR" "FATAL" "PANIC" encrypted.bin --key <hex> --nonce <hex>
```

### Benchmarks

AMD EPYC 9354P (2 vCPUs), 64 MB:

**Single-needle search (v1):**

| Implementation | GB/s |
|---|---:|
| **Ea fused decrypt+search** | **1.28** |
| Ea decrypt → C memmem (two-pass) | 0.96 |
| C memmem on plaintext | 2.22 |

**Multi-needle search (v2, 3 needles + context lines):**

| Implementation | GB/s |
|---|---:|
| **Ea v2 multi-needle (1 pass)** | **0.52** |
| Ea v1 single-needle x3 (3 passes) | 0.41 |
| C memmem x3 on plaintext | 0.78 |

- v1 fused vs two-pass: **1.34x faster**
- v2 multi-needle vs v1 x3: **1.28x faster** (one decrypt instead of three)

### Security Model

| Property | Guarantee |
|---|---|
| Full-file plaintext buffer | Never created — 4 KB window at a time (v2) |
| Working buffer | Zeroed after each iteration |
| Plaintext on disk | Never written |
| Kernel output | Match offsets + extracted lines only |

### How It Works

**v1 kernel** (`chacha20_search.ea`, 576 lines): Decrypts 256 bytes at a time, searches with `.==` + `movemask` SIMD first-byte filter (same algorithm as glibc memmem: `vpcmpeqb` + `vpmovmskb`), handles cross-block boundaries via overlap buffer.

**v2 kernel** (`chacha20_search_v2.ea`, 866 lines): Decrypts into a 4 KB window, searches for multiple needles by OR:ing `.==` + `movemask` bitmasks per unique first-byte, extracts matched log lines by finding `\n` boundaries with the same SIMD primitives.

## Encrypt + Statistics

The fused kernel encrypts data **and** computes sum/count/min/max in a single pass:

| Implementation | GB/s |
|---|---:|
| Generic C (-O3, no SIMD) | 0.54 |
| OpenSSL ChaCha20 (Python wrapper) | 0.59 |
| **Ea ChaCha20 (single core)** | **1.78** |
| **Ea fused (encrypt + stats)** | **1.43** |
| Ea encrypt + NumPy stats (two passes) | 1.08 |

Fusion adds ~20% overhead vs encrypt-only. The separate approach pays for a second memory traversal — fusion eliminates it.

## Complexity

| Kernel | Lines | Throughput |
|---|---:|---:|
| `chacha20.ea` (encrypt) | 272 | 1.78 GB/s |
| `chacha20_fused.ea` (encrypt+stats) | 384 | 1.43 GB/s |
| `chacha20_search.ea` (v1 search) | 576 | 1.28 GB/s |
| `chacha20_search_v2.ea` (v2 multi-needle) | 866 | 0.52 GB/s |
| **Total** | **2,098** | |

2,098 lines of Eä produce four production-grade kernels. For comparison, OpenSSL's ChaCha20 alone is ~100,000+ lines of C/ASM.

## Build from Source

Requires `ea-compiler` (`pip install ea-compiler`) and a C compiler.

```bash
./build.sh
python3 test_vectors.py && python3 test_fused.py && python3 test_search.py && python3 test_search_v2.py
```

## Files

| File | Purpose |
|---|---|
| `chacha20.ea` | ChaCha20 encrypt kernel (4-block ILP) |
| `chacha20_fused.ea` | Fused encrypt + statistics kernel |
| `chacha20_search.ea` | v1: single-needle fused decrypt+search |
| `chacha20_search_v2.ea` | v2: multi-needle + context-line extraction |
| `eachacha_grep.py` | CLI for searching encrypted files |
| `test_vectors.py` | RFC 7539 test vectors + OpenSSL cross-check (8 tests) |
| `test_fused.py` | Fused encrypt+stats tests (19 tests) |
| `test_search.py` | v1 search tests (17 tests, 38 assertions) |
| `test_search_v2.py` | v2 search tests (27 tests, 44 assertions) |
| `bench.py` | Encrypt benchmark suite |
| `bench_search.py` | v1 search benchmark suite |
| `bench_search_v2.py` | v2 multi-needle benchmark suite |
| `autoresearch/` | Automated kernel optimization loop |
