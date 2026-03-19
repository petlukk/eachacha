"""Benchmark suite for eachacha — ChaCha20 implementations compared."""
import ctypes as ct
import numpy as np
import os
import sys
import time
import platform
import statistics
import resource
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Ensure sufficient stack for SIMD kernels processing large buffers
_soft, _hard = resource.getrlimit(resource.RLIMIT_STACK)
_target = 64 * 1024 * 1024
if _soft != resource.RLIM_INFINITY and _soft < _target:
    _new = _target if _hard == resource.RLIM_INFINITY else min(_target, _hard)
    resource.setrlimit(resource.RLIMIT_STACK, (_new, _hard))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_SIZE = 64 * 1024 * 1024  # 64 MB
WARMUP = 3
TIMED = 10
NCORES = os.cpu_count() or 1

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def print_sysinfo():
    print("=" * 62)
    print("System info")
    print("=" * 62)
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
    print(f"  Cores     : {NCORES}")
    print(f"  Platform  : {platform.platform()}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  NumPy     : {np.__version__}")
    print(f"  Data size : {DATA_SIZE / 1024 / 1024:.0f} MB")
    print(f"  Warmup    : {WARMUP}   Timed: {TIMED}")
    print()

# ---------------------------------------------------------------------------
# Load shared libraries
# ---------------------------------------------------------------------------
_ea = ct.CDLL(str(_HERE / "chacha20.so"))
_ea.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
]
_ea.chacha20_encrypt.restype = None

_fused = ct.CDLL(str(_HERE / "chacha20_fused.so"))
_fused.chacha20_encrypt_stats.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int64), ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8),
]
_fused.chacha20_encrypt_stats.restype = None

_ref = ct.CDLL(str(_HERE / "libchacha20_ref.so"))
_ref.chacha20_encrypt_ref.argtypes = [
    ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint32), ct.c_uint32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
]
_ref.chacha20_encrypt_ref.restype = None

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
KEY_U32 = [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
           0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c]

NONCE_U32 = [0x00000000, 0x4a000000, 0x00000000]

def to_i32_array(values):
    arr = (ct.c_int32 * len(values))()
    for i, v in enumerate(values):
        arr[i] = ct.c_int32(v & 0xFFFFFFFF).value
    return arr

def make_scratch():
    scratch = (ct.c_uint8 * 64)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    return scratch, ks_i32, ks_u8

# Pre-allocate buffers
plaintext = np.random.randint(0, 256, size=DATA_SIZE, dtype=np.uint8)
ciphertext = np.empty(DATA_SIZE, dtype=np.uint8)

key_arr = to_i32_array(KEY_U32)
nonce_arr = to_i32_array(NONCE_U32)

# C ref key/nonce
ref_key = (ct.c_uint32 * 8)(*KEY_U32)
ref_nonce = (ct.c_uint32 * 3)(*NONCE_U32)

# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench(name, fn, warmup=WARMUP, timed=TIMED):
    """Run fn() warmup+timed times, report median GB/s and stddev."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(timed):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    gbps = [DATA_SIZE / t / 1e9 for t in times]
    med = statistics.median(gbps)
    sd = statistics.stdev(gbps) if len(gbps) > 1 else 0.0
    print(f"  {name:<42s}  {med:8.3f} GB/s  (sd {sd:.3f})")
    return med, sd

# ---------------------------------------------------------------------------
# 1. NumPy XOR baseline (not real crypto)
# ---------------------------------------------------------------------------
def bench_numpy_xor():
    key_repeated = np.tile(plaintext[:32], DATA_SIZE // 32 + 1)[:DATA_SIZE]
    out = np.empty(DATA_SIZE, dtype=np.uint8)
    def fn():
        np.bitwise_xor(plaintext, key_repeated, out=out)
    return bench("NumPy XOR (baseline, not crypto)", fn)

# ---------------------------------------------------------------------------
# 2. Generic C (-O3, no SIMD)
# ---------------------------------------------------------------------------
def bench_c_ref():
    pt_ptr = plaintext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_ptr = ciphertext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    def fn():
        _ref.chacha20_encrypt_ref(ref_key, ref_nonce, ct.c_uint32(1),
                                  pt_ptr, ct_ptr, ct.c_int32(DATA_SIZE))
    return bench("Generic C (scalar, -O3)", fn)

# ---------------------------------------------------------------------------
# 3. OpenSSL ChaCha20
# ---------------------------------------------------------------------------
def bench_openssl():
    import struct
    from cryptography.hazmat.primitives.ciphers import Cipher
    from cryptography.hazmat.primitives.ciphers.algorithms import ChaCha20

    key_bytes = b"".join(struct.pack("<I", w) for w in KEY_U32)
    nonce_12 = b"".join(struct.pack("<I", w) for w in NONCE_U32)
    counter_bytes = struct.pack("<I", 1)
    full_nonce = counter_bytes + nonce_12
    pt_bytes = bytes(plaintext)

    def fn():
        cipher = Cipher(ChaCha20(key_bytes, full_nonce), mode=None)
        enc = cipher.encryptor()
        enc.update(pt_bytes)
        enc.finalize()
    return bench("OpenSSL ChaCha20", fn)

# ---------------------------------------------------------------------------
# 4. Ea ChaCha20 (single core)
# ---------------------------------------------------------------------------
def bench_ea_single():
    scratch, ks_i32, ks_u8 = make_scratch()
    pt_ptr = plaintext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_ptr = ciphertext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pt_i32 = ct.cast(pt_ptr, ct.POINTER(ct.c_int32))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    def fn():
        _ea.chacha20_encrypt(key_arr, nonce_arr, ct.c_int32(1),
                             pt_ptr, ct_ptr, ct.c_int32(DATA_SIZE),
                             ks_i32, ks_u8, pt_i32, ct_i32)
    return bench("Ea ChaCha20 (single core)", fn)

# ---------------------------------------------------------------------------
# 5. Ea ChaCha20 parallel (N cores)
# ---------------------------------------------------------------------------
def bench_ea_parallel():
    total_blocks = DATA_SIZE // 64
    blocks_per_thread = total_blocks // NCORES

    def work(tid):
        start_block = tid * blocks_per_thread
        if tid == NCORES - 1:
            n_blocks = total_blocks - start_block
        else:
            n_blocks = blocks_per_thread
        byte_offset = start_block * 64
        byte_len = n_blocks * 64
        # Own scratch buffer per thread
        tsc = (ct.c_uint8 * 64)()
        ti32 = ct.cast(tsc, ct.POINTER(ct.c_int32))
        tu8 = ct.cast(tsc, ct.POINTER(ct.c_uint8))
        src = ct.cast(ct.addressof(ct.c_uint8.from_buffer(plaintext, byte_offset)),
                      ct.POINTER(ct.c_uint8))
        dst = ct.cast(ct.addressof(ct.c_uint8.from_buffer(ciphertext, byte_offset)),
                      ct.POINTER(ct.c_uint8))
        src_i32 = ct.cast(src, ct.POINTER(ct.c_int32))
        dst_i32 = ct.cast(dst, ct.POINTER(ct.c_int32))
        _ea.chacha20_encrypt(key_arr, nonce_arr, ct.c_int32(start_block),
                             src, dst, ct.c_int32(byte_len), ti32, tu8,
                             src_i32, dst_i32)

    def fn():
        threading.stack_size(64 * 1024 * 1024)
        with ThreadPoolExecutor(max_workers=NCORES) as pool:
            futures = [pool.submit(work, tid) for tid in range(NCORES)]
            for f in futures:
                f.result()

    return bench(f"Ea ChaCha20 parallel ({NCORES} cores)", fn)

# ---------------------------------------------------------------------------
# 6. Ea fused (encrypt + stats, single pass)
# ---------------------------------------------------------------------------
def bench_ea_fused():
    scratch, ks_i32, ks_u8 = make_scratch()
    pt_ptr = plaintext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_ptr = ciphertext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    out_sum = (ct.c_int64 * 1)()
    out_count = (ct.c_int32 * 1)()
    out_min = (ct.c_uint8 * 1)()
    out_max = (ct.c_uint8 * 1)()

    p_sum = ct.cast(out_sum, ct.POINTER(ct.c_int64))
    p_count = ct.cast(out_count, ct.POINTER(ct.c_int32))
    p_min = ct.cast(out_min, ct.POINTER(ct.c_uint8))
    p_max = ct.cast(out_max, ct.POINTER(ct.c_uint8))

    def fn():
        _fused.chacha20_encrypt_stats(
            key_arr, nonce_arr, ct.c_int32(1),
            pt_ptr, ct_ptr, ct.c_int32(DATA_SIZE),
            ks_i32, ks_u8,
            p_sum, p_count, p_min, p_max)
    return bench("Ea fused (encrypt + stats)", fn)

# ---------------------------------------------------------------------------
# 7. Separate encrypt + numpy stats
# ---------------------------------------------------------------------------
def bench_ea_plus_numpy_stats():
    scratch, ks_i32, ks_u8 = make_scratch()
    pt_ptr = plaintext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_ptr = ciphertext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pt_i32 = ct.cast(pt_ptr, ct.POINTER(ct.c_int32))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    def fn():
        _ea.chacha20_encrypt(key_arr, nonce_arr, ct.c_int32(1),
                             pt_ptr, ct_ptr, ct.c_int32(DATA_SIZE),
                             ks_i32, ks_u8, pt_i32, ct_i32)
        np.sum(plaintext)
        np.min(plaintext)
        np.max(plaintext)
    return bench("Ea encrypt + NumPy stats (separate)", fn)

# ===========================================================================
# Main
# ===========================================================================
def main():
    print_sysinfo()

    print("=" * 62)
    print("Benchmarks  (64 MB, median of 10 runs)")
    print("=" * 62)

    results = []
    results.append(("NumPy XOR (baseline)",)           + bench_numpy_xor())
    results.append(("Generic C (scalar, -O3)",)        + bench_c_ref())
    results.append(("OpenSSL ChaCha20",)               + bench_openssl())
    results.append(("Ea single core",)                 + bench_ea_single())
    results.append((f"Ea parallel ({NCORES} cores)",)  + bench_ea_parallel())
    results.append(("Ea fused (encrypt+stats)",)       + bench_ea_fused())
    results.append(("Ea encrypt + NumPy stats",)       + bench_ea_plus_numpy_stats())

    print()
    print("=" * 62)
    print(f"{'Implementation':<42s}  {'GB/s':>8s}  {'stddev':>8s}")
    print("-" * 62)
    for name, med, sd in results:
        print(f"  {name:<40s}  {med:8.3f}  {sd:8.3f}")
    print("=" * 62)

if __name__ == "__main__":
    main()
