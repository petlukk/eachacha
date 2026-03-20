"""Benchmark suite for searchable ChaCha20 cipher — 5 implementations compared."""
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
    print("=" * 66)
    print("System info")
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

_search = ct.CDLL(str(_HERE / "chacha20_search.so"))
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

_libc = ct.CDLL(ctypes.util.find_library("c"))
_libc.memmem.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
_libc.memmem.restype = ct.c_void_p

# ---------------------------------------------------------------------------
# Key / nonce constants (same as test files)
# ---------------------------------------------------------------------------
KEY_U32 = [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
           0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c]
NONCE_U32 = [0x00000000, 0x4a000000, 0x00000000]
COUNTER = 1

def to_i32_array(values):
    arr = (ct.c_int32 * len(values))()
    for i, v in enumerate(values):
        arr[i] = ct.c_int32(v & 0xFFFFFFFF).value
    return arr

# ---------------------------------------------------------------------------
# Test data: random bytes with "ERROR" injected every ~4096 bytes
# ---------------------------------------------------------------------------
NEEDLE = b"ERROR"
NEEDLE_BYTES = np.frombuffer(NEEDLE, dtype=np.uint8)

rng = np.random.RandomState(42)
plaintext = rng.randint(0, 256, size=DATA_SIZE, dtype=np.uint8)

# Inject "ERROR" every ~4096 bytes
_inject_positions = range(0, DATA_SIZE - len(NEEDLE), 4096)
for _pos in _inject_positions:
    plaintext[_pos : _pos + len(NEEDLE)] = np.frombuffer(NEEDLE, dtype=np.uint8)

# Pre-encrypt the plaintext once for benchmarks 1-3
ciphertext = np.empty(DATA_SIZE, dtype=np.uint8)

# Pre-allocate key/nonce arrays OUTSIDE timed closures
key_arr = to_i32_array(KEY_U32)
nonce_arr = to_i32_array(NONCE_U32)

def _do_encrypt(pt, ct_out):
    """Encrypt pt into ct_out using chacha20_encrypt."""
    scratch = (ct.c_uint8 * 64)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    pt_ptr = pt.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_ptr = ct_out.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pt_i32 = ct.cast(pt_ptr, ct.POINTER(ct.c_int32))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    _ea.chacha20_encrypt(key_arr, nonce_arr, ct.c_int32(COUNTER),
                         pt_ptr, ct_ptr, ct.c_int32(DATA_SIZE),
                         ks_i32, ks_u8, pt_i32, ct_i32)

# Encrypt once to populate ciphertext array
_do_encrypt(plaintext, ciphertext)

# Also encrypt the needle
_needle_pt = np.frombuffer(NEEDLE, dtype=np.uint8)
_needle_ct_arr = (ct.c_uint8 * len(NEEDLE))(*NEEDLE)

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
    print(f"  {name:<46s}  {med:8.3f} GB/s  (sd {sd:.3f})")
    return med, sd

# ---------------------------------------------------------------------------
# 1. Ea fused decrypt+search (chacha20_search on ciphertext)
# ---------------------------------------------------------------------------
def bench_fused_search():
    # Pre-allocate all buffers outside the timed closure
    ks_scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks_scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks_scratch, ct.POINTER(ct.c_uint8))

    pt_buf_arr = (ct.c_uint8 * 256)()
    pt_i32 = ct.cast(pt_buf_arr, ct.POINTER(ct.c_int32))

    overlap_arr = (ct.c_uint8 * 64)()
    overlap_ptr = ct.cast(overlap_arr, ct.POINTER(ct.c_uint8))

    matches_arr = (ct.c_int32 * 100000)()
    match_count_arr = (ct.c_int32 * 1)()

    ct_ptr = ciphertext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    needle_ptr = ct.cast(_needle_ct_arr, ct.POINTER(ct.c_uint8))
    pt_buf_ptr = ct.cast(pt_buf_arr, ct.POINTER(ct.c_uint8))

    def fn():
        match_count_arr[0] = 0
        _search.chacha20_search(
            key_arr, nonce_arr, ct.c_int32(COUNTER),
            ct_ptr, ct.c_int32(DATA_SIZE),
            needle_ptr, ct.c_int32(len(NEEDLE)),
            ks_i32, ks_u8,
            ct_i32,
            pt_buf_ptr, pt_i32,
            overlap_ptr,
            matches_arr, ct.c_int32(100000),
            match_count_arr,
        )

    return bench("Ea fused decrypt+search", fn)

# ---------------------------------------------------------------------------
# 2. Ea decrypt → Python find
# ---------------------------------------------------------------------------
def bench_ea_decrypt_python_find():
    scratch = (ct.c_uint8 * 64)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))

    ct_ptr = ciphertext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pt_out = np.empty(DATA_SIZE, dtype=np.uint8)
    pt_ptr = pt_out.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    pt_i32 = ct.cast(pt_ptr, ct.POINTER(ct.c_int32))

    def fn():
        _ea.chacha20_encrypt(key_arr, nonce_arr, ct.c_int32(COUNTER),
                             ct_ptr, pt_ptr, ct.c_int32(DATA_SIZE),
                             ks_i32, ks_u8, ct_i32, pt_i32)
        pt_bytes = pt_out.tobytes()
        pos = 0
        count = 0
        while True:
            idx = pt_bytes.find(NEEDLE, pos)
            if idx == -1:
                break
            count += 1
            pos = idx + 1

    return bench("Ea decrypt → Python find", fn)

# ---------------------------------------------------------------------------
# 3. Ea decrypt → C memmem
# ---------------------------------------------------------------------------
def bench_ea_decrypt_memmem():
    scratch = (ct.c_uint8 * 64)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))

    ct_ptr = ciphertext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pt_out = np.empty(DATA_SIZE, dtype=np.uint8)
    pt_ptr = pt_out.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    pt_i32 = ct.cast(pt_ptr, ct.POINTER(ct.c_int32))

    needle_c = ct.c_char_p(NEEDLE)
    needle_len = ct.c_size_t(len(NEEDLE))
    data_len = ct.c_size_t(DATA_SIZE)

    def fn():
        _ea.chacha20_encrypt(key_arr, nonce_arr, ct.c_int32(COUNTER),
                             ct_ptr, pt_ptr, ct.c_int32(DATA_SIZE),
                             ks_i32, ks_u8, ct_i32, pt_i32)
        base = pt_out.ctypes.data
        remaining = DATA_SIZE
        ptr = base
        count = 0
        while remaining >= len(NEEDLE):
            result = _libc.memmem(ptr, ct.c_size_t(remaining), needle_c, needle_len)
            if result is None:
                break
            offset = result - base
            count += 1
            advance = offset + 1
            ptr = base + advance
            remaining = DATA_SIZE - advance

    return bench("Ea decrypt → C memmem", fn)

# ---------------------------------------------------------------------------
# 4. grep on plaintext file
# ---------------------------------------------------------------------------
def bench_grep_plaintext():
    # Write plaintext to a temp file once (outside timed loop)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp.write(plaintext.tobytes())
    tmp.flush()
    tmp.close()
    tmppath = tmp.name

    needle_str = NEEDLE.decode("ascii")

    def fn():
        subprocess.run(
            ["grep", "-c", needle_str, tmppath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        result = bench("grep on plaintext file", fn)
    finally:
        os.unlink(tmppath)

    return result

# ---------------------------------------------------------------------------
# 5. C memmem on plaintext (in-memory)
# ---------------------------------------------------------------------------
def bench_memmem_plaintext():
    pt_ptr_base = plaintext.ctypes.data
    needle_c = ct.c_char_p(NEEDLE)
    needle_len = ct.c_size_t(len(NEEDLE))

    def fn():
        base = pt_ptr_base
        remaining = DATA_SIZE
        ptr = base
        count = 0
        while remaining >= len(NEEDLE):
            result = _libc.memmem(ptr, ct.c_size_t(remaining), needle_c, needle_len)
            if result is None:
                break
            offset = result - base
            count += 1
            advance = offset + 1
            ptr = base + advance
            remaining = DATA_SIZE - advance

    return bench("C memmem on plaintext (in-memory)", fn)

# ===========================================================================
# Main
# ===========================================================================
def main():
    print_sysinfo()

    print("=" * 66)
    print("Benchmarks  (64 MB, median of 10 runs)")
    print("=" * 66)

    results = []
    results.append(("Ea fused decrypt+search",)           + bench_fused_search())
    results.append(("Ea decrypt → Python find",)          + bench_ea_decrypt_python_find())
    results.append(("Ea decrypt → C memmem",)             + bench_ea_decrypt_memmem())
    results.append(("grep on plaintext file",)            + bench_grep_plaintext())
    results.append(("C memmem on plaintext (in-memory)",) + bench_memmem_plaintext())

    print()
    print("=" * 66)
    print(f"{'Implementation':<46s}  {'GB/s':>8s}  {'stddev':>8s}")
    print("-" * 66)
    for name, med, sd in results:
        print(f"  {name:<44s}  {med:8.3f}  {sd:8.3f}")
    print("=" * 66)

    # Ratios
    fused_gbps    = results[0][1]
    decrypt_memmem_gbps  = results[2][1]
    plaintext_memmem_gbps = results[4][1]

    print()
    print("Ratios")
    print("-" * 66)
    if decrypt_memmem_gbps > 0:
        ratio1 = fused_gbps / decrypt_memmem_gbps
        print(f"  Fused vs decrypt+memmem    : {ratio1:.2f}x")
    if plaintext_memmem_gbps > 0:
        ratio2 = fused_gbps / plaintext_memmem_gbps
        print(f"  Fused vs plaintext memmem  : {ratio2:.2f}x")
    print("=" * 66)

if __name__ == "__main__":
    main()
