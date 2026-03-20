"""Benchmark suite for multi-needle searchable ChaCha20 — v2 vs v1x3 vs plaintext."""
import ctypes as ct
import ctypes.util
import numpy as np
import os
import sys
import time
import platform
import statistics
import resource
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
DATA_SIZE   = 64 * 1024 * 1024  # 64 MB
WARMUP      = 3
TIMED       = 10
NCORES      = os.cpu_count() or 1
WINDOW_SIZE = 4096
OVERLAP_SZ  = 64
KS_SZ       = 256
MAX_MATCHES = 200000

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def print_sysinfo():
    print("=" * 70)
    print("System info")
    print("=" * 70)
    cpu_name = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    print(f"  CPU        : {cpu_name}")
    print(f"  Cores      : {NCORES}")
    print(f"  Platform   : {platform.platform()}")
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  NumPy      : {np.__version__}")
    print(f"  Data size  : {DATA_SIZE / 1024 / 1024:.0f} MB")
    print(f"  Warmup     : {WARMUP}   Timed: {TIMED}")
    print(f"  Needles    : ERROR, FATAL, PANIC")
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

_v2lib = ct.CDLL(str(_HERE / "chacha20_search_v2.so"))
_v2lib.chacha20_search_v2.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,  # key, nonce, ctr
    ct.POINTER(ct.c_uint8), ct.c_int32,                           # ct_u8, len
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),               # ks_i32, ks_u8
    ct.POINTER(ct.c_int32),                                        # ct_i32
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),               # pt_buf, pt_i32
    ct.POINTER(ct.c_uint8),                                        # overlap
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),               # needles, needle_offsets
    ct.POINTER(ct.c_int32), ct.c_int32,                           # needle_lens, needle_count
    ct.POINTER(ct.c_uint8), ct.c_int32,                           # lines_buf, lines_buf_cap
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),               # line_offsets, line_lens
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),               # match_offsets, needle_ids
    ct.c_int32, ct.c_int32, ct.c_int32,                           # max_matches, max_line_len, window_size
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),               # match_count, lines_written
]
_v2lib.chacha20_search_v2.restype = None

_libc = ct.CDLL(ctypes.util.find_library("c"))
_libc.memmem.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
_libc.memmem.restype = ct.c_void_p

# ---------------------------------------------------------------------------
# Key / nonce constants
# ---------------------------------------------------------------------------
KEY_U32   = [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
             0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c]
NONCE_U32 = [0x00000000, 0x4a000000, 0x00000000]
COUNTER   = 1

def to_i32_array(values):
    arr = (ct.c_int32 * len(values))()
    for i, v in enumerate(values):
        arr[i] = ct.c_int32(v & 0xFFFFFFFF).value
    return arr

# ---------------------------------------------------------------------------
# Test data: random bytes with "ERROR", "FATAL", "PANIC" injected every ~4 KB
# ---------------------------------------------------------------------------
NEEDLES     = [b"ERROR", b"FATAL", b"PANIC"]
NEEDLE_COUNT = len(NEEDLES)

rng = np.random.RandomState(42)
plaintext = rng.randint(0, 256, size=DATA_SIZE, dtype=np.uint8)

# Inject each needle in rotation every ~4096 bytes
for _idx, _pos in enumerate(range(0, DATA_SIZE - 5, 4096)):
    _needle = NEEDLES[_idx % NEEDLE_COUNT]
    plaintext[_pos : _pos + len(_needle)] = np.frombuffer(_needle, dtype=np.uint8)

# Pre-encrypt the plaintext once
ciphertext = np.empty(DATA_SIZE, dtype=np.uint8)

key_arr   = to_i32_array(KEY_U32)
nonce_arr = to_i32_array(NONCE_U32)

def _do_encrypt(pt, ct_out):
    scratch = (ct.c_uint8 * 64)()
    ks_i32  = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8   = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    pt_ptr  = pt.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_ptr  = ct_out.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pt_i32  = ct.cast(pt_ptr, ct.POINTER(ct.c_int32))
    ct_i32  = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    _ea.chacha20_encrypt(key_arr, nonce_arr, ct.c_int32(COUNTER),
                         pt_ptr, ct_ptr, ct.c_int32(DATA_SIZE),
                         ks_i32, ks_u8, pt_i32, ct_i32)

_do_encrypt(plaintext, ciphertext)

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
    sd  = statistics.stdev(gbps) if len(gbps) > 1 else 0.0
    print(f"  {name:<50s}  {med:8.3f} GB/s  (sd {sd:.3f})")
    return med, sd

# ---------------------------------------------------------------------------
# Build packed needle buffer for v2 (ERROR + FATAL + PANIC)
# ---------------------------------------------------------------------------
def _build_needle_buffers(needle_list):
    packed   = b""
    offsets  = []
    lens     = []
    for n in needle_list:
        offsets.append(len(packed))
        lens.append(len(n))
        packed += n
    needles_arr       = (ct.c_uint8  * len(packed))(*packed)
    needle_offsets_arr = (ct.c_int32 * len(offsets))(*offsets)
    needle_lens_arr    = (ct.c_int32 * len(lens))(*lens)
    return needles_arr, needle_offsets_arr, needle_lens_arr

_needles_arr, _needle_offsets_arr, _needle_lens_arr = _build_needle_buffers(NEEDLES)

# lines_buf: disable context lines to avoid overhead — use cap=1 so kernel
# never writes lines (lines_buf_cap < any line length).  All matches are
# still counted and stored in match_offsets / needle_ids.
_LINES_BUF_CAP = 1
_MAX_LINE_LEN   = 1024

# ---------------------------------------------------------------------------
# Benchmark 1: Ea v2 multi-needle (3 needles, single call)
# ---------------------------------------------------------------------------
def bench_v2_multi():
    # Pre-allocate ALL ctypes arrays outside timed closure
    ks_scratch    = (ct.c_uint8  * KS_SZ)()
    ks_i32        = ct.cast(ks_scratch, ct.POINTER(ct.c_int32))
    ks_u8         = ct.cast(ks_scratch, ct.POINTER(ct.c_uint8))

    ct_ptr        = ciphertext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_i32        = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))

    pt_buf_arr    = (ct.c_uint8  * WINDOW_SIZE)()
    pt_i32        = ct.cast(pt_buf_arr, ct.POINTER(ct.c_int32))
    pt_buf_ptr    = ct.cast(pt_buf_arr, ct.POINTER(ct.c_uint8))

    overlap_arr   = (ct.c_uint8  * OVERLAP_SZ)()
    overlap_ptr   = ct.cast(overlap_arr, ct.POINTER(ct.c_uint8))

    lines_buf     = (ct.c_uint8  * _LINES_BUF_CAP)()
    line_offsets  = (ct.c_int32  * MAX_MATCHES)()
    line_lens_arr = (ct.c_int32  * MAX_MATCHES)()
    match_offsets = (ct.c_int32  * MAX_MATCHES)()
    needle_ids    = (ct.c_int32  * MAX_MATCHES)()
    match_count   = (ct.c_int32  * 1)()
    lines_written = (ct.c_int32  * 1)()

    needle_ptr    = ct.cast(_needles_arr,       ct.POINTER(ct.c_uint8))
    off_ptr       = ct.cast(_needle_offsets_arr, ct.POINTER(ct.c_int32))
    len_ptr       = ct.cast(_needle_lens_arr,    ct.POINTER(ct.c_int32))

    def fn():
        match_count[0]   = 0
        lines_written[0] = 0
        _v2lib.chacha20_search_v2(
            key_arr, nonce_arr, ct.c_int32(COUNTER),
            ct_ptr, ct.c_int32(DATA_SIZE),
            ks_i32, ks_u8,
            ct_i32,
            pt_buf_ptr, pt_i32,
            overlap_ptr,
            needle_ptr, off_ptr, len_ptr, ct.c_int32(NEEDLE_COUNT),
            lines_buf, ct.c_int32(_LINES_BUF_CAP),
            line_offsets, line_lens_arr,
            match_offsets, needle_ids,
            ct.c_int32(MAX_MATCHES), ct.c_int32(_MAX_LINE_LEN), ct.c_int32(WINDOW_SIZE),
            match_count, lines_written,
        )

    return bench("Ea v2 multi-needle (3 needles, 1 call)", fn)

# ---------------------------------------------------------------------------
# Benchmark 2: Ea v1 single-needle x3 (3 separate calls)
# ---------------------------------------------------------------------------
def bench_v1_x3():
    ks_scratch  = (ct.c_uint8 * KS_SZ)()
    ks_i32      = ct.cast(ks_scratch, ct.POINTER(ct.c_int32))
    ks_u8       = ct.cast(ks_scratch, ct.POINTER(ct.c_uint8))

    ct_ptr      = ciphertext.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_i32      = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))

    pt_buf_arr  = (ct.c_uint8 * WINDOW_SIZE)()
    pt_i32      = ct.cast(pt_buf_arr, ct.POINTER(ct.c_int32))
    pt_buf_ptr  = ct.cast(pt_buf_arr, ct.POINTER(ct.c_uint8))

    overlap_arr = (ct.c_uint8 * OVERLAP_SZ)()
    overlap_ptr = ct.cast(overlap_arr, ct.POINTER(ct.c_uint8))

    matches_arr     = (ct.c_int32 * MAX_MATCHES)()
    match_count_arr = (ct.c_int32 * 1)()

    # Pre-build needle ctypes buffers for each of the 3 needles
    needle_bufs = []
    for n in NEEDLES:
        nb = (ct.c_uint8 * len(n))(*n)
        needle_bufs.append((nb, len(n)))

    def fn():
        total = 0
        for nb, nlen in needle_bufs:
            match_count_arr[0] = 0
            _search.chacha20_search(
                key_arr, nonce_arr, ct.c_int32(COUNTER),
                ct_ptr, ct.c_int32(DATA_SIZE),
                ct.cast(nb, ct.POINTER(ct.c_uint8)), ct.c_int32(nlen),
                ks_i32, ks_u8,
                ct_i32,
                pt_buf_ptr, pt_i32,
                overlap_ptr,
                matches_arr, ct.c_int32(MAX_MATCHES),
                match_count_arr,
            )
            total += match_count_arr[0]

    return bench("Ea v1 single-needle x3 (3 calls)", fn)

# ---------------------------------------------------------------------------
# Benchmark 3: Ea decrypt -> Python find x3
# ---------------------------------------------------------------------------
def bench_ea_decrypt_python_find_x3():
    scratch = (ct.c_uint8 * 64)()
    ks_i32  = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8   = ct.cast(scratch, ct.POINTER(ct.c_uint8))

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
        total = 0
        for needle in NEEDLES:
            pos = 0
            while True:
                idx = pt_bytes.find(needle, pos)
                if idx == -1:
                    break
                total += 1
                pos = idx + 1

    return bench("Ea decrypt -> Python find x3", fn)

# ---------------------------------------------------------------------------
# Benchmark 4: C memmem x3 on plaintext (in-memory, no decrypt)
# ---------------------------------------------------------------------------
def bench_memmem_plaintext_x3():
    pt_ptr_base = plaintext.ctypes.data

    # Pre-build needle c_char_p and sizes
    needle_c_list = [(ct.c_char_p(n), ct.c_size_t(len(n))) for n in NEEDLES]
    data_len = ct.c_size_t(DATA_SIZE)

    def fn():
        total = 0
        for needle_c, needle_len in needle_c_list:
            base      = pt_ptr_base
            remaining = DATA_SIZE
            ptr       = base
            while remaining >= needle_len.value:
                result = _libc.memmem(ptr, ct.c_size_t(remaining), needle_c, needle_len)
                if result is None:
                    break
                offset = result - base
                total += 1
                advance   = offset + 1
                ptr       = base + advance
                remaining = DATA_SIZE - advance

    return bench("C memmem x3 on plaintext (no decrypt)", fn)

# ===========================================================================
# Main
# ===========================================================================
def main():
    print_sysinfo()

    print("=" * 70)
    print("Benchmarks  (64 MB, 3 needles: ERROR/FATAL/PANIC, median of 10 runs)")
    print("=" * 70)

    results = []
    results.append(("Ea v2 multi-needle (3 needles, 1 call)",)   + bench_v2_multi())
    results.append(("Ea v1 single-needle x3 (3 calls)",)         + bench_v1_x3())
    results.append(("Ea decrypt -> Python find x3",)             + bench_ea_decrypt_python_find_x3())
    results.append(("C memmem x3 on plaintext (no decrypt)",)    + bench_memmem_plaintext_x3())

    print()
    print("=" * 70)
    print(f"  {'Implementation':<50s}  {'GB/s':>8s}  {'stddev':>8s}")
    print("-" * 70)
    for name, med, sd in results:
        print(f"  {name:<50s}  {med:8.3f}  {sd:8.3f}")
    print("=" * 70)

    v2_gbps       = results[0][1]
    v1x3_gbps     = results[1][1]
    memmem_gbps   = results[3][1]

    print()
    print("Ratios")
    print("-" * 70)
    if v1x3_gbps > 0:
        ratio1 = v2_gbps / v1x3_gbps
        print(f"  v2 multi-needle vs v1x3              : {ratio1:.2f}x")
    if memmem_gbps > 0:
        ratio2 = v2_gbps / memmem_gbps
        print(f"  v2 multi-needle vs plaintext memmem  : {ratio2:.2f}x")
    print("=" * 70)

if __name__ == "__main__":
    main()
