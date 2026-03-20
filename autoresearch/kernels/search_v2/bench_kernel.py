#!/usr/bin/env python3
"""Compile and benchmark the v2 search kernel. Outputs one JSON line to stdout.

Measures throughput across multiple ciphertext sizes. The reported time_us
is at the largest size (real-world, exceeds cache). Correctness is verified
by comparing match offsets against a Python reference implementation.
"""

import ctypes as ct
import ctypes.util
import json
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

# Dataset sizes for benchmarking (bytes of ciphertext)
DATASET_SIZES = [64_000, 256_000, 1_000_000, 16_000_000]
BYTES_PER_ELEM = 1  # 1 byte ciphertext read per byte processed
NUM_RUNS = 20
WARMUP_RUNS = 3
SEED = 42
NEEDLE_DENSITY = 4096  # inject one needle every ~4KB

# Search parameters
NEEDLES = [b"ERROR", b"FATAL", b"PANIC"]
WINDOW_SIZE = 4096
MAX_LINE_LEN = 1024
MAX_MATCHES = 100000

# ChaCha20 key/nonce (same as test suites)
KEY_U32 = [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
           0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c]
NONCE_U32 = [0x00000000, 0x4a000000, 0x00000000]
COUNTER = 1

# Paths
EACHACHA_DIR = Path(__file__).resolve().parent.parent.parent.parent
CHACHA_SO = EACHACHA_DIR / "chacha20.so"


def to_i32_array(values):
    arr = (ct.c_int32 * len(values))()
    for i, v in enumerate(values):
        arr[i] = ct.c_int32(v & 0xFFFFFFFF).value
    return arr


def count_loc(path):
    count = 0
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                count += 1
    return count


def output(correct, time_us=None, min_us=None, loc=None, error=None, breakdown=None):
    result = {"correct": correct, "time_us": time_us, "min_us": min_us,
              "loc": loc, "error": error}
    if breakdown:
        result["breakdown"] = breakdown
    print(json.dumps(result))
    sys.exit(0)


def generate_test_data(size):
    """Generate random bytes with needles injected every ~NEEDLE_DENSITY bytes."""
    rng = np.random.RandomState(SEED)
    data = rng.randint(0, 256, size=size, dtype=np.uint8)
    for needle in NEEDLES:
        narr = np.frombuffer(needle, dtype=np.uint8)
        for offset in range(0, size - len(needle), NEEDLE_DENSITY):
            pos = offset + rng.randint(0, min(NEEDLE_DENSITY, size - offset - len(needle)))
            data[pos:pos + len(narr)] = narr
    return data


def encrypt_data(plaintext_np, encrypt_func):
    """Encrypt numpy u8 array, return ciphertext numpy array."""
    size = len(plaintext_np)
    ct_buf = np.empty(size, dtype=np.uint8)
    scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    pt_ptr = plaintext_np.ctypes.data_as(ct.POINTER(ct.c_uint8))
    ct_ptr = ct_buf.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pt_i32 = ct.cast(pt_ptr, ct.POINTER(ct.c_int32))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    encrypt_func(to_i32_array(KEY_U32), to_i32_array(NONCE_U32), ct.c_int32(COUNTER),
                 pt_ptr, ct_ptr, ct.c_int32(size), ks_i32, ks_u8, pt_i32, ct_i32)
    return ct_buf


def find_all_multi_py(data_bytes, needles):
    """Python reference: find all occurrences of all needles."""
    results = []
    for idx, needle in enumerate(needles):
        pos = 0
        while True:
            p = data_bytes.find(needle, pos)
            if p == -1:
                break
            results.append((p, idx))
            pos = p + 1
    results.sort()
    return results


def pack_needles(needle_list):
    """Pack needles into concatenated buffer + offset/length arrays."""
    packed = b""
    offsets = []
    lens = []
    for n in needle_list:
        offsets.append(len(packed))
        lens.append(len(n))
        packed += n
    return packed, offsets, lens


def bench_at_size(search_func, encrypt_func, search_argtypes, size):
    """Benchmark at a specific ciphertext size. Returns (median_us, min_us) or error string."""
    # Generate and encrypt test data
    plaintext = generate_test_data(size)
    ciphertext = encrypt_data(plaintext, encrypt_func)
    plaintext_bytes = bytes(plaintext)

    # Python reference
    ref_matches = find_all_multi_py(plaintext_bytes, NEEDLES)
    ref_offsets = [m[0] for m in ref_matches]

    # Pack needles
    packed, noffsets, nlens = pack_needles(NEEDLES)
    needles_buf = (ct.c_uint8 * len(packed))(*packed)
    offsets_arr = (ct.c_int32 * len(noffsets))(*noffsets)
    lens_arr = (ct.c_int32 * len(nlens))(*nlens)

    # Allocate buffers
    key_arr = to_i32_array(KEY_U32)
    nonce_arr = to_i32_array(NONCE_U32)
    ct_buf = (ct.c_uint8 * size).from_buffer(ciphertext)
    ct_ptr = ct.cast(ct_buf, ct.POINTER(ct.c_uint8))
    ks = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks, ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_ptr, ct.POINTER(ct.c_int32))
    pt_buf = (ct.c_uint8 * WINDOW_SIZE)()
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
    overlap = (ct.c_uint8 * 64)()
    lines_buf_cap = MAX_MATCHES * 256
    lines_buf = (ct.c_uint8 * lines_buf_cap)()
    line_offsets = (ct.c_int32 * MAX_MATCHES)()
    line_lens = (ct.c_int32 * MAX_MATCHES)()
    match_offsets = (ct.c_int32 * MAX_MATCHES)()
    needle_ids = (ct.c_int32 * MAX_MATCHES)()
    match_count = (ct.c_int32 * 1)()
    lines_written = (ct.c_int32 * 1)()

    def run():
        search_func(
            key_arr, nonce_arr, ct.c_int32(COUNTER),
            ct_ptr, ct.c_int32(size),
            ks_i32, ks_u8, ct_i32,
            pt_buf, pt_i32, overlap,
            needles_buf, offsets_arr, lens_arr, ct.c_int32(len(NEEDLES)),
            lines_buf, ct.c_int32(lines_buf_cap),
            line_offsets, line_lens, match_offsets, needle_ids,
            ct.c_int32(MAX_MATCHES), ct.c_int32(MAX_LINE_LEN), ct.c_int32(WINDOW_SIZE),
            match_count, lines_written)

    # Correctness check
    run()
    mc = match_count[0]
    got_offsets = sorted([match_offsets[i] for i in range(mc)])

    if got_offsets != ref_offsets:
        return (f"correctness: got {len(got_offsets)} matches, expected {len(ref_offsets)}. "
                f"First diff at index {next((i for i,(a,b) in enumerate(zip(got_offsets, ref_offsets)) if a != b), min(len(got_offsets), len(ref_offsets)))}")

    # Benchmark
    for _ in range(WARMUP_RUNS):
        run()

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)

    times.sort()
    median_us = round(times[len(times) // 2] * 1e6, 1)
    min_us = round(times[0] * 1e6, 1)
    return (median_us, min_us)


def main():
    if len(sys.argv) < 2:
        print("Usage: bench_kernel.py <kernel.ea> [--no-compile]", file=sys.stderr)
        sys.exit(1)

    kernel_path = Path(sys.argv[1]).resolve()
    no_compile = "--no-compile" in sys.argv

    ea_binary = os.environ.get("EA_BINARY",
                               "/usr/local/lib/python3.13/dist-packages/ea/bin/ea")

    # --- Compile ---
    kernel_dir = kernel_path.parent
    so_name = kernel_path.stem + ".so"
    so_path = kernel_dir / so_name

    if not no_compile:
        for stale in [so_path]:
            if stale.exists():
                stale.unlink()

        result = subprocess.run(
            [ea_binary, str(kernel_path), "--lib", "--opt-level=3"],
            capture_output=True, text=True, cwd=str(kernel_dir))
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            output(False, error=f"compile: {error_msg}")

        if not so_path.exists():
            output(False, error="compile: .so not found")

    # --- Load libraries ---
    try:
        search_lib = ct.CDLL(str(so_path.resolve()))
        encrypt_lib = ct.CDLL(str(CHACHA_SO.resolve()))
    except OSError as e:
        output(False, error=f"load: {e}")

    # Set up argtypes
    encrypt_lib.chacha20_encrypt.argtypes = [
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
        ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)]
    encrypt_lib.chacha20_encrypt.restype = None

    try:
        search_func = search_lib.chacha20_search_v2
    except AttributeError as e:
        output(False, error=f"symbol: {e}")

    search_func.argtypes = [
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
        ct.POINTER(ct.c_uint8), ct.c_int32,
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int32), ct.c_int32,
        ct.POINTER(ct.c_uint8), ct.c_int32,
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.c_int32, ct.c_int32, ct.c_int32,
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)]
    search_func.restype = None

    # --- Benchmark across sizes ---
    breakdown = {}
    for size in DATASET_SIZES:
        label = f"{size // 1000}K" if size < 1_000_000 else f"{size // 1_000_000}M"
        result = bench_at_size(search_func, encrypt_lib.chacha20_encrypt,
                               search_func.argtypes, size)

        if isinstance(result, str):
            output(False, error=result)

        median_us, min_us = result
        gbs = size / (median_us / 1e6) / 1e9
        breakdown[label] = {"median_us": median_us, "min_us": min_us, "gbs": round(gbs, 2)}
        print(f"  {label}: {median_us} µs median  |  {gbs:.2f} GB/s", file=sys.stderr)

    # Primary metric: largest size
    largest_label = list(breakdown.keys())[-1]
    aggregate = breakdown[largest_label]
    loc = count_loc(kernel_path)

    output(True, time_us=aggregate["median_us"], min_us=aggregate["min_us"],
           loc=loc, breakdown=breakdown)


if __name__ == "__main__":
    main()
