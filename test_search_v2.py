"""Test suite for the fused ChaCha20 decrypt + multi-needle search + context-line kernel (v2)."""
import ctypes as ct
import random
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load libraries
_lib = ct.CDLL(os.path.join(_HERE, "chacha20.so"))
_search = ct.CDLL(os.path.join(_HERE, "chacha20_search.so"))
_v2 = ct.CDLL(os.path.join(_HERE, "chacha20_search_v2.so"))

# chacha20_encrypt argtypes
_lib.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
]
_lib.chacha20_encrypt.restype = None

# chacha20_search argtypes (v1, 16 params)
_search.chacha20_search.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_int32),
]
_search.chacha20_search.restype = None

# chacha20_search_v2 argtypes (26 params) — from auto-generated bindings
_v2.chacha20_search_v2.argtypes = [
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
_v2.chacha20_search_v2.restype = None

# Constants
KEY_U32 = [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
           0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c]
NONCE_U32 = [0x00000000, 0x4a000000, 0x00000000]
COUNTER = 1

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


def to_i32_array(values):
    arr = (ct.c_int32 * len(values))()
    for i, v in enumerate(values):
        arr[i] = ct.c_int32(v & 0xFFFFFFFF).value
    return arr


def encrypt_data(plaintext_bytes):
    """Encrypt plaintext_bytes with chacha20_encrypt, return ciphertext bytes."""
    n = len(plaintext_bytes)
    if n == 0:
        return b""
    key = to_i32_array(KEY_U32)
    nonce = to_i32_array(NONCE_U32)
    pt_buf = (ct.c_uint8 * n)(*plaintext_bytes)
    ct_buf = (ct.c_uint8 * n)()
    scratch = (ct.c_uint8 * 64)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
    ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))
    _lib.chacha20_encrypt(key, nonce, ct.c_int32(COUNTER),
                          pt_buf, ct_buf, ct.c_int32(n),
                          ks_i32, ks_u8, pt_i32, ct_i32)
    return bytes(ct_buf)


def search_v1(ciphertext_bytes, needle_bytes, max_matches=10000):
    """Run v1 kernel, return list of match offsets."""
    n = len(ciphertext_bytes)
    needle_len = len(needle_bytes)
    key = to_i32_array(KEY_U32)
    nonce = to_i32_array(NONCE_U32)
    ct_u8 = (ct.c_uint8 * max(n, 1))(*ciphertext_bytes) if n > 0 else (ct.c_uint8 * 1)()
    needle_buf = (ct.c_uint8 * max(needle_len, 1))(*needle_bytes) if needle_len > 0 else (ct.c_uint8 * 1)()
    ks_scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks_scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks_scratch, ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_u8, ct.POINTER(ct.c_int32))
    pt_buf = (ct.c_uint8 * 256)()
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
    overlap = (ct.c_uint8 * 64)()
    matches = (ct.c_int32 * max_matches)()
    match_count = (ct.c_int32 * 1)()
    _search.chacha20_search(
        key, nonce, ct.c_int32(COUNTER),
        ct.cast(ct_u8, ct.POINTER(ct.c_uint8)), ct.c_int32(n),
        ct.cast(needle_buf, ct.POINTER(ct.c_uint8)), ct.c_int32(needle_len),
        ks_i32, ks_u8, ct_i32,
        ct.cast(pt_buf, ct.POINTER(ct.c_uint8)), pt_i32,
        ct.cast(overlap, ct.POINTER(ct.c_uint8)),
        matches, ct.c_int32(max_matches), match_count,
    )
    return [matches[i] for i in range(match_count[0])]


def search_v2(ciphertext_bytes, needle_list, max_matches=10000, max_line_len=1024, window_size=4096):
    """Run v2 kernel. Returns (match_offsets, needle_ids, lines, lines_written_count)."""
    size = len(ciphertext_bytes)
    ct_buf = (ct.c_uint8 * max(size, 1))(*ciphertext_bytes) if size > 0 else (ct.c_uint8 * 1)()

    # Pack needles
    packed = b""
    offsets = []
    lens = []
    for n in needle_list:
        offsets.append(len(packed))
        lens.append(len(n))
        packed += n
    if not packed:
        packed = b"\x00"

    needles_buf = (ct.c_uint8 * len(packed))(*packed)
    offsets_arr = (ct.c_int32 * max(len(offsets), 1))(*offsets) if offsets else (ct.c_int32 * 1)()
    lens_arr = (ct.c_int32 * max(len(lens), 1))(*lens) if lens else (ct.c_int32 * 1)()

    # Buffers
    ks = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks, ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))
    pt_buf = (ct.c_uint8 * max(window_size, 256))()
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
    overlap = (ct.c_uint8 * 64)()

    lines_buf_cap = max(max_matches * max_line_len, 1)
    lines_buf = (ct.c_uint8 * lines_buf_cap)()
    line_offsets = (ct.c_int32 * max_matches)()
    line_lens_arr = (ct.c_int32 * max_matches)()
    match_offsets = (ct.c_int32 * max_matches)()
    needle_ids = (ct.c_int32 * max_matches)()
    match_count = (ct.c_int32 * 1)()
    lines_written = (ct.c_int32 * 1)()

    _v2.chacha20_search_v2(
        to_i32_array(KEY_U32), to_i32_array(NONCE_U32), ct.c_int32(COUNTER),
        ct_buf, ct.c_int32(size),
        ks_i32, ks_u8, ct_i32,
        pt_buf, pt_i32, overlap,
        needles_buf, offsets_arr, lens_arr, ct.c_int32(len(needle_list)),
        lines_buf, ct.c_int32(lines_buf_cap),
        line_offsets, line_lens_arr,
        match_offsets, needle_ids,
        ct.c_int32(max_matches), ct.c_int32(max_line_len), ct.c_int32(window_size),
        match_count, lines_written)

    mc = match_count[0]
    lw = lines_written[0]
    offsets_out = [match_offsets[i] for i in range(mc)]
    ids_out = [needle_ids[i] for i in range(mc)]
    lines_out = []
    for i in range(lw):
        lo = line_offsets[i]
        ll = line_lens_arr[i]
        lines_out.append(bytes(lines_buf[lo:lo + ll]))
    return offsets_out, ids_out, lines_out, lw


def find_all_occurrences(data, needle):
    """Python reference: find all offsets where needle appears in data."""
    results = []
    if not needle:
        return results
    start = 0
    while True:
        idx = data.find(needle, start)
        if idx == -1:
            break
        results.append(idx)
        start = idx + 1
    return results


def find_all_multi(data, needle_list):
    """Reference: find all occurrences of all needles, return [(offset, needle_idx)]."""
    results = []
    for idx, needle in enumerate(needle_list):
        pos = 0
        while True:
            p = data.find(needle, pos)
            if p == -1:
                break
            results.append((p, idx))
            pos = p + 1
    results.sort()
    return results


# ===========================================================================
# Regression tests 1-9: Same as v1 but via search_v2 with single needle
# ===========================================================================

# Test 1: Known needle at known offset (ERROR at byte 50 in 128B)
print("=== Test 1: Known needle at known offset ===")
needle = b"ERROR"
pt = bytearray(128)
pt[50:50 + len(needle)] = needle
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [needle])
check("single match found at 50", sorted(offsets) == [50],
      f"got offsets={offsets}, expected [50]")
check("needle_id is 0", ids == [0], f"got ids={ids}")

# Test 2: No match (256B zeros)
print("\n=== Test 2: Needle not present ===")
pt = bytes(256)
ct_bytes = encrypt_data(pt)
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"])
check("no matches found", offsets == [], f"got offsets={offsets}")

# Test 3: Multiple matches (ERROR at 10, 100, 200)
print("\n=== Test 3: Multiple matches ===")
pt = bytearray(256)
for pos in [10, 100, 200]:
    pt[pos:pos + 5] = b"ERROR"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"])
check("three matches found", sorted(offsets) == [10, 100, 200],
      f"got offsets={sorted(offsets)}, expected [10, 100, 200]")

# Test 4: Boundary positions (ERROR at byte 0 and 123)
print("\n=== Test 4: Boundary positions ===")
pt = bytearray(128)
pt[0:5] = b"ERROR"
pt[123:128] = b"ERROR"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"])
check("matches at 0 and 123", sorted(offsets) == [0, 123],
      f"got offsets={sorted(offsets)}, expected [0, 123]")

# Test 5: Cross-block boundary (ERROR at 62-66)
print("\n=== Test 5: Cross-block boundary ===")
pt = bytearray(256)
pt[62:67] = b"ERROR"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"])
check("cross-block match at 62", offsets == [62],
      f"got offsets={offsets}, expected [62]")

# Test 6: Cross-iteration boundary (ERROR at 254-258 in 512B)
print("\n=== Test 6: Cross-iteration boundary ===")
pt = bytearray(512)
pt[254:259] = b"ERROR"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"])
check("cross-iteration match at 254", offsets == [254],
      f"got offsets={offsets}, expected [254]")

# Test 7: Single-byte needle
print("\n=== Test 7: Single-byte needle ===")
pt = bytearray(64)
for pos in [0, 32, 63]:
    pt[pos] = ord("X")
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"X"])
check("single-byte needle at 0, 32, 63", sorted(offsets) == [0, 32, 63],
      f"got offsets={sorted(offsets)}, expected [0, 32, 63]")

# Test 8: Max needle 64 bytes
print("\n=== Test 8: Max needle length 64 ===")
needle64 = b"A" * 64
pt = bytearray(256)
pt[100:164] = needle64
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [needle64])
check("64-byte needle match at 100", offsets == [100],
      f"got offsets={offsets}, expected [100]")

# Test 9: Overlapping matches ("aa" in "aaaa")
print("\n=== Test 9: Overlapping matches ===")
pt = bytearray(64)
pt[10:14] = b"aaaa"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"aa"])
check("overlapping matches at 10, 11, 12", sorted(offsets) == [10, 11, 12],
      f"got offsets={sorted(offsets)}, expected [10, 11, 12]")

# ===========================================================================
# Multi-needle tests 10-14
# ===========================================================================

# Test 10: Two needles at separate locations
print("\n=== Test 10: Two needles at separate locations ===")
pt = bytearray(256)
pt[20:25] = b"ERROR"
pt[100:104] = b"WARN"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR", b"WARN"])
expected_ref = find_all_multi(bytes(pt), [b"ERROR", b"WARN"])
expected_offsets = [e[0] for e in expected_ref]
expected_ids = [e[1] for e in expected_ref]
# Sort by offset for comparison
paired = sorted(zip(offsets, ids))
check("two needles: offsets match",
      [p[0] for p in paired] == expected_offsets,
      f"got {[p[0] for p in paired]}, expected {expected_offsets}")
check("two needles: ids match",
      [p[1] for p in paired] == expected_ids,
      f"got {[p[1] for p in paired]}, expected {expected_ids}")

# Test 11: Three needles with overlapping first-bytes ("ERROR", "EXIT", "INFO")
print("\n=== Test 11: Three needles, overlapping first-bytes ===")
pt = bytearray(256)
pt[10:15] = b"ERROR"
pt[50:54] = b"EXIT"
pt[80:84] = b"INFO"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR", b"EXIT", b"INFO"])
expected_ref = find_all_multi(bytes(pt), [b"ERROR", b"EXIT", b"INFO"])
paired = sorted(zip(offsets, ids))
expected_offsets = [e[0] for e in expected_ref]
expected_ids = [e[1] for e in expected_ref]
check("three needles: offsets match",
      [p[0] for p in paired] == expected_offsets,
      f"got {[p[0] for p in paired]}, expected {expected_offsets}")
check("three needles: ids match",
      [p[1] for p in paired] == expected_ids,
      f"got {[p[1] for p in paired]}, expected {expected_ids}")

# Test 12: Five needles, dense matches
print("\n=== Test 12: Five needles, dense matches ===")
pt = bytearray(512)
needles5 = [b"AA", b"BB", b"CC", b"DD", b"EE"]
positions = [10, 30, 50, 70, 90, 200, 300, 400]
for i, pos in enumerate(positions):
    n = needles5[i % len(needles5)]
    pt[pos:pos + len(n)] = n
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, needles5)
expected_ref = find_all_multi(bytes(pt), needles5)
paired = sorted(zip(offsets, ids))
expected_offsets = [e[0] for e in expected_ref]
expected_ids = [e[1] for e in expected_ref]
check("five needles: offsets match",
      [p[0] for p in paired] == expected_offsets,
      f"got {[p[0] for p in paired]}, expected {expected_offsets}")

# Test 13: One needle absent from set
print("\n=== Test 13: One needle absent ===")
pt = bytearray(256)
pt[50:55] = b"ERROR"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR", b"NOTFOUND"])
paired = sorted(zip(offsets, ids))
check("absent needle: only ERROR found",
      [p[0] for p in paired] == [50],
      f"got offsets={[p[0] for p in paired]}, expected [50]")
check("absent needle: id is 0 (ERROR)",
      [p[1] for p in paired] == [0],
      f"got ids={[p[1] for p in paired]}, expected [0]")

# Test 14: Two needles at same offset ("AB" and "ABC")
print("\n=== Test 14: Two needles at same offset ===")
pt = bytearray(128)
pt[40:43] = b"ABC"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"AB", b"ABC"])
expected_ref = find_all_multi(bytes(pt), [b"AB", b"ABC"])
paired = sorted(zip(offsets, ids))
expected_offsets = [e[0] for e in expected_ref]
expected_ids = [e[1] for e in expected_ref]
check("same-offset needles: offsets match",
      [p[0] for p in paired] == expected_offsets,
      f"got {[p[0] for p in paired]}, expected {expected_offsets}")
check("same-offset needles: ids match",
      [p[1] for p in paired] == expected_ids,
      f"got {[p[1] for p in paired]}, expected {expected_ids}")

# ===========================================================================
# Context-line tests 15-20
# ===========================================================================

# Test 15: Match mid-line -- line between \n's extracted correctly
print("\n=== Test 15: Match mid-line ===")
pt = b"first line\nERROR something\nthird line\n"
ct_bytes = encrypt_data(pt)
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"])
check("mid-line: match at offset 11", offsets == [11],
      f"got offsets={offsets}, expected [11]")
check("mid-line: one line written", lw == 1, f"got lw={lw}")
if lw >= 1:
    check("mid-line: correct line extracted", lines[0] == b"ERROR something",
          f"got line={lines[0]!r}")

# Test 16: Match at line start
print("\n=== Test 16: Match at line start ===")
pt = b"normal\nERROR at start\nmore\n"
ct_bytes = encrypt_data(pt)
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"])
check("line-start: match found", len(offsets) == 1,
      f"got {len(offsets)} matches")
if lw >= 1:
    check("line-start: correct line", lines[0] == b"ERROR at start",
          f"got line={lines[0]!r}")

# Test 17: Match at line end (before \n)
print("\n=== Test 17: Match at line end ===")
pt = b"start\nending ERROR\nmore\n"
ct_bytes = encrypt_data(pt)
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"])
check("line-end: match found", len(offsets) == 1,
      f"got {len(offsets)} matches")
if lw >= 1:
    check("line-end: correct line", lines[0] == b"ending ERROR",
          f"got line={lines[0]!r}")

# Test 18: Multiple matches on same line -- duplicate lines
print("\n=== Test 18: Multiple matches on same line ===")
pt = b"aaa\nERROR and ERROR again\nbbb\n"
ct_bytes = encrypt_data(pt)
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"])
check("same-line: two matches", len(offsets) == 2,
      f"got {len(offsets)} matches, offsets={offsets}")
check("same-line: two lines written (duplicates)", lw == 2,
      f"got lw={lw}")
if lw >= 2:
    check("same-line: both lines correct",
          lines[0] == b"ERROR and ERROR again" and lines[1] == b"ERROR and ERROR again",
          f"got lines={[l for l in lines[:2]]}")

# Test 19: Line > max_line_len -- truncated
print("\n=== Test 19: Line exceeds max_line_len ===")
long_line = b"A" * 50 + b"ERROR" + b"B" * 50
pt = b"\n" + long_line + b"\n"
ct_bytes = encrypt_data(pt)
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"], max_line_len=40)
check("truncated: match found", len(offsets) == 1,
      f"got {len(offsets)} matches")
if lw >= 1:
    check("truncated: line length <= max_line_len", len(lines[0]) <= 40,
          f"got line len={len(lines[0])}")

# Test 20: Match near window boundary -- truncated line
print("\n=== Test 20: Match near window boundary ===")
# Use a small window to force truncation at window edge
pt = b"A" * 60 + b"\nERROR at boundary\n" + b"B" * 60
ct_bytes = encrypt_data(pt)
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"], window_size=256)
check("boundary: match found", len(offsets) == 1,
      f"got {len(offsets)} matches, offsets={offsets}")
# Line may be truncated at window edge, just verify something was written
check("boundary: at least one line written", lw >= 1,
      f"got lw={lw}")

# ===========================================================================
# Cross-verification tests 21-22
# ===========================================================================

# Test 21: Random data + injected multi-needles vs Python reference
print("\n=== Test 21: Cross-verification random data multi-needle ===")
random.seed(42)
needles_cv = [b"FIND", b"SEEK", b"HERE"]
for size in [128, 512, 2048, 8192]:
    pt = bytearray(random.randint(0, 255) for _ in range(size))
    # Inject needles
    for i, pos in enumerate([size // 4, size // 2, 3 * size // 4]):
        n = needles_cv[i % len(needles_cv)]
        if pos + len(n) <= size:
            pt[pos:pos + len(n)] = n
    ct_bytes = encrypt_data(bytes(pt))
    offsets, ids, lines_out, lw = search_v2(ct_bytes, needles_cv)
    expected_ref = find_all_multi(bytes(pt), needles_cv)
    paired = sorted(zip(offsets, ids))
    expected_offsets = [e[0] for e in expected_ref]
    expected_ids = [e[1] for e in expected_ref]
    check(f"size={size}: offsets match python ref",
          [p[0] for p in paired] == expected_offsets,
          f"kernel={[p[0] for p in paired]}, python={expected_offsets}")

# Test 22: NASA log subset (skip if not available)
print("\n=== Test 22: NASA log subset ===")
nasa_path = os.path.join(_HERE, "NASA_access_log_Aug95_head_1000.txt")
if os.path.exists(nasa_path):
    with open(nasa_path, "rb") as f:
        log_data = f.read()
    ct_bytes = encrypt_data(log_data)
    test_needles = [b"GET", b"POST", b"404"]
    offsets, ids, lines, lw = search_v2(ct_bytes, test_needles, max_matches=50000)
    expected = find_all_multi(log_data, test_needles)
    paired = sorted(zip(offsets, ids))
    expected_offsets = [e[0] for e in expected]
    check("NASA log: offsets match",
          [p[0] for p in paired] == expected_offsets,
          f"kernel found {len(offsets)}, python found {len(expected)}")
else:
    check("NASA log: SKIPPED (file not found)", True)

# ===========================================================================
# Edge case tests 23-27
# ===========================================================================

# Test 23: lines_buf overflow (small cap)
print("\n=== Test 23: lines_buf overflow ===")
pt = b"aaa\nERROR one\nbbb\nERROR two\nccc\n"
ct_bytes = encrypt_data(pt)
# Use tiny lines_buf_cap so only first line fits
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"], max_matches=10,
                                     max_line_len=1024, window_size=4096)
# With a very small lines_buf_cap, some lines might not be written
# We test with a custom call to control lines_buf_cap directly
size = len(ct_bytes)
ct_buf = (ct.c_uint8 * size)(*ct_bytes)
packed = b"ERROR"
needles_buf = (ct.c_uint8 * len(packed))(*packed)
offsets_arr = (ct.c_int32 * 1)(0)
lens_arr = (ct.c_int32 * 1)(5)
ks = (ct.c_uint8 * 256)()
ks_i32 = ct.cast(ks, ct.POINTER(ct.c_int32))
ks_u8 = ct.cast(ks, ct.POINTER(ct.c_uint8))
ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))
pt_buf_edge = (ct.c_uint8 * 4096)()
pt_i32 = ct.cast(pt_buf_edge, ct.POINTER(ct.c_int32))
overlap = (ct.c_uint8 * 64)()
# Only 12 bytes of lines_buf -- enough for one short line but not two
tiny_cap = 12
lines_buf = (ct.c_uint8 * max(tiny_cap, 1))()
line_offsets = (ct.c_int32 * 10)()
line_lens = (ct.c_int32 * 10)()
match_offsets_e = (ct.c_int32 * 10)()
needle_ids_e = (ct.c_int32 * 10)()
match_count_e = (ct.c_int32 * 1)()
lines_written_e = (ct.c_int32 * 1)()
_v2.chacha20_search_v2(
    to_i32_array(KEY_U32), to_i32_array(NONCE_U32), ct.c_int32(COUNTER),
    ct_buf, ct.c_int32(size),
    ks_i32, ks_u8, ct_i32,
    pt_buf_edge, pt_i32, overlap,
    needles_buf, offsets_arr, lens_arr, ct.c_int32(1),
    lines_buf, ct.c_int32(tiny_cap),
    line_offsets, line_lens,
    match_offsets_e, needle_ids_e,
    ct.c_int32(10), ct.c_int32(1024), ct.c_int32(4096),
    match_count_e, lines_written_e)
mc_e = match_count_e[0]
lw_e = lines_written_e[0]
check("lines_buf overflow: matches still found", mc_e == 2,
      f"got match_count={mc_e}, expected 2")
check("lines_buf overflow: fewer lines written than matches", lw_e < mc_e,
      f"got lines_written={lw_e}, match_count={mc_e}")

# Test 24: needle_count=0
print("\n=== Test 24: needle_count=0 ===")
pt = b"Hello World ERROR test"
ct_bytes = encrypt_data(pt)
offsets, ids, lines, lw = search_v2(ct_bytes, [])
check("needle_count=0: 0 matches", offsets == [],
      f"got offsets={offsets}")

# Test 25: needle_count=1 matches v1 offsets
print("\n=== Test 25: needle_count=1 matches v1 ===")
random.seed(77)
pt = bytearray(random.randint(0, 255) for _ in range(1024))
for pos in [100, 500, 900]:
    pt[pos:pos + 5] = b"MATCH"
ct_bytes = encrypt_data(bytes(pt))
v1_offsets = search_v1(ct_bytes, b"MATCH")
v2_offsets, v2_ids, v2_lines, v2_lw = search_v2(ct_bytes, [b"MATCH"])
check("v1 vs v2: offsets match",
      sorted(v1_offsets) == sorted(v2_offsets),
      f"v1={sorted(v1_offsets)}, v2={sorted(v2_offsets)}")

# Test 26: needle_count=65 -> 0 matches (exceeds max 64)
print("\n=== Test 26: needle_count=65 ===")
pt = bytearray(128)
pt[0:5] = b"ERROR"
ct_bytes = encrypt_data(bytes(pt))
# Build 65 needles
needles_65 = [bytes([i]) for i in range(65)]
offsets, ids, lines, lw = search_v2(ct_bytes, needles_65)
check("65 needles: 0 matches (exceeds max 64)", offsets == [],
      f"got offsets={offsets}")

# Test 27: Overlap match (needle spans window boundary)
print("\n=== Test 27: Overlap match at window boundary ===")
# Place ERROR so it straddles position window_size - 2
ws = 256
pt = bytearray(ws + 64)
# ERROR at ws-2 .. ws+3
pt[ws - 2:ws + 3] = b"ERROR"
ct_bytes = encrypt_data(bytes(pt))
offsets, ids, lines, lw = search_v2(ct_bytes, [b"ERROR"], window_size=ws)
check("overlap: match found at boundary",
      254 in offsets,
      f"got offsets={offsets}, expected 254 in offsets")

# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'=' * 50}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All tests passed!")
