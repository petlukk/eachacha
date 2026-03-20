"""Test suite for the fused ChaCha20 decrypt+search kernel (chacha20_search.so)."""
import ctypes as ct
import random
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load both libraries
_lib = ct.CDLL(os.path.join(_HERE, "chacha20.so"))
_search = ct.CDLL(os.path.join(_HERE, "chacha20_search.so"))

# chacha20_encrypt argtypes
_lib.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
]
_lib.chacha20_encrypt.restype = None

# chacha20_search argtypes (16 params)
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

# Constants — same as test_fused.py
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


def search_ciphertext(ciphertext_bytes, needle_bytes, max_matches=10000):
    """Run chacha20_search on ciphertext_bytes, return list of match offsets."""
    n = len(ciphertext_bytes)
    needle_len = len(needle_bytes)

    key = to_i32_array(KEY_U32)
    nonce = to_i32_array(NONCE_U32)

    # ct_u8: ciphertext buffer (or 1-byte dummy for zero-length)
    if n == 0:
        ct_u8 = (ct.c_uint8 * 1)()
    else:
        ct_u8 = (ct.c_uint8 * n)(*ciphertext_bytes)

    # needle buffer — use 1-byte dummy for zero-length needle
    if needle_len == 0:
        needle_buf = (ct.c_uint8 * 1)()
    else:
        needle_buf = (ct.c_uint8 * needle_len)(*needle_bytes)

    # scratch buffers
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
        ks_i32, ks_u8,
        ct_i32,
        ct.cast(pt_buf, ct.POINTER(ct.c_uint8)), pt_i32,
        ct.cast(overlap, ct.POINTER(ct.c_uint8)),
        matches, ct.c_int32(max_matches),
        match_count,
    )

    count = match_count[0]
    return [matches[i] for i in range(count)]


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
        start = idx + 1  # overlapping matches
    return results


# ---------------------------------------------------------------------------
# Test 1: Known needle at known offset (ERROR at byte 50 in 128-byte buffer)
# ---------------------------------------------------------------------------
print("=== Test 1: Known needle at known offset ===")

needle = b"ERROR"
pt = bytearray(128)
pt[50:50 + len(needle)] = needle
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("single match found", offsets == [50],
      f"got offsets={offsets}, expected [50]")

# ---------------------------------------------------------------------------
# Test 2: Needle not present (256 bytes of zeros)
# ---------------------------------------------------------------------------
print("\n=== Test 2: Needle not present ===")

needle = b"ERROR"
pt = bytes(256)  # all zeros
ct_bytes = encrypt_data(pt)
offsets = search_ciphertext(ct_bytes, needle)

check("no matches found", offsets == [],
      f"got offsets={offsets}, expected []")

# ---------------------------------------------------------------------------
# Test 3: Multiple matches (ERROR at 10, 100, 200 in 256 bytes)
# ---------------------------------------------------------------------------
print("\n=== Test 3: Multiple matches ===")

needle = b"ERROR"
pt = bytearray(256)
for pos in [10, 100, 200]:
    pt[pos:pos + len(needle)] = needle
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("three matches found", sorted(offsets) == [10, 100, 200],
      f"got offsets={sorted(offsets)}, expected [10, 100, 200]")

# ---------------------------------------------------------------------------
# Test 4: Boundary positions (ERROR at byte 0 and last possible byte 123)
# ---------------------------------------------------------------------------
print("\n=== Test 4: Boundary positions ===")

needle = b"ERROR"
n = 128
pt = bytearray(n)
pt[0:len(needle)] = needle
pt[123:123 + len(needle)] = needle  # last position: 123 + 5 = 128
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("matches at byte 0 and 123", sorted(offsets) == [0, 123],
      f"got offsets={sorted(offsets)}, expected [0, 123]")

# ---------------------------------------------------------------------------
# Test 5: Cross-block boundary (ERROR straddling bytes 62-66 in 256 bytes)
# ---------------------------------------------------------------------------
print("\n=== Test 5: Cross-block boundary ===")

needle = b"ERROR"
pt = bytearray(256)
# Place needle so it straddles the 64-byte block boundary (bytes 62..66)
pt[62:62 + len(needle)] = needle
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("cross-block match at byte 62", offsets == [62],
      f"got offsets={offsets}, expected [62]")

# ---------------------------------------------------------------------------
# Test 6: Cross-iteration boundary (ERROR straddling bytes 254-258 in 512 bytes)
# ---------------------------------------------------------------------------
print("\n=== Test 6: Cross-iteration boundary ===")

needle = b"ERROR"
pt = bytearray(512)
# Straddle the 256-byte iteration boundary
pt[254:254 + len(needle)] = needle
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("cross-iteration match at byte 254", offsets == [254],
      f"got offsets={offsets}, expected [254]")

# ---------------------------------------------------------------------------
# Test 7: Single-byte needle (X at 0, 32, 63 in 64 bytes)
# ---------------------------------------------------------------------------
print("\n=== Test 7: Single-byte needle ===")

needle = b"X"
pt = bytearray(64)
for pos in [0, 32, 63]:
    pt[pos] = ord("X")
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("single-byte needle matches at 0, 32, 63", sorted(offsets) == [0, 32, 63],
      f"got offsets={sorted(offsets)}, expected [0, 32, 63]")

# ---------------------------------------------------------------------------
# Test 8: Max needle length 64 (64 A's at offset 100 in 256 bytes)
# ---------------------------------------------------------------------------
print("\n=== Test 8: Max needle length 64 ===")

needle = b"A" * 64
pt = bytearray(256)
pt[100:100 + 64] = needle
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("64-byte needle match at offset 100", offsets == [100],
      f"got offsets={offsets}, expected [100]")

# ---------------------------------------------------------------------------
# Test 9: Overlapping matches ("aaaa" at byte 10 → "aa" matches at 10, 11, 12)
# ---------------------------------------------------------------------------
print("\n=== Test 9: Overlapping matches ===")

needle = b"aa"
pt = bytearray(64)
pt[10:14] = b"aaaa"
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("overlapping matches at 10, 11, 12", sorted(offsets) == [10, 11, 12],
      f"got offsets={sorted(offsets)}, expected [10, 11, 12]")

# ---------------------------------------------------------------------------
# Test 10: Cross-verification random data
# ---------------------------------------------------------------------------
print("\n=== Test 10: Cross-verification with random data ===")

random.seed(99)
needle = b"FIND"

for size in [64, 128, 256, 512, 1024, 4096]:
    pt = bytearray(random.randint(0, 255) for _ in range(size))
    # Inject needle at a few known positions (avoid tail that would be cut off)
    inject_positions = []
    for pos in [0, size // 3, size // 2, size - len(needle)]:
        if 0 <= pos <= size - len(needle):
            pt[pos:pos + len(needle)] = needle
            inject_positions.append(pos)

    ct_bytes = encrypt_data(bytes(pt))
    offsets = search_ciphertext(ct_bytes, needle)

    # Cross-verify against Python reference
    expected = find_all_occurrences(bytes(pt), needle)

    check(f"size={size}: matches kernel==python ref",
          sorted(offsets) == sorted(expected),
          f"kernel={sorted(offsets)}, python={sorted(expected)}")

# ---------------------------------------------------------------------------
# Test 11: Realistic log data
# ---------------------------------------------------------------------------
print("\n=== Test 11: Realistic log data ===")

random.seed(7)
log_lines = []
error_offsets = []
current_offset = 0
error_marker = b"ERROR"

for i in range(200):
    if random.random() < 0.10:
        line = f"[2026-03-20 12:{i % 60:02d}:00] ERROR Something went wrong #{i}\n".encode()
        error_offsets.append(current_offset + line.index(error_marker))
    else:
        line = f"[2026-03-20 12:{i % 60:02d}:00] INFO  Normal operation #{i}\n".encode()
    log_lines.append(line)
    current_offset += len(line)

plaintext = b"".join(log_lines)
ct_bytes = encrypt_data(plaintext)
offsets = search_ciphertext(ct_bytes, error_marker, max_matches=len(error_offsets) + 10)

# Cross-verify
expected = find_all_occurrences(plaintext, error_marker)
check("log data: kernel matches python ref",
      sorted(offsets) == sorted(expected),
      f"kernel={sorted(offsets)[:5]}..., python={sorted(expected)[:5]}...")
check("log data: ~10% ERROR lines found",
      5 <= len(offsets) <= 30,
      f"found {len(offsets)} ERROR occurrences in 200 lines")

# ---------------------------------------------------------------------------
# Test 12: Size sweep with "AB" needle at offset 0
# ---------------------------------------------------------------------------
print("\n=== Test 12: Size sweep ===")

needle = b"AB"
for size in [0, 1, 15, 16, 63, 64, 65, 127, 128, 255, 256, 257, 1000, 4096, 1048576]:
    if size < len(needle):
        # Can't place needle, expect 0 matches
        if size == 0:
            ct_bytes = b""
        else:
            ct_bytes = encrypt_data(bytes(size))
        offsets = search_ciphertext(ct_bytes, needle)
        check(f"size={size}: 0 matches (too small)", offsets == [],
              f"got offsets={offsets}")
    else:
        pt = bytearray(size)
        pt[0:2] = needle
        ct_bytes = encrypt_data(bytes(pt))
        offsets = search_ciphertext(ct_bytes, needle, max_matches=100)
        expected = find_all_occurrences(bytes(pt), needle)
        check(f"size={size}: kernel matches python ref",
              sorted(offsets) == sorted(expected),
              f"kernel={sorted(offsets)[:5]}, python={sorted(expected)[:5]}")

# ---------------------------------------------------------------------------
# Test 13: max_matches overflow (256 A's, search for "A" with max_matches=5)
# ---------------------------------------------------------------------------
print("\n=== Test 13: max_matches overflow ===")

needle = b"A"
pt = b"A" * 256
ct_bytes = encrypt_data(pt)
offsets = search_ciphertext(ct_bytes, needle, max_matches=5)

check("max_matches=5 limits results to 5", len(offsets) == 5,
      f"got {len(offsets)} matches, expected 5")
check("max_matches=5 results are first 5 offsets", sorted(offsets) == list(range(5)),
      f"got offsets={sorted(offsets)}, expected {list(range(5))}")

# ---------------------------------------------------------------------------
# Test 14: Empty needle (needle_len=0 → 0 matches)
# ---------------------------------------------------------------------------
print("\n=== Test 14: Empty needle ===")

pt = b"Hello World"
ct_bytes = encrypt_data(pt)
offsets = search_ciphertext(ct_bytes, b"")

check("empty needle: 0 matches", offsets == [],
      f"got offsets={offsets}, expected []")

# ---------------------------------------------------------------------------
# Test 15: Zero-length input (0 bytes ciphertext → 0 matches)
# ---------------------------------------------------------------------------
print("\n=== Test 15: Zero-length input ===")

offsets = search_ciphertext(b"", b"ERROR")

check("zero-length input: 0 matches", offsets == [],
      f"got offsets={offsets}, expected []")

# ---------------------------------------------------------------------------
# Test 16: Needle too long (65 bytes → 0 matches)
# ---------------------------------------------------------------------------
print("\n=== Test 16: Needle too long (65 bytes) ===")

needle = b"B" * 65
pt = bytearray(256)
# Even if we embed the pattern, a 65-byte needle should yield 0 matches
pt[0:65] = needle
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("65-byte needle: 0 matches", offsets == [],
      f"got offsets={offsets}, expected []")

# ---------------------------------------------------------------------------
# Test 17: Tier 2→Tier 3 boundary (ERROR at byte 318 in 350-byte buffer)
# ---------------------------------------------------------------------------
print("\n=== Test 17: Tier 2→Tier 3 boundary ===")

needle = b"ERROR"
# 350 bytes total; tier boundary at 320 (5 * 64); place needle at 318..322
pt = bytearray(350)
pt[318:318 + len(needle)] = needle
ct_bytes = encrypt_data(bytes(pt))
offsets = search_ciphertext(ct_bytes, needle)

check("Tier2→Tier3 boundary match at byte 318", offsets == [318],
      f"got offsets={offsets}, expected [318]")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 50}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All tests passed!")
