"""Verify fused ChaCha20 encrypt+stats kernel produces identical ciphertext
and correct statistics compared to numpy."""
import ctypes as ct
import numpy as np
import struct
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load both libraries
_lib = ct.CDLL(os.path.join(_HERE, "chacha20.so"))
_fused = ct.CDLL(os.path.join(_HERE, "chacha20_fused.so"))

# Set up argtypes for non-fused
_lib.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
]
_lib.chacha20_encrypt.restype = None

# Set up argtypes for fused (14 params: key, nonce, counter, pt, ct, len,
#   ks_i32, ks_u8, pt_i32, ct_i32, out_sum, out_count, out_min, out_max)
_fused.chacha20_encrypt_stats.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_int64), ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8),
]
_fused.chacha20_encrypt_stats.restype = None


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


KEY_U32 = [
    0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
    0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
]
ENC_NONCE_U32 = [0x00000000, 0x4a000000, 0x00000000]
ENC_COUNTER = 1

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


# ---------------------------------------------------------------------------
# Test 1: RFC 7539 plaintext — fused ciphertext == non-fused ciphertext
# ---------------------------------------------------------------------------
print("=== Test 1: Fused ciphertext matches non-fused (RFC 7539 plaintext) ===")

PLAINTEXT = (b"Ladies and Gentlemen of the class of '99: "
             b"If I could offer you only one tip for the future, "
             b"sunscreen would be it.")
pt_len = len(PLAINTEXT)

# Non-fused
key1 = to_i32_array(KEY_U32)
nonce1 = to_i32_array(ENC_NONCE_U32)
pt_buf1 = (ct.c_uint8 * pt_len)(*PLAINTEXT)
ct_buf1 = (ct.c_uint8 * pt_len)()
s1, ks1_i32, ks1_u8 = make_scratch()

pt1_i32 = ct.cast(pt_buf1, ct.POINTER(ct.c_int32))
ct1_i32 = ct.cast(ct_buf1, ct.POINTER(ct.c_int32))
_lib.chacha20_encrypt(key1, nonce1, ct.c_int32(ENC_COUNTER),
                      pt_buf1, ct_buf1, ct.c_int32(pt_len), ks1_i32, ks1_u8,
                      pt1_i32, ct1_i32)

# Fused
key2 = to_i32_array(KEY_U32)
nonce2 = to_i32_array(ENC_NONCE_U32)
pt_buf2 = (ct.c_uint8 * pt_len)(*PLAINTEXT)
ct_buf2 = (ct.c_uint8 * pt_len)()
s2, ks2_i32, ks2_u8 = make_scratch()
pt2_i32 = ct.cast(pt_buf2, ct.POINTER(ct.c_int32))
ct2_i32 = ct.cast(ct_buf2, ct.POINTER(ct.c_int32))
out_sum = (ct.c_int64 * 1)()
out_count = (ct.c_int32 * 1)()
out_min = (ct.c_uint8 * 1)()
out_max = (ct.c_uint8 * 1)()

_fused.chacha20_encrypt_stats(
    key2, nonce2, ct.c_int32(ENC_COUNTER),
    pt_buf2, ct_buf2, ct.c_int32(pt_len), ks2_i32, ks2_u8,
    pt2_i32, ct2_i32,
    ct.cast(out_sum, ct.POINTER(ct.c_int64)),
    ct.cast(out_count, ct.POINTER(ct.c_int32)),
    ct.cast(out_min, ct.POINTER(ct.c_uint8)),
    ct.cast(out_max, ct.POINTER(ct.c_uint8)),
)

nonfused_ct = bytes(ct_buf1)
fused_ct = bytes(ct_buf2)

check("ciphertext matches", nonfused_ct == fused_ct,
      f"first diff at byte {next((i for i in range(min(len(nonfused_ct), len(fused_ct))) if nonfused_ct[i] != fused_ct[i]), 'N/A')}" if nonfused_ct != fused_ct else "")

# ---------------------------------------------------------------------------
# Test 2: Stats match numpy
# ---------------------------------------------------------------------------
print("\n=== Test 2: Stats match numpy ===")

pt_np = np.frombuffer(PLAINTEXT, dtype=np.uint8)
expected_sum = int(pt_np.sum())
expected_count = len(PLAINTEXT)
expected_min = int(pt_np.min())
expected_max = int(pt_np.max())

got_sum = out_sum[0]
got_count = out_count[0]
got_min = out_min[0]
got_max = out_max[0]

check("sum matches numpy", got_sum == expected_sum,
      f"got {got_sum}, expected {expected_sum}")
check("count matches", got_count == expected_count,
      f"got {got_count}, expected {expected_count}")
check("min matches numpy", got_min == expected_min,
      f"got {got_min}, expected {expected_min}")
check("max matches numpy", got_max == expected_max,
      f"got {got_max}, expected {expected_max}")

# ---------------------------------------------------------------------------
# Test 3: Various sizes (boundary conditions)
# ---------------------------------------------------------------------------
print("\n=== Test 3: Various input sizes ===")

import random
random.seed(42)

for size in [0, 1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 256, 1000]:
    if size == 0:
        # Zero-length: just ensure no crash
        key_t = to_i32_array(KEY_U32)
        nonce_t = to_i32_array(ENC_NONCE_U32)
        pt_t = (ct.c_uint8 * 1)()  # dummy
        ct_t1 = (ct.c_uint8 * 1)()
        ct_t2 = (ct.c_uint8 * 1)()
        s_t1, ks_t1_i32, ks_t1_u8 = make_scratch()
        s_t2, ks_t2_i32, ks_t2_u8 = make_scratch()
        o_sum = (ct.c_int64 * 1)()
        o_count = (ct.c_int32 * 1)()
        o_min = (ct.c_uint8 * 1)(255)
        o_max = (ct.c_uint8 * 1)(0)

        pt_t_i32 = ct.cast(pt_t, ct.POINTER(ct.c_int32))
        ct_t1_i32 = ct.cast(ct_t1, ct.POINTER(ct.c_int32))
        _lib.chacha20_encrypt(to_i32_array(KEY_U32), to_i32_array(ENC_NONCE_U32),
                              ct.c_int32(ENC_COUNTER), pt_t, ct_t1, ct.c_int32(0),
                              ks_t1_i32, ks_t1_u8, pt_t_i32, ct_t1_i32)
        pt_t2_i32 = ct.cast(pt_t, ct.POINTER(ct.c_int32))
        ct_t2_i32 = ct.cast(ct_t2, ct.POINTER(ct.c_int32))
        _fused.chacha20_encrypt_stats(
            to_i32_array(KEY_U32), to_i32_array(ENC_NONCE_U32), ct.c_int32(ENC_COUNTER),
            pt_t, ct_t2, ct.c_int32(0), ks_t2_i32, ks_t2_u8,
            pt_t2_i32, ct_t2_i32,
            ct.cast(o_sum, ct.POINTER(ct.c_int64)),
            ct.cast(o_count, ct.POINTER(ct.c_int32)),
            ct.cast(o_min, ct.POINTER(ct.c_uint8)),
            ct.cast(o_max, ct.POINTER(ct.c_uint8)),
        )
        check(f"size={size}: no crash, count=0", o_count[0] == 0,
              f"count={o_count[0]}")
        continue

    data = bytes(random.randint(0, 255) for _ in range(size))
    pt_arr = (ct.c_uint8 * size)(*data)

    # Non-fused
    ct_nf = (ct.c_uint8 * size)()
    s_nf, ks_nf_i32, ks_nf_u8 = make_scratch()
    pt_nf_buf = (ct.c_uint8 * size)(*data)
    pt_nf_i32 = ct.cast(pt_nf_buf, ct.POINTER(ct.c_int32))
    ct_nf_i32 = ct.cast(ct_nf, ct.POINTER(ct.c_int32))
    _lib.chacha20_encrypt(to_i32_array(KEY_U32), to_i32_array(ENC_NONCE_U32),
                          ct.c_int32(ENC_COUNTER),
                          pt_nf_buf, ct_nf, ct.c_int32(size),
                          ks_nf_i32, ks_nf_u8, pt_nf_i32, ct_nf_i32)

    # Fused
    ct_f = (ct.c_uint8 * size)()
    s_f, ks_f_i32, ks_f_u8 = make_scratch()
    pt_f_buf = (ct.c_uint8 * size)(*data)
    pt_f_i32 = ct.cast(pt_f_buf, ct.POINTER(ct.c_int32))
    ct_f_i32 = ct.cast(ct_f, ct.POINTER(ct.c_int32))
    o_s = (ct.c_int64 * 1)()
    o_c = (ct.c_int32 * 1)()
    o_mn = (ct.c_uint8 * 1)(255)
    o_mx = (ct.c_uint8 * 1)(0)
    _fused.chacha20_encrypt_stats(
        to_i32_array(KEY_U32), to_i32_array(ENC_NONCE_U32), ct.c_int32(ENC_COUNTER),
        pt_f_buf, ct_f, ct.c_int32(size), ks_f_i32, ks_f_u8,
        pt_f_i32, ct_f_i32,
        ct.cast(o_s, ct.POINTER(ct.c_int64)),
        ct.cast(o_c, ct.POINTER(ct.c_int32)),
        ct.cast(o_mn, ct.POINTER(ct.c_uint8)),
        ct.cast(o_mx, ct.POINTER(ct.c_uint8)),
    )

    ct_ok = bytes(ct_nf) == bytes(ct_f)
    np_data = np.frombuffer(data, dtype=np.uint8)
    stats_ok = (o_s[0] == int(np_data.sum()) and
                o_c[0] == size and
                o_mn[0] == int(np_data.min()) and
                o_mx[0] == int(np_data.max()))

    detail = ""
    if not ct_ok:
        detail += "ciphertext mismatch "
    if not stats_ok:
        detail += (f"sum={o_s[0]}(exp {int(np_data.sum())}) "
                   f"count={o_c[0]}(exp {size}) "
                   f"min={o_mn[0]}(exp {int(np_data.min())}) "
                   f"max={o_mx[0]}(exp {int(np_data.max())})")
    check(f"size={size}: ciphertext+stats", ct_ok and stats_ok, detail)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All tests passed!")
