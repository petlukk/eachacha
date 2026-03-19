"""RFC 7539 test vectors for ChaCha20 Eä kernel."""
import ctypes as ct
import numpy as np
import struct
import sys
import os

# ---------------------------------------------------------------------------
# Load .so directly via ctypes (bypass numpy wrapper for precise control)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ct.CDLL(os.path.join(_HERE, "chacha20.so"))

_lib.chacha20_block.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
    ct.c_int32, ct.POINTER(ct.c_int32),
]
_lib.chacha20_block.restype = None

_lib.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
]
_lib.chacha20_encrypt.restype = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_i32_array(values):
    """Convert a list of unsigned 32-bit ints to a ctypes c_int32 array."""
    arr = (ct.c_int32 * len(values))()
    for i, v in enumerate(values):
        arr[i] = ct.c_int32(v & 0xFFFFFFFF).value
    return arr


def i32_array_to_u32_list(arr, n):
    """Read n i32 values back as unsigned 32-bit ints."""
    return [arr[i] & 0xFFFFFFFF for i in range(n)]


def u32_to_le_bytes(words):
    """Convert list of u32 words to bytes (little-endian)."""
    return b"".join(struct.pack("<I", w) for w in words)


def key_bytes_from_u32(key_u32):
    return u32_to_le_bytes(key_u32)


def nonce_bytes_from_u32(nonce_u32):
    return u32_to_le_bytes(nonce_u32)


def make_scratch():
    """Create dual-pointer scratch buffer for chacha20_encrypt."""
    scratch = (ct.c_uint8 * 64)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    return scratch, ks_i32, ks_u8


# ===========================================================================
# RFC 7539 Test Data
# ===========================================================================

KEY_U32 = [
    0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
    0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
]

# Section 2.3.2 block test
BLOCK_NONCE_U32 = [0x09000000, 0x4a000000, 0x00000000]
BLOCK_COUNTER = 1

EXPECTED_BLOCK_U32 = [
    0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3,
    0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
    0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9,
    0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2,
]

# Section 2.4.2 encryption test
ENC_NONCE_U32 = [0x00000000, 0x4a000000, 0x00000000]
ENC_COUNTER = 1

PLAINTEXT = (b"Ladies and Gentlemen of the class of '99: "
             b"If I could offer you only one tip for the future, "
             b"sunscreen would be it.")

EXPECTED_CIPHERTEXT_HEX = (
    "6e2e359a2568f98041ba0728dd0d6981"
    "e97e7aec1d4360c20a27afccfd9fae0b"
    "f91b65c5524733ab8f593dabcd62b357"
    "1639d624e65152ab8f530c359f0861d8"
    "07ca0dbf500d6a6156a38e088a22b65e"
    "52bc514d16ccf806818ce91ab7793736"
    "5af90bbf74a35be6b40b8eedf2785e42"
    "874d"
)

# ===========================================================================
# Tests
# ===========================================================================

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
# Test 1: Block function (RFC 7539 Section 2.3.2)
# ---------------------------------------------------------------------------
print("=== Test 1: chacha20_block (RFC 7539 s2.3.2) ===")

key = to_i32_array(KEY_U32)
nonce = to_i32_array(BLOCK_NONCE_U32)
out = (ct.c_int32 * 16)()

_lib.chacha20_block(key, nonce, ct.c_int32(BLOCK_COUNTER), out)

got = i32_array_to_u32_list(out, 16)
expected = EXPECTED_BLOCK_U32

if got != expected:
    print("  DIAGNOSTIC — block output mismatch:")
    for i in range(16):
        flag = " <-- MISMATCH" if got[i] != expected[i] else ""
        print(f"    [{i:2d}] got 0x{got[i]:08x}  expected 0x{expected[i]:08x}{flag}")

check("block output matches RFC 7539 s2.3.2", got == expected)

# ---------------------------------------------------------------------------
# Test 2: Encryption (RFC 7539 Section 2.4.2)
# ---------------------------------------------------------------------------
print("\n=== Test 2: chacha20_encrypt (RFC 7539 s2.4.2) ===")

key_enc = to_i32_array(KEY_U32)
nonce_enc = to_i32_array(ENC_NONCE_U32)
pt_len = len(PLAINTEXT)

pt_buf = (ct.c_uint8 * pt_len)(*PLAINTEXT)
ct_buf = (ct.c_uint8 * pt_len)()
scratch, ks_i32, ks_u8 = make_scratch()
pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))

_lib.chacha20_encrypt(
    key_enc, nonce_enc, ct.c_int32(ENC_COUNTER),
    pt_buf, ct_buf, ct.c_int32(pt_len),
    ks_i32, ks_u8, pt_i32, ct_i32,
)

got_ct = bytes(ct_buf)
expected_ct = bytes.fromhex(EXPECTED_CIPHERTEXT_HEX)

if got_ct != expected_ct:
    print("  DIAGNOSTIC — ciphertext mismatch:")
    print(f"    got:      {got_ct.hex()}")
    print(f"    expected: {expected_ct.hex()}")
    # Find first differing byte
    for i in range(min(len(got_ct), len(expected_ct))):
        if got_ct[i] != expected_ct[i]:
            print(f"    first diff at byte {i}: got 0x{got_ct[i]:02x} expected 0x{expected_ct[i]:02x}")
            break

check("ciphertext matches RFC 7539 s2.4.2", got_ct == expected_ct)

# Also verify decrypt (encrypt again = decrypt, since XOR is its own inverse)
pt_buf2 = (ct.c_uint8 * pt_len)(*got_ct)
dec_buf = (ct.c_uint8 * pt_len)()
scratch2, ks_i32_2, ks_u8_2 = make_scratch()
pt_i32_2 = ct.cast(pt_buf2, ct.POINTER(ct.c_int32))
ct_i32_2 = ct.cast(dec_buf, ct.POINTER(ct.c_int32))

_lib.chacha20_encrypt(
    to_i32_array(KEY_U32), to_i32_array(ENC_NONCE_U32), ct.c_int32(ENC_COUNTER),
    pt_buf2, dec_buf, ct.c_int32(pt_len),
    ks_i32_2, ks_u8_2, pt_i32_2, ct_i32_2,
)
check("decrypt(encrypt(pt)) == pt", bytes(dec_buf) == PLAINTEXT)

# ---------------------------------------------------------------------------
# Test 3: Cross-verification with OpenSSL (cryptography library)
# ---------------------------------------------------------------------------
print("\n=== Test 3: Cross-verification with OpenSSL ===")

try:
    from cryptography.hazmat.primitives.ciphers import Cipher
    from cryptography.hazmat.primitives.ciphers.algorithms import ChaCha20

    # cryptography lib ChaCha20 nonce = 16 bytes = counter_le(4) + nonce(12)
    key_bytes = key_bytes_from_u32(KEY_U32)
    nonce_12 = nonce_bytes_from_u32(ENC_NONCE_U32)
    counter_bytes = struct.pack("<I", ENC_COUNTER)
    full_nonce = counter_bytes + nonce_12  # 16 bytes

    # 3a: Encrypt with Eä, decrypt with OpenSSL
    cipher = Cipher(ChaCha20(key_bytes, full_nonce), mode=None)
    decryptor = cipher.decryptor()
    ossl_decrypted = decryptor.update(got_ct) + decryptor.finalize()
    check("Eä encrypt -> OpenSSL decrypt == plaintext", ossl_decrypted == PLAINTEXT,
          f"got: {ossl_decrypted!r}")

    # 3b: Encrypt with OpenSSL, decrypt with Eä
    cipher2 = Cipher(ChaCha20(key_bytes, full_nonce), mode=None)
    encryptor = cipher2.encryptor()
    ossl_ct = encryptor.update(PLAINTEXT) + encryptor.finalize()

    pt_buf3 = (ct.c_uint8 * pt_len)(*ossl_ct)
    dec_buf3 = (ct.c_uint8 * pt_len)()
    scratch3, ks_i32_3, ks_u8_3 = make_scratch()
    pt_i32_3 = ct.cast(pt_buf3, ct.POINTER(ct.c_int32))
    ct_i32_3 = ct.cast(dec_buf3, ct.POINTER(ct.c_int32))

    _lib.chacha20_encrypt(
        to_i32_array(KEY_U32), to_i32_array(ENC_NONCE_U32), ct.c_int32(ENC_COUNTER),
        pt_buf3, dec_buf3, ct.c_int32(pt_len),
        ks_i32_3, ks_u8_3, pt_i32_3, ct_i32_3,
    )
    check("OpenSSL encrypt -> Eä decrypt == plaintext", bytes(dec_buf3) == PLAINTEXT,
          f"got: {bytes(dec_buf3)!r}")

    # 3c: Both produce same ciphertext
    check("Eä ciphertext == OpenSSL ciphertext", got_ct == ossl_ct,
          f"ea: {got_ct.hex()}\nossl: {ossl_ct.hex()}")

except ImportError:
    print("  SKIP  cryptography library not installed")

# ---------------------------------------------------------------------------
# Test 4: C reference produces same ciphertext as Eä
# ---------------------------------------------------------------------------
print("\n=== Test 4: C reference vs Eä ===")

ref_path = os.path.join(_HERE, "libchacha20_ref.so")
if os.path.exists(ref_path):
    _ref = ct.CDLL(ref_path)
    _ref.chacha20_encrypt_ref.argtypes = [
        ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint32), ct.c_uint32,
        ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ]
    _ref.chacha20_encrypt_ref.restype = None

    ref_key = (ct.c_uint32 * 8)(*KEY_U32)
    ref_nonce = (ct.c_uint32 * 3)(*ENC_NONCE_U32)
    ref_pt = (ct.c_uint8 * pt_len)(*PLAINTEXT)
    ref_ct = (ct.c_uint8 * pt_len)()

    _ref.chacha20_encrypt_ref(ref_key, ref_nonce, ct.c_uint32(ENC_COUNTER),
                              ref_pt, ref_ct, ct.c_int32(pt_len))

    ref_ct_bytes = bytes(ref_ct)
    check("C ref ciphertext matches RFC 7539 s2.4.2", ref_ct_bytes == expected_ct,
          f"got: {ref_ct_bytes.hex()}")
    check("C ref ciphertext == Eä ciphertext", ref_ct_bytes == got_ct)
else:
    print("  SKIP  libchacha20_ref.so not found (run: cc -O3 -shared -fPIC -o libchacha20_ref.so chacha20_ref.c)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All tests passed!")
